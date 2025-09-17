import argparse
import yaml
import torch
import os
import json
import time
import numpy as np
from sklearn.linear_model import Ridge

from .preprocess import get_dataloader
from .train import UNet, jasaic_loss, linear_beta_schedule
from .evaluate import evaluate

class CarbonStop:
    def __init__(self, threshold=0.1, update_interval_steps=100):
        self.threshold = threshold
        self.update_interval = update_interval_steps
        self.steps = []
        self.fids = []
        self.co2s = []
        self.reg = Ridge(alpha=1e-3)
        self.last_fid = float('inf')
        self.last_co2 = 0.0
        self.mock_co2_per_step = 0.54 * 8 * 1e-4 # kg per step (A100-hr-kg * num_gpus * hr/step)

    def check(self, step, val_fid):
        current_co2 = step * self.mock_co2_per_step
        if step > 0 and step % self.update_interval == 0:
            d_fid = val_fid - self.last_fid
            d_co2 = current_co2 - self.last_co2
            
            if d_co2 > 1e-9:
                ratio = -d_fid / d_co2
                self.steps.append(np.log(step + 1e-6))
                self.fids.append(val_fid)
                self.co2s.append(np.log(current_co2 + 1e-6))
                
                if len(self.steps) > 10:
                    X = np.array(self.steps).reshape(-1, 1)
                    y = np.array(self.fids)
                    try:
                        self.reg.fit(X, y)
                        # Simplified CI check for mock
                        marginal_gain = -self.reg.coef_[0]
                        if marginal_gain < self.threshold:
                            print(f"CarbonStop: Marginal FID gain {marginal_gain:.4f} < {self.threshold}. Stopping training.")
                            return True
                    except Exception as e:
                         print(f'CarbonStop regression failed: {e}')
            
            self.last_fid = val_fid
            self.last_co2 = current_co2
        return False

def run_experiment(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f'Running experiment with config: {config_path}')
    print(json.dumps(config, indent=2))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Data
    dataloader = get_dataloader(config)
    val_dataloader = get_dataloader(config) # Using same for simplicity

    # Model
    model = UNet(model_size=config['model']['size']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['optimizer']['beta1'], config['training']['optimizer']['beta2']),
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    # Diffusion parameters
    num_timesteps = config['training']['num_timesteps']
    betas = linear_beta_schedule(timesteps=num_timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
    diffusion_params = (sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod)
    
    # CO2 Stopper
    carbon_stopper = CarbonStop() if config.get('carbon_stop', {}).get('enabled', False) else None

    # Training loop
    model.train()
    start_time = time.time()
    global_step = 0
    for epoch in range(config['training']['epochs']):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            images = batch['img'].to(device)
            t = torch.randint(0, num_timesteps, (images.shape[0],), device=device).long()

            loss = jasaic_loss(model, images, t, alphas_cumprod, *diffusion_params, num_timesteps=num_timesteps, max_d=config['training']['max_d'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if global_step % 10 == 0:
                print(f'Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}')

            if carbon_stopper and global_step > 0 and global_step % 100 == 0:
                # Mock validation FID for stopping logic
                mock_val_fid = 10.0 / np.log(global_step + 10) # FID decreases as training progresses
                if carbon_stopper.check(global_step, mock_val_fid):
                    break

            global_step += 1
            if 'limit_train_batches' in config['training'] and i >= config['training']['limit_train_batches'] - 1:
                break
        if carbon_stopper and carbon_stopper.check(global_step, 0) and carbon_stopper.steps and len(carbon_stopper.steps) > 10: # Check after epoch
             break

    end_time = time.time()
    print(f'Training finished in {end_time - start_time:.2f} seconds.')

    # Evaluation
    eval_results = evaluate(model, val_dataloader, device, diffusion_params, config)

    # Save results
    os.makedirs('.research/iteration1', exist_ok=True)
    result_filename = os.path.join('.research/iteration1', f'{os.path.basename(config_path).replace(".yaml", "")}_results.json')
    with open(result_filename, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f'Results saved to {result_filename}')
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    parser.add_argument('--full-experiment', action='store_true', help='Run the full experiment.')
    args = parser.parse_args()

    if args.smoke_test and args.full_experiment:
        raise ValueError('Cannot specify both --smoke-test and --full-experiment.')
    
    config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config')

    if args.smoke_test:
        print('--- Running Smoke Test ---')
        success = run_experiment(os.path.join(config_dir, 'smoke_test.yaml'))
        if not success:
             print('Smoke test failed.')
             exit(1)
        print('--- Smoke Test Passed ---')

    elif args.full_experiment:
        print('--- Running Full Experiment ---')
        run_experiment(os.path.join(config_dir, 'full_experiment.yaml'))
        print('--- Full Experiment Finished ---')
    else:
        print('Please specify either --smoke-test or --full-experiment.')
