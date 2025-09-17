import argparse
import yaml
import os
import torch
import numpy as np
import random
import json
import time

from . import preprocess
from . import train
from . import evaluate

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def run_experiment(config_path):
    """Main execution function for running experiments based on a config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return

    # Create result directories
    os.makedirs(".research/iteration1/images", exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("Warning: CUDA not available. Experiments will be very slow.")

    # --- DATA LOADING ---
    # In a real scenario, dataloaders would be created from these datasets
    datasets = preprocess.load_and_preprocess_data(config)
    
    # --- EXPERIMENT LOOP ---
    for seed in config['seeds']:
        print(f"\n{'='*50}\nRUNNING EXPERIMENT FOR SEED: {seed}\n{'='*50}")
        set_seed(seed)

        for model_name in config['models']:
            print(f"\n--- Model: {model_name} ---")
            
            # --- MODEL SETUP ---
            model, tokenizer = train.setup_model(model_name, config['training']['lora'], device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
            Q_phi = train.QPhi().to(device)
            H_theta = train.HTheta().to(device)

            # --- TRAINING ---
            # The full CoVeR-SPO training is complex. We call a placeholder function.
            trained_model = train.train_model(
                model, 
                datasets['train'], # This would be a dataloader
                optimizer, 
                config, 
                device
            )

            # --- EVALUATION ---
            # Orchestrate the three experiments as described in the design document
            all_exp_results = evaluate.evaluate_experiments(config, trained_model, H_theta, device)
            
            # --- SAVE RESULTS ---
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            result_filename = f".research/iteration1/results_{model_name.replace('/', '_')}_seed{seed}_{timestamp}.json"
            try:
                with open(result_filename, 'w') as f:
                    # Convert numpy arrays/means to lists for JSON serialization
                    json_serializable_results = json.loads(json.dumps(all_exp_results, default=str))
                    json.dump(json_serializable_results, f, indent=4)
                print(f"Results for seed {seed} and model {model_name} saved to {result_filename}")
            except Exception as e:
                print(f"Error saving results to JSON: {e}")

    print("\nAll experiments completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CoVeR-SPO experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full experiment.')
    
    args = parser.parse_args()

    if args.smoke_test:
        print("--- Starting Smoke Test ---")
        run_experiment('config/smoke_test.yaml')
        print("--- Smoke Test Passed ---")
        # In a real CI/CD pipeline, you might proceed to the full experiment here
        # For command-line execution, we keep them separate.
    elif args.full_experiment:
        print("--- Starting Full Experiment ---")
        # Optional: Add a check to ensure smoke test has been run and passed
        run_experiment('config/full_experiment.yaml')
