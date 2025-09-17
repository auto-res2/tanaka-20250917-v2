import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os

# This class is a core component of CoVeR-SPO for streaming quantile estimation.
class StreamingQuantile:
    """Calculates a quantile on a stream of data using a sliding window buffer."""
    def __init__(self, q=0.9, max_k=5000):
        if not 0 < q < 1:
            raise ValueError("Quantile 'q' must be between 0 and 1.")
        self.q = q
        self.buf = []
        self.max_k = max_k

    def update(self, v):
        """Adds a new value to the buffer, maintaining the max window size."""
        try:
            self.buf.append(float(v))
            if len(self.buf) > self.max_k:
                self.buf.pop(0)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert {v} to float. Skipping update. Error: {e}")

    def quantile(self):
        """Computes the quantile of the current buffer."""
        if not self.buf:
            return 1.0
        # The k-th order statistic for the (1-q) quantile
        k = int((1 - self.q) * len(self.buf))
        if k >= len(self.buf):
             k = len(self.buf) - 1
        # Return the (k+1)-th largest element, which corresponds to the desired quantile
        return torch.tensor(sorted(self.buf)[-k - 1])


# This function implements the certificate-aware spectral weight calculation.
def spectral_weight(kappa, Q_phi, sq_instance, tau):
    """Calculates the spectral weight, incorporating the conformal risk certificate."""
    if not isinstance(kappa, torch.Tensor):
        kappa = torch.tensor(kappa, dtype=torch.float32)

    qhat = sq_instance.quantile().to(kappa.device)
    # Conformal non-conformity score e_t = 1 if kappa > qhat
    e = (kappa > qhat).float()
    
    # The spectral weight w_t = sigma(Q_phi(kappa) - tau * e_t)
    w = torch.sigmoid(Q_phi(kappa.unsqueeze(0))) - tau * e
    
    # Update the streaming quantile estimator with the new kappa score
    # Detach from graph to prevent backprop through the buffer
    sq_instance.update(kappa.detach().cpu().item())
    return w

# Represents the Q_phi network in the paper, a simple MLP.
class QPhi(nn.Module):
    """A simple MLP to model the spectral preference function Q_phi."""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Represents the H_theta network for the zero-back-prop risk dial.
class HTheta(nn.Module):
    """An MLP to predict affine updates (a, b) for the risk dial q_hat."""
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, alpha_star):
        # H_theta outputs an affine update (a, b)
        return self.net(alpha_star)


def setup_model(model_name, lora_config, device):
    """Loads a base model and tokenizer and applies LoRA configuration."""
    try:
        token = os.getenv("HF_TOKEN", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            token=token
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        
        lora_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config['rank'],
            lora_alpha=lora_config['alpha'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, lora_peft_config)
        model.print_trainable_parameters()
        return model, tokenizer
    except Exception as e:
        print(f"Error setting up model {model_name}: {e}")
        raise

def train_model(model, train_loader, optimizer, config, device):
    """A placeholder for the main training loop (e.g., for RLHF)."""
    # This function would contain the complex logic for CoVeR-SPO training.
    # For this refactoring, we simulate a single training step to show integration.
    print("\n--- Starting Dummy Training Epoch ---")
    model.train()
    # In a real scenario, you'd loop through your dataloader
    # For now, we just print a message.
    print("Model training epoch completed (simulated). The real process would involve a full RLHF loop.")
    print("--- Dummy Training Epoch Finished ---\n")
    return model
