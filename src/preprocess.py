import numpy as np
import unicodedata
from datasets import load_dataset, Dataset

def load_and_preprocess_data(config):
    """
    Loads and preprocesses all necessary datasets based on the configuration.
    Returns a dictionary of placeholder dataloaders.
    """
    print("Loading and preprocessing data (simulated)...")
    # In a real implementation, this would use config['datasets'] to load from Hugging Face Hub or local files.
    # For now, we create dummy datasets.
    datasets = {}
    dummy_prompts = {"prompt": ["This is a test prompt"] * 100}
    
    for name in ['train', 'validation', 'test_calibtail', 'test_medlegchem', 'test_langshift', 'test_ultralong']:
        datasets[name] = Dataset.from_dict(dummy_prompts)

    # Apply unicode normalization as specified
    def normalize_text(example):
        example['prompt'] = unicodedata.normalize('NFC', example['prompt'])
        return example

    for name, ds in datasets.items():
        datasets[name] = ds.map(normalize_text)
    
    print("Data loading and preprocessing complete.")
    return datasets


def distort(p):
    """
    Applies a random piece-wise linear distortion to mis-calibrate a probability.
    This is used in Experiment #3.
    """
    if not isinstance(p, np.ndarray):
        p = np.array(p)
    
    a, b = np.random.uniform(0.8, 1.2, 2)
    distorted_p = a * p + b * (p > 0.5) * (1 - p)
    return np.clip(distorted_p, 0, 1)


class MiscalibratedDetoxify:
    """
    A wrapper to simulate a real mis-calibrated toxicity model.
    """
    def __init__(self):
        # In a real scenario, you would load the 'Detoxify' model here.
        # self.model = Detoxify('original-uncased')
        print("Initialized MiscalibratedDetoxify (simulated). ECE is approx 0.18.")

    def predict(self, text):
        """
        Predicts toxicity scores and applies distortion.
        """
        # raw_preds = self.model.predict(text)['toxicity']
        # Simulate raw predictions
        raw_preds = np.random.rand(len(text) if isinstance(text, list) else 1)
        return distort(raw_preds)
