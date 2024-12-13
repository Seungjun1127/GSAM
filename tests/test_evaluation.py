"""Entire pipeline test"""

import sys
import os
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.GSAM.data.loaders import load_word_dataset
from src.GSAM.models.model_loader import LargeModelLoader
from src.GSAM.metrics.gsam import compute_gsam, batch_compute_gsam, compare_models

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_gsam_pipeline():

    torch.cuda.empty_cache()

    # Load dataset
    dataset = load_word_dataset()
    print("Loaded dataset for testing.")

    # Load model
    model_loader = LargeModelLoader()
    model_name = 'pythia-7b'  # Changed model name to Llama-2-7

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = model_loader.load_model(model_name, device=device)  # Set device

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token

    # Tokenize multiple samples from the dataset
    samples = [dataset[i]['sentence'] for i in range(10)]  # Extract 'sentence' field from the entire dataset
    inputs = tokenizer(samples, return_tensors='pt', padding=True, truncation=True)

    # Extract model's activations
    with torch.no_grad():
        activations = model_loader.extract_activation(model, inputs['input_ids'].to(device), inputs['attention_mask'].to(device))

    # Calculate GSAM
    #metric = 'kl_divergence'  # Define metric variable
    metric = 'neg_log_likelihood'
    gsam_score = compute_gsam(activations.cpu().numpy(), metric=metric)  # Move to CPU and convert to NumPy array
    print(f"GSAM Score: {gsam_score}")

    # Record GSAM results to a text file
    with open("gsam_results.txt", "a") as f:  # Open file in 'a' mode (append mode)
        f.write(f"Model: {model_name}, Metric: {metric}, GSAM Score: {gsam_score}\n")  # Record results

    # Extract the first 10 samples
    samples = dataset[:10]

    # Count the total number of data points in the 'sentence' field
    total_data_points = sum(len(sample['sentence']) for sample in samples)

    print(f"Total data points in the 'sentence' field for the first 10 samples: {total_data_points}")

if __name__ == "__main__":
    test_gsam_pipeline()
