import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple
import os
import numpy as np
import warnings
import glob

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Environment & global config
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------------------------------------------------------
# Utility functions for I/O
# -----------------------------------------------------------------------------

def load_csv_prompts(file_path: str) -> List[str]:
    """Load prompts from a CSV file."""
    df = pd.read_csv(file_path)
    return df["prompt"].tolist()


def save_hidden_states(
    hidden_states: np.ndarray,
    model_name: str,
    category: str,
    csv_name: str,
    layer_idx: int = -1,
):
    """Save hidden states to .npy file with organized directory structure."""
    model_short_name = model_name.split("/")[-1]
    
    # Create model directory if it doesn't exist
    model_dir = model_short_name
    os.makedirs(model_dir, exist_ok=True)
    
    # Create filename: {category}_{csv_name}.npy
    csv_basename = os.path.splitext(os.path.basename(csv_name))[0]
    filename = f"{category}_{csv_basename}.npy"
    filepath = os.path.join(model_dir, filename)
    
    np.save(filepath, hidden_states)
    print(f"[saved] {filepath} • shape={hidden_states.shape}")

# -----------------------------------------------------------------------------
# Hidden states extraction
# -----------------------------------------------------------------------------

def extract_hidden_state_single(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer_idx: int = -1,
) -> np.ndarray:
    """Extract hidden state from the last token of a single input prompt.
    Returns:
        last_token_hidden: (hidden_dim,) ndarray – hidden state of the last input token
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get the hidden states from the specified layer
        hidden_states_layer = outputs.hidden_states[layer_idx]  # Shape: (1, L, D)
        
        # Get the hidden state of the last token
        last_token_hidden = hidden_states_layer[0, -1, :] # Shape: (D,)
        
        # Move to CPU and convert to numpy
        last_token_hidden = last_token_hidden.detach().cpu().float().numpy()

    return last_token_hidden


def process_csv_file(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    csv_path: str,
    category: str,
    model_name: str,
    layer_idx: int,
):
    """Process a single CSV file and extract hidden states one-by-one."""
    print(f"[processing] {csv_path}")
    
    # Load prompts from CSV
    prompts = load_csv_prompts(csv_path)
    print(f"[loaded] {len(prompts)} prompts from {csv_path}")
    
    # Container for all hidden states
    all_hidden_states = []
    
    # Process one-by-one
    for prompt in tqdm(prompts, desc=f"Processing {category}", unit="prompt"):
        # Extract hidden state for this single prompt
        hidden_state = extract_hidden_state_single(
            model, tokenizer, prompt, layer_idx=layer_idx
        )
        all_hidden_states.append(hidden_state)
    
    # Combine all hidden states
    final_hidden_states = np.vstack(all_hidden_states)
    
    # Save to file
    save_hidden_states(final_hidden_states, model_name, category, csv_path, layer_idx)
    
    return final_hidden_states


def process_category_folder(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    category: str,
    model_path: str,
    layer_idx: int,
):
    """Process all CSV files in a category folder."""
    if os.path.exists(category):
        csv_files = glob.glob(os.path.join(category, "*.csv"))
        print(f"[found] {len(csv_files)} CSV files in {category} folder")
        
        for csv_file in csv_files:
            process_csv_file(
                model, tokenizer, csv_file, category, model_path, layer_idx
            )
    else:
        print(f"[warning] Folder '{category}' not found")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Hidden state extraction from CSV files")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer_idx", type=int, default=-1)
    args = parser.parse_args()

    # Model / tokenizer
    print(f"[load] {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        output_hidden_states=True,
    ).eval()

    print(f"[folders] Using base folders: malicious, benign, cleaned")

    # Process each category
    categories = ["malicious","paraphrased", "benign"]

    
    for category in categories:
        process_category_folder(
            model, tokenizer, category, args.model_path, args.layer_idx
        )

    print("[complete] All hidden states extracted and saved")


if __name__ == "__main__":
    main()