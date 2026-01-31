#!/usr/bin/env python3
"""
Export GPT-2 model from HuggingFace to TorchScript format.

This script downloads a GPT-2 model from HuggingFace Hub and exports it
to TorchScript format that can be loaded in C++ via LibTorch.

Usage:
    python export_gpt2.py --model gpt2 --output gpt2.pt
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import json


def export_gpt2_to_torchscript(model_name: str, output_path: str, max_seq_length: int = 128):
    """
    Export a GPT-2 model to TorchScript format.
    
    Args:
        model_name: Name of the HuggingFace model (e.g., 'gpt2')
        output_path: Path to save the TorchScript model
        max_seq_length: Maximum sequence length for the model
    """
    print(f"Loading GPT-2 model: {model_name}")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded. Config:")
    print(f"  - Hidden size: {model.config.n_embd}")
    print(f"  - Num layers: {model.config.n_layer}")
    print(f"  - Num attention heads: {model.config.n_head}")
    print(f"  - Vocab size: {model.config.vocab_size}")
    
    # Create dummy input for tracing
    batch_size = 1
    dummy_input_ids = torch.ones(batch_size, max_seq_length, dtype=torch.long)
    
    print(f"\nTracing model with input shape: {dummy_input_ids.shape}")
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, 
            dummy_input_ids,
            strict=False
        )
    
    print(f"Model traced successfully")
    
    # Save the traced model
    print(f"Saving TorchScript model to: {output_path}")
    traced_model.save(output_path)
    
    # Verify the saved model can be loaded
    print("Verifying saved model...")
    loaded_model = torch.jit.load(output_path)
    
    # Test inference
    with torch.no_grad():
        output = loaded_model(dummy_input_ids)
        logits = output[0] if isinstance(output, tuple) else output.logits
        print(f"Verification successful! Logits shape: {logits.shape}")
    
    # Save model config as JSON for reference
    config_path = output_path.replace('.pt', '_config.json')
    config_dict = {
        'model_name': model_name,
        'hidden_size': model.config.n_embd,
        'num_hidden_layers': model.config.n_layer,
        'num_attention_heads': model.config.n_head,
        'vocab_size': model.config.vocab_size,
        'max_position_embeddings': model.config.n_positions,
        'max_seq_length': max_seq_length
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    # Save tokenizer
    tokenizer_dir = output_path.replace('.pt', '_tokenizer')
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"Saved tokenizer to: {tokenizer_dir}")
    
    print("\nExport complete!")
    print(f"  Model: {output_path}")
    print(f"  Config: {config_path}")
    print(f"  Tokenizer: {tokenizer_dir}")


def main():
    parser = argparse.ArgumentParser(description='Export GPT-2 model to TorchScript')
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        help='HuggingFace model name (default: gpt2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='gpt2.pt',
        help='Output path for TorchScript model (default: gpt2.pt)'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=128,
        help='Maximum sequence length (default: 128)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    export_gpt2_to_torchscript(args.model, args.output, args.max_seq_length)


if __name__ == '__main__':
    main()
