#!/usr/bin/env python3
"""
Export BERT model from HuggingFace to TorchScript format.

This script downloads a BERT model from HuggingFace Hub and exports it
to TorchScript format that can be loaded in C++ via LibTorch.

Usage:
    python export_bert.py --model bert-base-uncased --output bert_base.pt
"""

import argparse
import torch
from transformers import BertModel, BertTokenizer
import os


def export_bert_to_torchscript(model_name: str, output_path: str, max_seq_length: int = 128):
    """
    Export a BERT model to TorchScript format.
    
    Args:
        model_name: Name of the HuggingFace model (e.g., 'bert-base-uncased')
        output_path: Path to save the TorchScript model
        max_seq_length: Maximum sequence length for the model
    """
    print(f"Loading BERT model: {model_name}")
    
    # Load model and tokenizer
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded. Config:")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Num layers: {model.config.num_hidden_layers}")
    print(f"  - Num attention heads: {model.config.num_attention_heads}")
    print(f"  - Vocab size: {model.config.vocab_size}")
    
    # Create dummy input for tracing
    batch_size = 1
    dummy_input_ids = torch.ones(batch_size, max_seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, max_seq_length, dtype=torch.long)
    
    print(f"\nTracing model with input shape: {dummy_input_ids.shape}")
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, 
            (dummy_input_ids, dummy_attention_mask),
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
        output = loaded_model(dummy_input_ids, dummy_attention_mask)
        print(f"Verification successful! Output shape: {output.last_hidden_state.shape}")
    
    # Save model config as JSON for reference
    config_path = output_path.replace('.pt', '_config.json')
    import json
    config_dict = {
        'model_name': model_name,
        'hidden_size': model.config.hidden_size,
        'num_hidden_layers': model.config.num_hidden_layers,
        'num_attention_heads': model.config.num_attention_heads,
        'vocab_size': model.config.vocab_size,
        'max_position_embeddings': model.config.max_position_embeddings,
        'type_vocab_size': model.config.type_vocab_size,
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
    parser = argparse.ArgumentParser(description='Export BERT model to TorchScript')
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-uncased',
        help='HuggingFace model name (default: bert-base-uncased)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='bert_base.pt',
        help='Output path for TorchScript model (default: bert_base.pt)'
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
    
    export_bert_to_torchscript(args.model, args.output, args.max_seq_length)


if __name__ == '__main__':
    main()
