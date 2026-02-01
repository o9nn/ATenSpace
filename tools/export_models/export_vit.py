#!/usr/bin/env python3
"""
Export Vision Transformer (ViT) model from HuggingFace to TorchScript format.

This script downloads a ViT model from HuggingFace Hub and exports it
to TorchScript format that can be loaded in C++ via LibTorch.

Usage:
    python export_vit.py --model google/vit-base-patch16-224 --output vit_base.pt
"""

import argparse
import torch
from transformers import ViTModel, ViTImageProcessor
import os
import json


def export_vit_to_torchscript(model_name: str, output_path: str, image_size: int = 224):
    """
    Export a ViT model to TorchScript format.
    
    Args:
        model_name: Name of the HuggingFace model (e.g., 'google/vit-base-patch16-224')
        output_path: Path to save the TorchScript model
        image_size: Image size (default: 224)
    """
    print(f"Loading ViT model: {model_name}")
    
    # Load model and processor
    model = ViTModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded. Config:")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Num layers: {model.config.num_hidden_layers}")
    print(f"  - Num attention heads: {model.config.num_attention_heads}")
    print(f"  - Patch size: {model.config.patch_size}")
    print(f"  - Image size: {model.config.image_size}")
    
    # Create dummy input for tracing
    batch_size = 1
    channels = 3
    dummy_pixel_values = torch.randn(batch_size, channels, image_size, image_size)
    
    print(f"\nTracing model with input shape: {dummy_pixel_values.shape}")
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, 
            dummy_pixel_values,
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
        output = loaded_model(dummy_pixel_values)
        # ViT returns BaseModelOutputWithPooling
        if hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        else:
            # If traced output is a tuple/list
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        print(f"Verification successful! Output shape: {hidden_states.shape}")
    
    # Save model config as JSON for reference
    config_path = output_path.replace('.pt', '_config.json')
    config_dict = {
        'model_name': model_name,
        'hidden_size': model.config.hidden_size,
        'num_hidden_layers': model.config.num_hidden_layers,
        'num_attention_heads': model.config.num_attention_heads,
        'patch_size': model.config.patch_size,
        'image_size': model.config.image_size,
        'num_channels': model.config.num_channels,
        'intermediate_size': model.config.intermediate_size
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    # Save processor config
    processor_dir = output_path.replace('.pt', '_processor')
    processor.save_pretrained(processor_dir)
    print(f"Saved processor to: {processor_dir}")
    
    print("\nExport complete!")
    print(f"  Model: {output_path}")
    print(f"  Config: {config_path}")
    print(f"  Processor: {processor_dir}")
    
    print("\nNote: For image preprocessing in C++, you'll need to:")
    print("  1. Resize images to {}x{}".format(image_size, image_size))
    print("  2. Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]")
    print("  3. Convert to tensor [batch, 3, {}, {}]".format(image_size, image_size))


def main():
    parser = argparse.ArgumentParser(description='Export ViT model to TorchScript')
    parser.add_argument(
        '--model',
        type=str,
        default='google/vit-base-patch16-224',
        help='HuggingFace model name (default: google/vit-base-patch16-224)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='vit_base.pt',
        help='Output path for TorchScript model (default: vit_base.pt)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size (default: 224)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    export_vit_to_torchscript(args.model, args.output, args.image_size)


if __name__ == '__main__':
    main()
