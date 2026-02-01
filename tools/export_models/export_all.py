#!/usr/bin/env python3
"""
Export all models from HuggingFace to TorchScript format.

This convenience script exports BERT, GPT-2, ViT, and YOLO models
in one go, creating a complete model directory structure.

Usage:
    python export_all.py --output-dir ../../models
"""

import argparse
import os
import sys
from pathlib import Path

# Import individual export scripts
from export_bert import export_bert_to_torchscript
from export_gpt2 import export_gpt2_to_torchscript
from export_vit import export_vit_to_torchscript
from export_yolo import export_yolo_torchscript_simple, export_yolo_from_hub


def export_all_models(output_dir: str, simplified_yolo: bool = True):
    """
    Export all models to the specified directory.
    
    Args:
        output_dir: Base directory for exported models
        simplified_yolo: Use simplified YOLO (True) or download from hub (False)
    """
    print("=" * 60)
    print("ATenSpace Model Export Tool")
    print("Exporting all models to:", output_dir)
    print("=" * 60)
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    models = []
    
    # Export BERT
    print("\n" + "=" * 60)
    print("1/4: Exporting BERT")
    print("=" * 60)
    try:
        bert_path = os.path.join(output_dir, "bert_base.pt")
        export_bert_to_torchscript("bert-base-uncased", bert_path)
        models.append(("BERT", bert_path, "✓"))
    except Exception as e:
        print(f"Error exporting BERT: {e}")
        models.append(("BERT", "", "✗"))
    
    # Export GPT-2
    print("\n" + "=" * 60)
    print("2/4: Exporting GPT-2")
    print("=" * 60)
    try:
        gpt2_path = os.path.join(output_dir, "gpt2.pt")
        export_gpt2_to_torchscript("gpt2", gpt2_path)
        models.append(("GPT-2", gpt2_path, "✓"))
    except Exception as e:
        print(f"Error exporting GPT-2: {e}")
        models.append(("GPT-2", "", "✗"))
    
    # Export ViT
    print("\n" + "=" * 60)
    print("3/4: Exporting ViT")
    print("=" * 60)
    try:
        vit_path = os.path.join(output_dir, "vit_base.pt")
        export_vit_to_torchscript("google/vit-base-patch16-224", vit_path)
        models.append(("ViT", vit_path, "✓"))
    except Exception as e:
        print(f"Error exporting ViT: {e}")
        models.append(("ViT", "", "✗"))
    
    # Export YOLO
    print("\n" + "=" * 60)
    print("4/4: Exporting YOLO")
    print("=" * 60)
    try:
        yolo_path = os.path.join(output_dir, "yolov5s.pt")
        if simplified_yolo:
            print("Using simplified YOLO model (no internet required)")
            export_yolo_torchscript_simple(yolo_path)
        else:
            print("Downloading YOLOv5 from torch.hub")
            export_yolo_from_hub("yolov5s", yolo_path)
        models.append(("YOLO", yolo_path, "✓"))
    except Exception as e:
        print(f"Error exporting YOLO: {e}")
        models.append(("YOLO", "", "✗"))
    
    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    
    success_count = 0
    for model_name, model_path, status in models:
        print(f"{status} {model_name:10s} ", end="")
        if status == "✓":
            size_mb = os.path.getsize(model_path) / (1024 * 1024) if model_path else 0
            print(f"({size_mb:.1f} MB) - {model_path}")
            success_count += 1
        else:
            print("- Export failed")
    
    print("\n" + "=" * 60)
    print(f"Successfully exported {success_count}/{len(models)} models")
    print("=" * 60)
    
    if success_count == len(models):
        print("\n✓ All models exported successfully!")
        print("\nNext steps:")
        print("1. Build ATenSpace C++ examples:")
        print("   cd aten/build && cmake .. && make atomspace_example_model_loader")
        print("2. Run the example:")
        print("   ./atomspace_example_model_loader")
    else:
        print("\n✗ Some models failed to export. Check the errors above.")
        print("\nCommon issues:")
        print("- Missing dependencies: pip install transformers torch")
        print("- No internet connection: Use --simplified-yolo flag")
        print("- Out of memory: Close other applications")


def main():
    parser = argparse.ArgumentParser(description='Export all models to TorchScript')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../models',
        help='Output directory for all models (default: ../../models)'
    )
    parser.add_argument(
        '--simplified-yolo',
        action='store_true',
        default=True,
        help='Use simplified YOLO model (default: True)'
    )
    parser.add_argument(
        '--full-yolo',
        action='store_true',
        help='Download full YOLOv5 from torch.hub (requires internet)'
    )
    
    args = parser.parse_args()
    
    # Override simplified_yolo if full-yolo is requested
    simplified_yolo = not args.full_yolo
    
    export_all_models(args.output_dir, simplified_yolo)


if __name__ == '__main__':
    main()
