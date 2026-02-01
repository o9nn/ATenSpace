#!/usr/bin/env python3
"""
Export YOLO model to TorchScript format.

This script downloads a YOLO model (using ultralytics/YOLOv5 or similar)
and exports it to TorchScript format that can be loaded in C++ via LibTorch.

Note: YOLO models typically come from different sources than HuggingFace.
This script uses the ultralytics/yolov5 repository as the source.

Usage:
    python export_yolo.py --model yolov5s --output yolov5s.pt
"""

import argparse
import torch
import os
import json

try:
    # Try importing YOLOv5 from ultralytics
    import sys
    sys.path.append('yolov5')  # Add yolov5 directory to path if cloned
except ImportError:
    pass


def export_yolo_torchscript_simple(output_path: str):
    """
    Create a simplified YOLO-style model for demonstration.
    
    In production, you would:
    1. Clone YOLOv5: git clone https://github.com/ultralytics/yolov5
    2. Install requirements: pip install -r yolov5/requirements.txt
    3. Load pretrained model: model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    4. Export to TorchScript
    
    For now, we create a simplified architecture for demonstration.
    """
    print("Creating simplified YOLO-style model for demonstration...")
    print("Note: For production, use actual YOLOv5/YOLOv8 models from ultralytics")
    
    class SimplifiedYOLO(torch.nn.Module):
        """Simplified YOLO-style network for demonstration"""
        def __init__(self):
            super().__init__()
            # Backbone (feature extraction)
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                
                torch.nn.Conv2d(128, 256, 3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )
            
            # Detection head (simplified)
            self.head = torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, 3, padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 85, 1),  # 80 classes + 4 bbox + 1 objectness
            )
        
        def forward(self, x):
            x = self.backbone(x)
            x = self.head(x)
            return x
    
    model = SimplifiedYOLO()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    print(f"Tracing simplified YOLO model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    # Save model
    print(f"Saving to: {output_path}")
    traced_model.save(output_path)
    
    # Save config
    config_path = output_path.replace('.pt', '_config.json')
    config_dict = {
        'model_name': 'simplified_yolo',
        'model_type': 'yolo',
        'input_size': 640,
        'num_classes': 80,
        'note': 'This is a simplified demonstration model. For production, use real YOLOv5/YOLOv8.'
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    print("\nSimplified YOLO model exported!")
    print("\nFor production use:")
    print("  1. Clone YOLOv5: git clone https://github.com/ultralytics/yolov5")
    print("  2. Install: pip install -r yolov5/requirements.txt")
    print("  3. Export:")
    print("     cd yolov5")
    print("     python export.py --weights yolov5s.pt --include torchscript")


def export_yolo_from_hub(model_name: str, output_path: str):
    """
    Export YOLO model from torch.hub (ultralytics/yolov5).
    
    Requires: pip install ultralytics
    """
    print(f"Loading YOLOv5 model: {model_name}")
    
    try:
        # Load model from torch.hub
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        model.eval()
        
        print(f"Model loaded successfully")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        
        print(f"Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input, strict=False)
        
        # Save model
        print(f"Saving to: {output_path}")
        traced_model.save(output_path)
        
        # Save config
        config_path = output_path.replace('.pt', '_config.json')
        config_dict = {
            'model_name': model_name,
            'model_type': 'yolov5',
            'input_size': 640,
            'num_classes': 80,  # COCO classes
            'stride': 32
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Saved config to: {config_path}")
        
        print("\nYOLO model exported successfully!")
        
    except Exception as e:
        print(f"\nError loading from torch.hub: {e}")
        print("\nFalling back to simplified model...")
        export_yolo_torchscript_simple(output_path)


def main():
    parser = argparse.ArgumentParser(description='Export YOLO model to TorchScript')
    parser.add_argument(
        '--model',
        type=str,
        default='yolov5s',
        help='YOLO model name (yolov5s, yolov5m, yolov5l, yolov5x) (default: yolov5s)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='yolov5s.pt',
        help='Output path for TorchScript model (default: yolov5s.pt)'
    )
    parser.add_argument(
        '--simplified',
        action='store_true',
        help='Create simplified model for demonstration (no internet required)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    if args.simplified:
        export_yolo_torchscript_simple(args.output)
    else:
        export_yolo_from_hub(args.model, args.output)


if __name__ == '__main__':
    main()
