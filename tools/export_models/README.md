# Model Export Tools

This directory contains utilities to export HuggingFace models to TorchScript format for use in ATenSpace's C++ implementation.

## Prerequisites

```bash
pip install transformers torch torchvision
```

## Exporting Models

### BERT

Export BERT-base-uncased (default):
```bash
python export_bert.py
```

Export a specific BERT model:
```bash
python export_bert.py --model bert-large-uncased --output models/bert_large.pt
```

### GPT-2

Export GPT-2 (default):
```bash
python export_gpt2.py
```

Export GPT-2 medium:
```bash
python export_gpt2.py --model gpt2-medium --output models/gpt2_medium.pt
```

### ViT (Vision Transformer)

Coming soon...

### YOLO

Coming soon...

## Output Files

Each export script generates:
- `*.pt` - TorchScript model file (loadable in C++)
- `*_config.json` - Model configuration
- `*_tokenizer/` - Tokenizer files (for language models)

## Usage in C++

```cpp
#include <torch/script.h>

// Load the model
torch::jit::script::Module model = torch::jit::load("bert_base.pt");

// Prepare input
std::vector<int64_t> input_ids = {101, 2054, 2003, 102};  // [CLS] what is [SEP]
torch::Tensor input_tensor = torch::tensor(input_ids).unsqueeze(0);

// Run inference
std::vector<torch::jit::IValue> inputs;
inputs.push_back(input_tensor);
auto output = model.forward(inputs);
```

## Model Directory Structure

Recommended directory structure for exported models:
```
models/
├── bert/
│   ├── bert_base.pt
│   ├── bert_base_config.json
│   └── bert_base_tokenizer/
├── gpt2/
│   ├── gpt2.pt
│   ├── gpt2_config.json
│   └── gpt2_tokenizer/
├── vit/
│   └── vit_base.pt
└── yolo/
    └── yolov5.pt
```

## Notes

- TorchScript models are platform-independent and can be used on any system with LibTorch
- Model files can be large (100MB-1GB depending on the model)
- The first time you run these scripts, models will be downloaded from HuggingFace Hub
- Traced models are optimized for inference and cannot be fine-tuned

## Troubleshooting

### Out of Memory
- Use smaller batch sizes or shorter sequences when tracing
- Try tracing on a machine with more RAM

### Tracing Warnings
- Some warnings are expected due to dynamic behavior in transformer models
- Use `strict=False` in `torch.jit.trace()` to suppress these

### Model Not Found
- Check that the model name is correct on HuggingFace Hub
- Ensure you have internet connection for the first download
- Models are cached in `~/.cache/huggingface/`
