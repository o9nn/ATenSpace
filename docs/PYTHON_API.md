# ATenSpace Python Bindings

Python interface for ATenSpace - a tensor-based cognitive architecture.

## Installation

### From Source

```bash
# Install dependencies
pip install torch numpy pybind11

# Build and install
cd /path/to/ATenSpace
pip install -e .
```

### Requirements

- Python 3.7+
- PyTorch 1.9.0+
- pybind11 (automatically installed)
- C++17 compiler
- CMake 3.5+

## Quick Start

```python
import torch
import atenspace as at

# Create knowledge base
space = at.AtomSpace()

# Add concepts
cat = at.create_concept_node(space, "cat")
mammal = at.create_concept_node(space, "mammal")

# Create relationships
inheritance = at.create_inheritance_link(space, cat, mammal)

# Add embeddings for similarity search
dog = at.create_concept_node(space, "dog", torch.randn(128))

# Query similar concepts
similar = space.query_similar(torch.randn(128), k=5)
```

## Features

### Core AtomSpace

- **Knowledge Representation**: Hypergraph-based knowledge storage
- **Embeddings**: Native PyTorch tensor integration
- **Similarity Search**: GPU-accelerated semantic queries
- **Type System**: Rich atom types (concepts, predicates, links)

### PLN (Probabilistic Logic Networks)

- **Truth Values**: Probabilistic reasoning with strength and confidence
- **Deduction**: Logical inference with uncertainty
- **Pattern Matching**: Variable binding and unification
- **Forward/Backward Chaining**: Automated reasoning

### ECAN (Economic Attention Networks)

- **Attention Values**: STI, LTI, VLTI for cognitive focus
- **Importance Spreading**: Attention propagation
- **Forgetting**: Economic memory management

### NLU & Vision

- **Text Processing**: Entity recognition, relation extraction
- **Knowledge Graph Building**: Text to AtomSpace conversion
- **Visual Understanding**: Object detection, spatial reasoning
- **Multimodal**: Vision + Language integration

## Examples

See `examples/python/` directory:
- `basic_usage.py`: Core features
- `advanced_usage.py`: PLN, ECAN, NLU, Vision

## Version

Current version: **0.6.0** (Phase 6 - Python Bindings)
