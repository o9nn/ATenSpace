# ATenSpace

ATenSpace is an ATen Tensor-based adaptation of [OpenCog's AtomSpace](https://github.com/opencog/atomspace), providing a hypergraph knowledge representation system with efficient tensor operations for AI applications.

## Overview

ATenSpace bridges symbolic AI and neural AI by combining:
- **Hypergraph Knowledge Representation** - Store and query complex knowledge structures
- **Tensor Embeddings** - Integrate neural network representations directly into the knowledge graph
- **Efficient Operations** - Leverage ATen's optimized tensor operations for semantic similarity and reasoning

## Features

- 🧠 **Hybrid AI**: Combine symbolic knowledge graphs with neural embeddings
- ⚡ **High Performance**: Built on ATen's optimized tensor library
- 🔍 **Semantic Search**: Find similar concepts using tensor similarity
- 🔗 **Hypergraph Structure**: Represent complex relationships beyond simple graphs
- 🔒 **Thread-Safe**: Concurrent access to the knowledge base
- 🎯 **Type-Safe**: Strong typing for atoms and links
- ⏰ **Temporal Reasoning**: TimeServer for tracking temporal information
- 🎯 **Attention Mechanisms**: AttentionBank for cognitive focus management
- 🧠 **ECAN**: Economic Attention Networks for attention allocation **(NEW - Phase 3)**
- 💾 **Persistence**: Serialization support for saving/loading knowledge graphs
- 🔗 **Rich Link Types**: Logical, temporal, contextual, and set-based relationships
- 🧮 **PLN Reasoning**: Probabilistic Logic Networks for uncertain inference
- 🔍 **Pattern Matching**: Variable binding and unification for queries
- ⚡ **Forward Chaining**: Automatic inference of new knowledge
- 🎯 **Backward Chaining**: Goal-directed reasoning and proof search
- 🧠 **TensorLogicEngine**: GPU-accelerated batch logical operations **(NEW - Phase 4)**
- 🎛️ **CognitiveEngine**: Master algorithm integration framework **(NEW - Phase 4)**
- 📝 **NLU**: Natural Language Understanding and generation **(NEW - Phase 5)**
- 👁️ **Vision**: Visual perception and scene understanding **(NEW - Phase 5)**
- 🐍 **Python Bindings**: Full Python API with PyTorch integration **(NEW - Phase 6)**
- 🤖 **ATenNN**: Neural Networks with pre-trained models (BERT, GPT, ViT, YOLO) **(NEW - Phase 7)**
- 🔗 **Neuro-Symbolic**: Seamless integration between neural and symbolic AI **(NEW - Phase 7)**

## Quick Start

### C++ API

```cpp
#include <ATen/atomspace/ATenSpace.h>

using namespace at::atomspace;

// Create knowledge base
AtomSpace space;

// Add concepts
auto cat = createConceptNode(space, "cat");
auto mammal = createConceptNode(space, "mammal");

// Create relationships
auto inheritance = createInheritanceLink(space, cat, mammal);

// Add tensor embeddings for similarity search
auto dog = createConceptNode(space, "dog", torch::randn({128}));

// Query similar concepts
auto results = space.querySimilar(torch::randn({128}), /*k=*/5);
```

### Python API

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

See [Python API Documentation](docs/PYTHON_API.md) for complete guide.

## Architecture

ATenSpace implements the core concepts from OpenCog's AtomSpace with PLN reasoning and ECAN:

- **Atoms** - Immutable knowledge units (base class)
  - **Nodes** - Represent entities/concepts (can have tensor embeddings)
  - **Links** - Represent relationships (hypergraph edges)
- **AtomSpace** - Container managing the hypergraph database
- **TimeServer** - Tracks temporal information and events
- **AttentionBank** - Manages attention values and cognitive focus
- **ECAN** - Economic Attention Networks **(NEW - Phase 3)**
  - **HebbianLinks** - Track co-occurrence correlations
  - **ImportanceSpreading** - Attention propagation
  - **ForgettingAgent** - Memory management
  - **RentAgent** - Economic scarcity mechanism
  - **WageAgent** - Reward useful knowledge
- **Serializer** - Provides persistence (save/load)
- **Truth Values** - Tensor-based probabilistic values
- **Incoming Sets** - Track what links reference each atom
- **PatternMatcher** - Pattern matching with variable binding
- **TruthValue** - PLN formulas for uncertain reasoning
- **ForwardChainer** - Forward chaining inference engine
- **BackwardChainer** - Goal-directed backward chaining
- **TensorLogicEngine** - GPU-accelerated batch logical inference **(NEW - Phase 4)**
- **CognitiveEngine** - Master cognitive architecture orchestrator **(NEW - Phase 4)**
- **NLU** - Natural Language Understanding for text processing **(NEW - Phase 5)**
  - **TextProcessor** - Tokenization and preprocessing
  - **EntityRecognizer** - Named entity recognition
  - **RelationExtractor** - Semantic relation extraction
  - **SemanticExtractor** - Text to knowledge graph conversion
  - **LanguageGenerator** - Knowledge graph to text generation
- **Vision** - Visual perception for grounded knowledge **(NEW - Phase 5)**
  - **ObjectDetector** - Object detection integration
  - **SpatialAnalyzer** - Spatial relationship extraction
  - **SceneUnderstanding** - Scene graph construction
  - **VisualReasoning** - Visual reasoning with PLN
  - **MultimodalIntegration** - Vision + Language integration
- **ATenNN** - Neural Networks Framework **(NEW - Phase 7)**
  - **NeuralModule** - Base class for neural components
  - **ModelRegistry** - Centralized model management
  - **EmbeddingExtractor** - Extract embeddings to AtomSpace
  - **AttentionBridge** - Map neural attention to ECAN
  - **PerformanceMonitor** - Track metrics and performance
  - **BERT** - Language understanding (contextualized embeddings)
  - **GPT** - Text generation (autoregressive transformer)
  - **ViT** - Visual understanding (Vision Transformer)
  - **YOLO** - Object detection (real-time detection)

## Documentation

### C++ Documentation
- [ATenSpace API Documentation](aten/src/ATen/atomspace/README.md)
- [Example Usage](aten/src/ATen/atomspace/example.cpp)
- [Advanced Examples](aten/src/ATen/atomspace/example_advanced.cpp)
- [PLN Examples](aten/src/ATen/atomspace/example_pln.cpp)
- [ECAN Examples](aten/src/ATen/atomspace/example_ecan.cpp) **(Phase 3)**
- [Cognitive Engine Examples](aten/src/ATen/atomspace/example_cognitive.cpp) **(Phase 4)**
- [NLU Examples](aten/src/ATen/atomspace/example_nlu.cpp) **(Phase 5)**
- [Vision Examples](aten/src/ATen/atomspace/example_vision.cpp) **(Phase 5)**
- [Neural Network Examples](aten/src/ATen/atomspace/example_nn.cpp) **(Phase 7)**
- [Tests](aten/src/ATen/atomspace/test.cpp)
- [PLN Tests](aten/src/ATen/atomspace/test_pln.cpp)
- [ECAN Tests](aten/src/ATen/atomspace/test_ecan.cpp) **(Phase 3)**

### Python Documentation **(NEW - Phase 6)**
- [Python API Guide](docs/PYTHON_API.md)
- [Python Basic Examples](examples/python/basic_usage.py)
- [Python Advanced Examples](examples/python/advanced_usage.py)
- [Python Neural Network Examples](examples/python/nn_integration.py) **(Phase 7)**

## Installation

### C++ Library

```bash
cd aten
mkdir -p build && cd build
cmake ..
make

# Run example
./atomspace_example

# Run advanced examples
./atomspace_example_advanced

# Run PLN examples
./atomspace_example_pln

# Run ECAN examples (NEW - Phase 3)
./atomspace_example_ecan

# Run Cognitive Engine examples (NEW - Phase 4)
./atomspace_example_cognitive

# Run NLU examples (Phase 5)
./atomspace_example_nlu

# Run Vision examples (Phase 5)
./atomspace_example_vision

# Run Neural Network examples (NEW - Phase 7)
./atomspace_example_nn

# Run tests
./atomspace_test
./atomspace_test_advanced
./atomspace_test_pln
./atomspace_test_ecan
./atomspace_test_cognitive
./atomspace_test_nlu
./atomspace_test_vision
```

### Python Package **(NEW - Phase 6)**

```bash
# Install dependencies
pip install torch numpy pybind11

# Install ATenSpace Python package
pip install -e .

# Or build with Python bindings enabled
cd aten
mkdir -p build && cd build
cmake -DBUILD_PYTHON_BINDINGS=ON ..
make

# Run Python examples
python examples/python/basic_usage.py
python examples/python/advanced_usage.py
python examples/python/nn_integration.py  # NEW - Phase 7
```

## Use Cases

- **Knowledge Graphs**: Represent complex domain knowledge
- **Semantic Search**: Find similar concepts using embeddings
- **Attention-Guided Reasoning**: Focus computational resources on important knowledge **(NEW - Phase 3)**
- **Reasoning Systems**: Build inference engines over knowledge **(Enhanced with PLN)**
- **NLP Applications**: Combine symbolic and neural language understanding
- **Recommendation Systems**: Graph-based recommendations with embeddings
- **Cognitive Architectures**: Foundation for AGI research **(PLN-powered)**
- **Probabilistic Reasoning**: Handle uncertain knowledge with truth values **(NEW)**
- **Automated Theorem Proving**: Goal-directed reasoning with backward chaining **(NEW)**
- **Knowledge Discovery**: Derive new facts via forward chaining **(NEW)**
- **Natural Language Processing**: Text understanding and generation **(Phase 5)**
- **Visual Understanding**: Scene understanding and visual reasoning **(Phase 5)**
- **Multimodal AI**: Integrate vision, language, and knowledge **(Phase 5)**
- **Python Development**: Rapid prototyping with Python API **(NEW - Phase 6)**
- **ML Integration**: Easy integration with PyTorch models **(NEW - Phase 6)**
- **Neuro-Symbolic AI**: Combine neural models with symbolic reasoning **(NEW - Phase 7)**
- **Language Understanding**: BERT embeddings for concept grounding **(NEW - Phase 7)**
- **Text Generation**: GPT-based knowledge expansion **(NEW - Phase 7)**
- **Visual Perception**: ViT and YOLO for vision-based AI **(NEW - Phase 7)**

## Comparison with OpenCog AtomSpace

| Feature | OpenCog AtomSpace | ATenSpace |
|---------|------------------|-----------|
| Core Structure | Hypergraph | Hypergraph |
| Implementation | Custom C++ | ATen Tensors |
| Embeddings | Via separate Values | Native tensor support |
| GPU Support | Limited | Full via ATen/PyTorch |
| Similarity Search | Pattern matching | Tensor operations |
| Backend | Custom memory mgmt | ATen tensor library |
| Python Bindings | Cython (partial) | pybind11 (complete) |

## Example: Knowledge Graph with Embeddings

```cpp
AtomSpace space;

// Create animal taxonomy
auto cat = createConceptNode(space, "cat", torch::randn({128}));
auto dog = createConceptNode(space, "dog", torch::randn({128}));
auto fish = createConceptNode(space, "fish", torch::randn({128}));
auto mammal = createConceptNode(space, "mammal");
auto animal = createConceptNode(space, "animal");

// Build hierarchy
createInheritanceLink(space, cat, mammal);
createInheritanceLink(space, dog, mammal);
createInheritanceLink(space, mammal, animal);

// Create properties
auto hasProperty = createPredicateNode(space, "has-property");
auto furry = createConceptNode(space, "furry");
createEvaluationLink(space, hasProperty, {cat, furry});
createEvaluationLink(space, hasProperty, {dog, furry});

// Set probabilistic truth values
cat->setTruthValue(torch::tensor({0.95f, 0.9f}));  // [strength, confidence]

// Query similar animals
Tensor query = cat->getEmbedding();
auto similar = space.querySimilar(query, /*k=*/3);

for (const auto& [atom, similarity] : similar) {
    std::cout << atom->toString() << " (sim: " << similarity << ")" << std::endl;
}
```

## Contributing

Contributions are welcome! Areas for enhancement:
- Pattern matching and unification
- Inference engines (forward/backward chaining)
- Persistent storage (serialization)
- Python bindings
- Distributed atomspace
- Advanced query languages

## References

- [OpenCog AtomSpace](https://github.com/opencog/atomspace) - Original implementation
- [OpenCog Wiki](https://wiki.opencog.org/w/AtomSpace) - Concepts and documentation
- [ATen Documentation](https://pytorch.org/cppdocs/) - PyTorch C++ tensor library

## License

This project follows the licensing of the ATen/PyTorch project.

## Acknowledgments

Based on the design and concepts from [OpenCog's AtomSpace](https://github.com/opencog/atomspace), reimagined with ATen tensors for modern deep learning integration.
