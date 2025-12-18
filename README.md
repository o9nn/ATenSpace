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
- 💾 **Persistence**: Serialization support for saving/loading knowledge graphs
- 🔗 **Rich Link Types**: Logical, temporal, contextual, and set-based relationships

## Quick Start

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

## Architecture

ATenSpace implements the core concepts from OpenCog's AtomSpace:

- **Atoms** - Immutable knowledge units (base class)
  - **Nodes** - Represent entities/concepts (can have tensor embeddings)
  - **Links** - Represent relationships (hypergraph edges)
- **AtomSpace** - Container managing the hypergraph database
- **TimeServer** - Tracks temporal information and events
- **AttentionBank** - Manages attention values and cognitive focus
- **Serializer** - Provides persistence (save/load)
- **Truth Values** - Tensor-based probabilistic values
- **Incoming Sets** - Track what links reference each atom

## Documentation

- [ATenSpace API Documentation](aten/src/ATen/atomspace/README.md)
- [Example Usage](aten/src/ATen/atomspace/example.cpp)
- [Tests](aten/src/ATen/atomspace/test.cpp)

## Building

```bash
cd aten
mkdir -p build && cd build
cmake ..
make

# Run example
./atomspace_example

# Run tests
./atomspace_test
```

## Use Cases

- **Knowledge Graphs**: Represent complex domain knowledge
- **Semantic Search**: Find similar concepts using embeddings
- **Reasoning Systems**: Build inference engines over knowledge
- **NLP Applications**: Combine symbolic and neural language understanding
- **Recommendation Systems**: Graph-based recommendations with embeddings
- **Cognitive Architectures**: Foundation for AGI research

## Comparison with OpenCog AtomSpace

| Feature | OpenCog AtomSpace | ATenSpace |
|---------|------------------|-----------|
| Core Structure | Hypergraph | Hypergraph |
| Implementation | Custom C++ | ATen Tensors |
| Embeddings | Via separate Values | Native tensor support |
| GPU Support | Limited | Full via ATen/PyTorch |
| Similarity Search | Pattern matching | Tensor operations |
| Backend | Custom memory mgmt | ATen tensor library |

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
