# ATenSpace - Tensor-based AtomSpace

ATenSpace is an ATen tensor-based adaptation of [OpenCog's AtomSpace](https://github.com/opencog/atomspace), providing a hypergraph knowledge representation system with efficient tensor operations.

## Overview

ATenSpace brings together symbolic AI (knowledge graphs) and neural AI (tensor embeddings) by representing knowledge as a hypergraph where:

- **Nodes** represent entities or concepts and can have tensor embeddings
- **Links** represent relationships between atoms (hypergraph edges)
- **Truth Values** are stored as tensors for flexible probabilistic reasoning
- **Similarity Queries** leverage tensor operations for efficient semantic search

## Core Concepts

### Atom
The base class for all knowledge units. Atoms are:
- **Immutable**: Once created, their core identity doesn't change
- **Unique**: AtomSpace ensures no duplicate atoms exist
- **Typed**: Each atom has a specific type (e.g., ConceptNode, InheritanceLink)

### Node
Represents entities or concepts. Features:
- Named entities (e.g., "cat", "mammal")
- Optional tensor embeddings for semantic representation
- Support for different node types (ConceptNode, PredicateNode, etc.)

### Link
Represents relationships between atoms. Features:
- Connects any number of atoms (not just pairs)
- Can connect nodes and other links (forming a hypergraph)
- Maintains ordered outgoing sets
- Support for various link types (InheritanceLink, EvaluationLink, etc.)

### AtomSpace
The container managing the hypergraph. Features:
- Thread-safe atom management
- Efficient indexing for fast lookup
- Similarity-based queries using tensor embeddings
- Incoming set tracking (what links reference each atom)

## Usage Example

```cpp
#include <ATen/atomspace/ATenSpace.h>

using namespace at::atomspace;

// Create an AtomSpace
AtomSpace space;

// Create concept nodes
auto cat = createConceptNode(space, "cat");
auto mammal = createConceptNode(space, "mammal");
auto animal = createConceptNode(space, "animal");

// Create inheritance hierarchy: cat -> mammal -> animal
auto inh1 = createInheritanceLink(space, cat, mammal);
auto inh2 = createInheritanceLink(space, mammal, animal);

// Create nodes with embeddings for similarity search
auto dog = createConceptNode(space, "dog", torch::randn({128}));
auto fish = createConceptNode(space, "fish", torch::randn({128}));

// Query for similar concepts
Tensor query = torch::randn({128});
auto results = space.querySimilar(query, /*k=*/5);

// Create relations using evaluation links
auto hasProperty = createPredicateNode(space, "has-property");
auto furry = createConceptNode(space, "furry");
auto eval = createEvaluationLink(space, hasProperty, {cat, furry});

// Set truth values (strength and confidence)
cat->setTruthValue(torch::tensor({0.9f, 0.8f}));

// Query atoms by type
auto conceptNodes = space.getAtomsByType(Atom::Type::CONCEPT_NODE);
```

## API Reference

### Creating Atoms

#### Nodes
```cpp
// Basic node creation
Handle addNode(Atom::Type type, const std::string& name);

// Node with embedding
Handle addNode(Atom::Type type, const std::string& name, const Tensor& embedding);

// Convenience functions
Handle createConceptNode(AtomSpace& space, const std::string& name);
Handle createPredicateNode(AtomSpace& space, const std::string& name);
```

#### Links
```cpp
// Generic link creation
Handle addLink(Atom::Type type, const std::vector<Handle>& outgoing);

// Convenience functions
Handle createInheritanceLink(AtomSpace& space, Handle from, Handle to);
Handle createEvaluationLink(AtomSpace& space, Handle predicate, 
                            const std::vector<Handle>& args);
Handle createListLink(AtomSpace& space, const std::vector<Handle>& atoms);
```

### Querying Atoms

```cpp
// Get all atoms
AtomSet getAtoms() const;

// Get atoms by type
std::vector<Handle> getAtomsByType(Atom::Type type) const;

// Get specific node
Handle getNode(Atom::Type type, const std::string& name) const;

// Similarity search (for atoms with embeddings)
std::vector<std::pair<Handle, float>> querySimilar(
    const Tensor& query, 
    size_t k = 10, 
    float threshold = 0.0) const;

// Get number of atoms
size_t size() const;
```

### Atom Operations

```cpp
// Truth values
void setTruthValue(const Tensor& tv);
Tensor getTruthValue() const;

// Attention values
void setAttention(float attention);
float getAttention() const;

// Node-specific
std::string getName() const;  // For nodes only
void setEmbedding(const Tensor& embedding);
Tensor getEmbedding() const;

// Link-specific
const OutgoingSet& getOutgoingSet() const;  // For links only
size_t getArity() const;
Handle getOutgoingAtom(size_t index) const;

// Incoming set (what references this atom)
const std::vector<WeakHandle>& getIncomingSet() const;
```

## Atom Types

### Node Types
- `NODE` - Generic node
- `CONCEPT_NODE` - Represents a concept or entity
- `PREDICATE_NODE` - Represents a predicate/relation name

### Link Types
- `LINK` - Generic link
- `INHERITANCE_LINK` - Represents inheritance (is-a) relationships
- `EVALUATION_LINK` - Represents predicate evaluation
- `LIST_LINK` - Ordered list of atoms
- `ORDERED_LINK` - Generic ordered link
- `UNORDERED_LINK` - Generic unordered link

## Design Principles

1. **Tensor Integration**: All value representations use ATen tensors for efficiency and GPU compatibility
2. **Immutability**: Atoms are immutable after creation, ensuring cache consistency
3. **Uniqueness**: AtomSpace maintains a single instance of each unique atom
4. **Thread Safety**: All AtomSpace operations are thread-safe
5. **Hypergraph Structure**: Links can connect any number of atoms, not just pairs

## Comparison with OpenCog AtomSpace

| Feature | OpenCog AtomSpace | ATenSpace |
|---------|------------------|-----------|
| Data Structure | Hypergraph | Hypergraph |
| Storage | Custom C++ | ATen Tensors |
| Embeddings | Via Values | Native Tensor Support |
| GPU Support | Limited | Full via ATen |
| Similarity Search | Pattern Matching | Tensor Operations |
| Language | C++ (with Scheme) | C++ (with ATen) |

## Building the Example

```bash
# Navigate to the ATen directory
cd aten

# Build ATen with atomspace support
mkdir -p build && cd build
cmake ..
make

# Run the example
./atomspace_example
```

## Future Enhancements

Potential areas for expansion:
- Pattern matching and unification
- Backward chaining inference
- Distributed atomspace support
- Persistent storage (serialization)
- Python bindings
- GPU-accelerated operations
- Advanced query languages

## References

- [OpenCog AtomSpace](https://github.com/opencog/atomspace)
- [OpenCog Wiki](https://wiki.opencog.org/w/AtomSpace)
- [ATen (PyTorch Tensor Library)](https://pytorch.org/cppdocs/)

## License

This project follows the same license as the parent ATen/PyTorch project.
