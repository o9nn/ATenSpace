#pragma once

/**
 * ATenSpace - ATen Tensor-based adaptation of OpenCog AtomSpace
 * 
 * This library provides a hypergraph knowledge representation system
 * using ATen tensors for efficient storage and computation.
 * 
 * Core concepts:
 * - Atom: Base class for all knowledge units (nodes and links)
 * - Node: Represents entities/concepts, can have tensor embeddings
 * - Link: Represents relationships between atoms (hypergraph edges)
 * - AtomSpace: Container managing the hypergraph database
 * 
 * Features:
 * - Tensor-based truth values and embeddings
 * - Similarity-based queries using tensor operations
 * - Thread-safe atom management
 * - Immutable atoms with unique identity
 */

#include <ATen/atomspace/Atom.h>
#include <ATen/atomspace/AtomSpace.h>

namespace at {
namespace atomspace {

/**
 * Convenience functions for creating atoms
 */

// Create a concept node
inline Atom::Handle createConceptNode(
    AtomSpace& space, 
    const std::string& name) {
    return space.addNode(Atom::Type::CONCEPT_NODE, name);
}

// Create a concept node with embedding
inline Atom::Handle createConceptNode(
    AtomSpace& space, 
    const std::string& name,
    const Tensor& embedding) {
    return space.addNode(Atom::Type::CONCEPT_NODE, name, embedding);
}

// Create a predicate node
inline Atom::Handle createPredicateNode(
    AtomSpace& space, 
    const std::string& name) {
    return space.addNode(Atom::Type::PREDICATE_NODE, name);
}

// Create an inheritance link: A inherits from B
inline Atom::Handle createInheritanceLink(
    AtomSpace& space,
    Atom::Handle from,
    Atom::Handle to) {
    return space.addLink(Atom::Type::INHERITANCE_LINK, {from, to});
}

// Create an evaluation link: predicate(args...)
inline Atom::Handle createEvaluationLink(
    AtomSpace& space,
    Atom::Handle predicate,
    const std::vector<Atom::Handle>& args) {
    // Create list link for arguments
    auto listLink = space.addLink(Atom::Type::LIST_LINK, args);
    // Create evaluation link
    return space.addLink(Atom::Type::EVALUATION_LINK, {predicate, listLink});
}

// Create a list link
inline Atom::Handle createListLink(
    AtomSpace& space,
    const std::vector<Atom::Handle>& atoms) {
    return space.addLink(Atom::Type::LIST_LINK, atoms);
}

} // namespace atomspace
} // namespace at
