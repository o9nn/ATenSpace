#!/usr/bin/env python3
"""
ATenSpace Python Examples - Basic Usage
Phase 6 - Production Integration

Demonstrates basic usage of ATenSpace Python bindings.
"""

import torch
import atenspace as at

def example_1_basic_atomspace():
    """Example 1: Basic AtomSpace operations"""
    print("\n=== Example 1: Basic AtomSpace ===")
    
    # Create AtomSpace
    space = at.AtomSpace()
    print(f"Created AtomSpace (size: {len(space)})")
    
    # Create nodes
    cat = at.create_concept_node(space, "cat")
    dog = at.create_concept_node(space, "dog")
    mammal = at.create_concept_node(space, "mammal")
    animal = at.create_concept_node(space, "animal")
    
    print(f"Created nodes: {cat.to_string()}, {dog.to_string()}")
    
    # Create links
    cat_is_mammal = at.create_inheritance_link(space, cat, mammal)
    dog_is_mammal = at.create_inheritance_link(space, dog, mammal)
    mammal_is_animal = at.create_inheritance_link(space, mammal, animal)
    
    print(f"Created inheritance hierarchy")
    print(f"AtomSpace size: {len(space)}")
    
    # Query by type
    concepts = space.get_atoms_by_type(at.AtomType.CONCEPT_NODE)
    print(f"Found {len(concepts)} concept nodes")

def example_2_embeddings():
    """Example 2: Working with embeddings"""
    print("\n=== Example 2: Embeddings and Similarity ===")
    
    space = at.AtomSpace()
    
    # Create nodes with random embeddings
    cat = at.create_concept_node(space, "cat", torch.randn(128))
    dog = at.create_concept_node(space, "dog", torch.randn(128))
    fish = at.create_concept_node(space, "fish", torch.randn(128))
    
    print("Created nodes with embeddings")
    print(f"Cat has embedding: {cat.has_embedding()}")
    
    # Query similar concepts
    query = cat.get_embedding()
    similar = space.query_similar(query, k=2)
    
    print(f"\nMost similar to 'cat':")
    for atom, similarity in similar:
        print(f"  {atom.get_name()}: {similarity:.4f}")

def example_3_truth_values():
    """Example 3: Probabilistic truth values"""
    print("\n=== Example 3: Truth Values ===")
    
    space = at.AtomSpace()
    
    # Create atoms with truth values
    cat = at.create_concept_node(space, "cat")
    mammal = at.create_concept_node(space, "mammal")
    
    # Set truth value [strength, confidence]
    tv = torch.tensor([0.95, 0.9])
    cat.set_truth_value(tv)
    
    link = at.create_inheritance_link(space, cat, mammal)
    link.set_truth_value(torch.tensor([0.9, 0.85]))
    
    print(f"Cat truth value: {cat.get_truth_value()}")
    print(f"Link truth value: {link.get_truth_value()}")

def example_4_attention():
    """Example 4: Attention allocation"""
    print("\n=== Example 4: Attention Bank ===")
    
    space = at.AtomSpace()
    bank = at.AttentionBank()
    
    # Create atoms
    important = at.create_concept_node(space, "important")
    normal = at.create_concept_node(space, "normal")
    unimportant = at.create_concept_node(space, "unimportant")
    
    # Set attention values (STI, LTI, VLTI)
    bank.set_attention_value(important, at.AttentionValue(100.0, 50.0, 10.0))
    bank.set_attention_value(normal, at.AttentionValue(50.0, 30.0, 5.0))
    bank.set_attention_value(unimportant, at.AttentionValue(10.0, 5.0, 1.0))
    
    # Stimulate an atom
    bank.stimulate(important, 20.0)
    
    # Get attentional focus
    focus = bank.get_attentional_focus(k=2)
    print(f"Top 2 atoms in focus:")
    for atom in focus:
        av = bank.get_attention_value(atom)
        print(f"  {atom.get_name()}: STI={av.sti:.1f}")

def example_5_time_tracking():
    """Example 5: Temporal tracking"""
    print("\n=== Example 5: Time Server ===")
    
    import time
    
    space = at.AtomSpace()
    time_server = at.TimeServer()
    
    # Create and track atom
    atom = at.create_concept_node(space, "tracked")
    time_server.record_creation(atom)
    
    time.sleep(0.1)
    
    # Record access
    time_server.record_access(atom)
    time_server.record_event(atom, "important_event")
    
    # Get times
    creation_time = time_server.get_creation_time(atom)
    access_time = time_server.get_last_access_time(atom)
    
    print(f"Atom created at: {creation_time}")
    print(f"Last accessed at: {access_time}")

def example_6_serialization():
    """Example 6: Save and load AtomSpace"""
    print("\n=== Example 6: Serialization ===")
    
    import tempfile
    import os
    
    # Create knowledge graph
    space = at.AtomSpace()
    cat = at.create_concept_node(space, "cat")
    mammal = at.create_concept_node(space, "mammal")
    at.create_inheritance_link(space, cat, mammal)
    
    print(f"Original AtomSpace size: {len(space)}")
    
    # Save to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        filename = f.name
    
    at.Serializer.save(space, filename)
    print(f"Saved to {filename}")
    
    # Load into new AtomSpace
    new_space = at.AtomSpace()
    at.Serializer.load(new_space, filename)
    print(f"Loaded AtomSpace size: {len(new_space)}")
    
    # Cleanup
    os.unlink(filename)

def example_7_logical_operations():
    """Example 7: Logical links"""
    print("\n=== Example 7: Logical Operations ===")
    
    space = at.AtomSpace()
    
    # Create propositions
    a = at.create_concept_node(space, "A")
    b = at.create_concept_node(space, "B")
    c = at.create_concept_node(space, "C")
    
    # Create logical combinations
    a_and_b = at.create_and_link(space, [a, b])
    a_or_b = at.create_or_link(space, [a, b])
    not_a = at.create_not_link(space, a)
    a_implies_c = at.create_implication_link(space, a, c)
    
    print(f"AND: {a_and_b.to_string()}")
    print(f"OR: {a_or_b.to_string()}")
    print(f"NOT: {not_a.to_string()}")
    print(f"IMPLIES: {a_implies_c.to_string()}")

def main():
    """Run all basic examples"""
    print("ATenSpace Python Examples")
    print("=" * 50)
    
    try:
        example_1_basic_atomspace()
        example_2_embeddings()
        example_3_truth_values()
        example_4_attention()
        example_5_time_tracking()
        example_6_serialization()
        example_7_logical_operations()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
