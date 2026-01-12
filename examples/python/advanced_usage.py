#!/usr/bin/env python3
"""
ATenSpace Python Examples - Advanced Features
Phase 6 - Production Integration

Demonstrates PLN reasoning, ECAN, NLU, and Vision capabilities.
"""

import torch
import atenspace as at

def example_1_pln_reasoning():
    """Example 1: PLN probabilistic reasoning"""
    print("\n=== Example 1: PLN Reasoning ===")
    
    space = at.AtomSpace()
    
    # Create knowledge with truth values
    socrates = at.create_concept_node(space, "Socrates")
    man = at.create_concept_node(space, "Man")
    mortal = at.create_concept_node(space, "Mortal")
    
    # Socrates is a Man [0.95, 0.9]
    link1 = at.create_inheritance_link(space, socrates, man)
    link1.set_truth_value(torch.tensor([0.95, 0.9]))
    
    # Man is Mortal [0.9, 0.85]
    link2 = at.create_inheritance_link(space, man, mortal)
    link2.set_truth_value(torch.tensor([0.9, 0.85]))
    
    # Deduce: Socrates is Mortal
    result_tv = at.TruthValueOps.deduction(
        link1.get_truth_value(),
        link2.get_truth_value()
    )
    
    print(f"Socrates is Man: {link1.get_truth_value()}")
    print(f"Man is Mortal: {link2.get_truth_value()}")
    print(f"Deduced: Socrates is Mortal: {result_tv}")

def example_2_pattern_matching():
    """Example 2: Pattern matching with variables"""
    print("\n=== Example 2: Pattern Matching ===")
    
    space = at.AtomSpace()
    
    # Create knowledge base
    cat = at.create_concept_node(space, "cat")
    dog = at.create_concept_node(space, "dog")
    mammal = at.create_concept_node(space, "mammal")
    
    at.create_inheritance_link(space, cat, mammal)
    at.create_inheritance_link(space, dog, mammal)
    
    # Create pattern with variable
    var_x = at.create_variable_node(space, "$X")
    pattern = at.create_inheritance_link(space, var_x, mammal)
    
    # Match pattern
    matcher = at.PatternMatcher(space)
    
    matches = []
    def callback(var_map):
        matches.append(var_map)
    
    matcher.query(pattern, callback)
    
    print(f"Pattern: $X inherits from {mammal.get_name()}")
    print(f"Found {len(matches)} matches:")
    for m in matches:
        if "$X" in m:
            print(f"  $X = {m['$X'].get_name()}")

def example_3_forward_chaining():
    """Example 3: Forward chaining inference"""
    print("\n=== Example 3: Forward Chaining ===")
    
    space = at.AtomSpace()
    chainer = at.ForwardChainer(space)
    
    # Add knowledge
    a = at.create_concept_node(space, "A")
    b = at.create_concept_node(space, "B")
    c = at.create_concept_node(space, "C")
    
    ab = at.create_implication_link(space, a, b)
    ab.set_truth_value(torch.tensor([0.9, 0.8]))
    
    bc = at.create_implication_link(space, b, c)
    bc.set_truth_value(torch.tensor([0.85, 0.75]))
    
    a.set_truth_value(torch.tensor([0.95, 0.9]))
    
    print(f"Initial knowledge: {len(space)} atoms")
    
    # Run inference
    chainer.run(max_iterations=10)
    results = chainer.get_results()
    
    print(f"After inference: {len(space)} atoms")
    print(f"New conclusions: {len(results)}")

def example_4_ecan_attention():
    """Example 4: ECAN attention dynamics"""
    print("\n=== Example 4: ECAN Attention ===")
    
    space = at.AtomSpace()
    bank = at.AttentionBank()
    
    # Create atoms
    a1 = at.create_concept_node(space, "Important1")
    a2 = at.create_concept_node(space, "Important2")
    a3 = at.create_concept_node(space, "Unimportant")
    
    # Set initial attention
    bank.set_attention_value(a1, at.AttentionValue(100.0, 50.0, 10.0))
    bank.set_attention_value(a2, at.AttentionValue(80.0, 40.0, 8.0))
    bank.set_attention_value(a3, at.AttentionValue(10.0, 5.0, 1.0))
    
    # Create Hebbian links
    hebbian = at.HebbianLink()
    hebbian.atom1 = a1
    hebbian.atom2 = a2
    hebbian.update(100.0, 80.0)
    
    print(f"Initial STI:")
    print(f"  {a1.get_name()}: {bank.get_attention_value(a1).sti:.1f}")
    print(f"  {a2.get_name()}: {bank.get_attention_value(a2).sti:.1f}")
    print(f"  {a3.get_name()}: {bank.get_attention_value(a3).sti:.1f}")
    
    # Apply importance spreading
    spreader = at.ImportanceSpreading()
    spreader.spread(bank, [hebbian], 5.0)
    
    print(f"\nAfter importance spreading:")
    print(f"  {a1.get_name()}: {bank.get_attention_value(a1).sti:.1f}")
    print(f"  {a2.get_name()}: {bank.get_attention_value(a2).sti:.1f}")
    
    # Apply rent (decay)
    rent_agent = at.RentAgent(rent_rate=2.0)
    rent_agent.collect_rent(bank)
    
    print(f"\nAfter rent collection:")
    print(f"  {a1.get_name()}: {bank.get_attention_value(a1).sti:.1f}")
    print(f"  {a2.get_name()}: {bank.get_attention_value(a2).sti:.1f}")

def example_5_nlu_text_processing():
    """Example 5: Natural language understanding"""
    print("\n=== Example 5: NLU Text Processing ===")
    
    space = at.AtomSpace()
    
    # Text processing
    text = "The cat sat on the mat. Dogs are mammals."
    processor = at.TextProcessor()
    
    tokens = processor.tokenize(text)
    print(f"Tokenized into {len(tokens)} tokens")
    
    sentences = processor.extract_sentences(text)
    print(f"Extracted {len(sentences)} sentences")
    
    # Entity recognition
    recognizer = at.EntityRecognizer()
    entities = recognizer.recognize(text)
    print(f"\nRecognized entities:")
    for entity in entities:
        print(f"  {entity.text} ({entity.type})")
    
    # Extract to knowledge graph
    extractor = at.SemanticExtractor(space)
    extractor.extract_from_text(text)
    
    print(f"\nExtracted {len(space)} atoms to knowledge graph")
    
    # Generate text from atoms
    generator = at.LanguageGenerator(space)
    cat_atom = space.get_atom(at.AtomType.CONCEPT_NODE, "cat")
    if cat_atom:
        description = generator.generate_from_atom(cat_atom)
        print(f"Generated: {description}")

def example_6_vision_processing():
    """Example 6: Visual perception"""
    print("\n=== Example 6: Vision Processing ===")
    
    space = at.AtomSpace()
    
    # Simulate detected objects
    obj1 = at.DetectedObject("cat", at.BoundingBox(10, 20, 50, 60), 0.95)
    obj2 = at.DetectedObject("mat", at.BoundingBox(5, 70, 100, 30), 0.90)
    
    print(f"Detected objects:")
    print(f"  {obj1.label} at ({obj1.bbox.x}, {obj1.bbox.y}) confidence={obj1.confidence}")
    print(f"  {obj2.label} at ({obj2.bbox.x}, {obj2.bbox.y}) confidence={obj2.confidence}")
    
    # Spatial analysis
    analyzer = at.SpatialAnalyzer()
    relations = analyzer.analyze_spatial_relations([obj1, obj2])
    
    print(f"\nSpatial relations:")
    for rel in relations:
        print(f"  {rel.object1} {rel.relation_type} {rel.object2} (conf={rel.confidence:.2f})")
    
    # Build scene graph
    scene = at.SceneUnderstanding(space)
    scene.build_scene_graph([obj1, obj2], relations)
    
    print(f"\nBuilt scene graph with {len(space)} atoms")
    
    # Describe scene
    description = scene.describe_scene([obj1, obj2], relations)
    print(f"Scene description: {description}")

def example_7_cognitive_engine():
    """Example 7: Integrated cognitive engine"""
    print("\n=== Example 7: Cognitive Engine ===")
    
    space = at.AtomSpace()
    engine = at.CognitiveEngine(space)
    
    # Setup components
    time_server = at.TimeServer()
    attention_bank = at.AttentionBank()
    forward_chainer = at.ForwardChainer(space)
    
    engine.set_time_server(time_server)
    engine.set_attention_bank(attention_bank)
    engine.set_forward_chainer(forward_chainer)
    
    # Enable ECAN
    engine.enable_economic_attention(rent_rate=1.0, wage_rate=1.0)
    engine.enable_forgetting(threshold=-50.0)
    
    # Add knowledge
    a = at.create_concept_node(space, "A")
    b = at.create_concept_node(space, "B")
    attention_bank.set_attention_value(a, at.AttentionValue(100.0, 50.0, 10.0))
    attention_bank.set_attention_value(b, at.AttentionValue(80.0, 40.0, 8.0))
    
    print(f"Initial state: {len(space)} atoms")
    
    # Run cognitive cycles
    engine.run(num_cycles=5, inference_per_cycle=5)
    
    metrics = engine.get_metrics()
    print(f"\nAfter {engine.get_cycle_count()} cycles:")
    print(f"  Atoms: {metrics['atom_count']}")
    print(f"  Avg STI: {metrics['avg_sti']:.2f}")

def example_8_tensor_logic():
    """Example 8: GPU-accelerated batch operations"""
    print("\n=== Example 8: Tensor Logic Engine ===")
    
    engine = at.TensorLogicEngine()
    
    # Batch logical operations
    tv1 = torch.tensor([[0.9, 0.8], [0.85, 0.75], [0.8, 0.7]])
    tv2 = torch.tensor([[0.95, 0.85], [0.9, 0.8], [0.85, 0.75]])
    
    # Batch AND
    and_result = engine.batch_and([tv1, tv2])
    print(f"Batch AND result shape: {and_result.shape}")
    print(f"First result: {and_result[0]}")
    
    # Batch deduction
    ab_tvs = torch.tensor([[0.9, 0.8], [0.85, 0.75]])
    bc_tvs = torch.tensor([[0.95, 0.85], [0.9, 0.8]])
    deduction_result = engine.batch_deduction(ab_tvs, bc_tvs)
    
    print(f"\nBatch deduction:")
    print(f"  Input: {ab_tvs.shape} premises")
    print(f"  Output: {deduction_result.shape} conclusions")
    
    # Statistics
    stats = engine.compute_statistics(tv1)
    print(f"\nTruth value statistics:")
    print(f"  Mean strength: {stats['mean_strength']:.3f}")
    print(f"  Mean confidence: {stats['mean_confidence']:.3f}")

def main():
    """Run all advanced examples"""
    print("ATenSpace Advanced Python Examples")
    print("=" * 60)
    
    try:
        example_1_pln_reasoning()
        example_2_pattern_matching()
        example_3_forward_chaining()
        example_4_ecan_attention()
        example_5_nlu_text_processing()
        example_6_vision_processing()
        example_7_cognitive_engine()
        example_8_tensor_logic()
        
        print("\n" + "=" * 60)
        print("All advanced examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
