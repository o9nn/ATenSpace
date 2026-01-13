"""
ATenNN Python Examples - Neural Network Integration

This module demonstrates how to integrate pre-trained neural models
with ATenSpace cognitive architecture using Python.

Examples:
1. BERT embeddings for concepts
2. GPT text generation
3. ViT visual embeddings
4. Neuro-symbolic reasoning
5. Attention bridging
6. End-to-end cognitive workflow
"""

import torch
import atenspace as at
from typing import List, Dict, Tuple


def example1_bert_embeddings():
    """Example 1: BERT for Concept Embeddings"""
    print("\n=== Example 1: BERT for Concept Embeddings ===")
    
    # Create AtomSpace
    space = at.AtomSpace()
    
    # Configure BERT model
    config = at.nn.ModelConfig("bert-base-uncased", "bert")
    config.hidden_size = 768
    config.num_layers = 12
    config.device = torch.device("cpu")
    
    # Register and load model
    at.nn.register_pretrained_models()
    registry = at.nn.ModelRegistry.get_instance()
    bert = registry.load_model(config)
    
    print(f"Loaded model: {bert.get_name()}")
    
    # Create concept nodes
    cat = at.create_concept_node(space, "cat")
    dog = at.create_concept_node(space, "dog")
    bird = at.create_concept_node(space, "bird")
    
    # Simulate tokenization
    cat_tokens = torch.tensor([[101, 4937, 102]])  # [CLS] cat [SEP]
    dog_tokens = torch.tensor([[101, 3899, 102]])  # [CLS] dog [SEP]
    bird_tokens = torch.tensor([[101, 4743, 102]]) # [CLS] bird [SEP]
    
    # Extract embeddings
    extractor = at.nn.EmbeddingExtractor(
        at.nn.EmbeddingExtractor.Strategy.CLS_TOKEN
    )
    monitor = at.nn.PerformanceMonitor()
    
    monitor.start_inference()
    cat_hidden = bert.forward(cat_tokens)
    cat_embedding = extractor.extract(cat_hidden)
    monitor.end_inference(3)
    
    monitor.start_inference()
    dog_hidden = bert.forward(dog_tokens)
    dog_embedding = extractor.extract(dog_hidden)
    monitor.end_inference(3)
    
    monitor.start_inference()
    bird_hidden = bert.forward(bird_tokens)
    bird_embedding = extractor.extract(bird_hidden)
    monitor.end_inference(3)
    
    # Attach embeddings to nodes
    extractor.attach_to_node(cat, cat_embedding.squeeze(0))
    extractor.attach_to_node(dog, dog_embedding.squeeze(0))
    extractor.attach_to_node(bird, bird_embedding.squeeze(0))
    
    print("Attached BERT embeddings to concepts")
    
    # Query similar concepts
    query_embedding = cat.get_embedding()
    similar = space.query_similar(query_embedding, k=2)
    
    print("\nConcepts similar to 'cat':")
    for atom, similarity in similar:
        print(f"  {atom.to_string()} (similarity: {similarity:.4f})")
    
    monitor.print_summary()


def example2_multimodal_grounding():
    """Example 2: Multi-modal Grounding with Vision + Language"""
    print("\n=== Example 2: Multi-modal Grounding ===")
    
    space = at.AtomSpace()
    
    # Register models
    at.nn.register_pretrained_models()
    registry = at.nn.ModelRegistry.get_instance()
    
    # Load ViT for vision
    vit_config = at.nn.ModelConfig("vit-base", "vit")
    vit_config.hidden_size = 768
    vit_config.device = torch.device("cpu")
    vit = registry.load_model(vit_config)
    
    # Load BERT for language
    bert_config = at.nn.ModelConfig("bert-base", "bert")
    bert_config.hidden_size = 768
    bert_config.device = torch.device("cpu")
    bert = registry.load_model(bert_config)
    
    print("Loaded vision and language models")
    
    # Visual perception
    image = torch.randn(1, 3, 224, 224)
    visual_hidden = vit.forward(image)
    
    extractor = at.nn.EmbeddingExtractor()
    visual_embedding = extractor.extract(visual_hidden)
    
    visual_concept = at.create_concept_node(space, "perceived_cat_image")
    visual_concept.set_embedding(visual_embedding.squeeze(0))
    
    # Language understanding
    text_tokens = torch.tensor([[101, 4937, 102]])  # "cat"
    text_hidden = bert.forward(text_tokens)
    text_embedding = extractor.extract(text_hidden)
    
    linguistic_concept = at.create_concept_node(space, "cat")
    linguistic_concept.set_embedding(text_embedding.squeeze(0))
    
    # Create grounding link
    grounding_link = at.create_similarity_link(
        space, visual_concept, linguistic_concept
    )
    
    # Calculate similarity
    visual_emb = visual_concept.get_embedding()
    ling_emb = linguistic_concept.get_embedding()
    similarity = torch.cosine_similarity(visual_emb, ling_emb, dim=0)
    
    grounding_link.set_truth_value(
        torch.tensor([similarity.item(), 0.8])
    )
    
    print(f"\nGrounded visual to linguistic concept")
    print(f"Similarity: {similarity.item():.4f}")
    print(f"Total atoms: {space.get_num_atoms()}")


def example3_attention_bridging():
    """Example 3: Bridge Neural Attention to ECAN"""
    print("\n=== Example 3: Attention Bridging ===")
    
    space = at.AtomSpace()
    attention_bank = at.AttentionBank()
    
    # Create concepts
    concepts = [
        at.create_concept_node(space, "cat"),
        at.create_concept_node(space, "dog"),
        at.create_concept_node(space, "bird"),
        at.create_concept_node(space, "fish"),
    ]
    
    # Initialize attention values
    for concept in concepts:
        attention_bank.set_attention_value(
            concept, at.AttentionValue(50.0, 50.0, 10.0)
        )
    
    print("Initialized attention bank")
    
    # Simulate neural attention scores
    attention_scores = torch.tensor([0.7, 0.8, 0.3, 0.2])
    
    # Bridge neural attention to ECAN
    bridge = at.nn.AttentionBridge(attention_bank)
    bridge.map_attention_to_sti(concepts, attention_scores, scale=100.0)
    
    print("\nAttention values after neural bridging:")
    for concept in concepts:
        av = attention_bank.get_attention_value(concept)
        print(f"  {concept.to_string()}: STI={av.sti:.2f}")
    
    # Extract focus
    focus = bridge.extract_focus(concepts, attention_scores, top_k=2)
    
    print("\nAttentional focus (top 2):")
    for atom in focus:
        print(f"  {atom.to_string()}")


def example4_neurosymbolic_reasoning():
    """Example 4: Neuro-Symbolic Integration"""
    print("\n=== Example 4: Neuro-Symbolic Reasoning ===")
    
    space = at.AtomSpace()
    
    # Build symbolic knowledge base
    mammal = at.create_concept_node(space, "mammal")
    cat = at.create_concept_node(space, "cat")
    dog = at.create_concept_node(space, "dog")
    
    # Add inheritance relationships
    inh1 = at.create_inheritance_link(space, cat, mammal)
    inh2 = at.create_inheritance_link(space, dog, mammal)
    
    # Set PLN truth values
    inh1.set_truth_value(torch.tensor([0.95, 0.9]))
    inh2.set_truth_value(torch.tensor([0.95, 0.9]))
    
    print("Created symbolic knowledge base")
    
    # Add neural embeddings
    at.nn.register_pretrained_models()
    registry = at.nn.ModelRegistry.get_instance()
    
    bert_config = at.nn.ModelConfig("bert-base", "bert")
    bert = registry.load_model(bert_config)
    
    extractor = at.nn.EmbeddingExtractor()
    
    # Embed concepts
    concepts_tokens = {
        mammal: torch.tensor([[101, 15718, 102]]),
        cat: torch.tensor([[101, 4937, 102]]),
        dog: torch.tensor([[101, 3899, 102]]),
    }
    
    for concept, tokens in concepts_tokens.items():
        hidden = bert.forward(tokens)
        embedding = extractor.extract(hidden)
        concept.set_embedding(embedding.squeeze(0))
    
    print("Attached neural embeddings")
    
    # Hybrid reasoning
    print("\nHybrid reasoning capabilities:")
    
    # 1. Symbolic: PLN inference
    forward_chainer = at.ForwardChainer(space)
    print("  - Symbolic inference engine: Ready")
    
    # 2. Neural: Similarity queries
    similar_to_cat = space.query_similar(cat.get_embedding(), k=3)
    print(f"  - Neural similarity search: {len(similar_to_cat)} results")
    
    # 3. Integration
    print(f"  - Total knowledge atoms: {space.get_num_atoms()}")
    print("  - Neuro-symbolic integration: Complete")


def example5_cognitive_workflow():
    """Example 5: Complete Cognitive Workflow"""
    print("\n=== Example 5: End-to-End Cognitive Workflow ===")
    
    # Initialize cognitive architecture
    space = at.AtomSpace()
    attention_bank = at.AttentionBank()
    time_server = at.TimeServer()
    
    print("Initialized cognitive architecture")
    
    # Register models
    at.nn.register_pretrained_models()
    registry = at.nn.ModelRegistry.get_instance()
    
    # Phase 1: Visual Perception
    print("\n[Phase 1: Visual Perception]")
    vit_config = at.nn.ModelConfig("vit-base", "vit")
    vit = registry.load_model(vit_config)
    
    image = torch.randn(1, 3, 224, 224)
    visual_hidden = vit.forward(image)
    
    extractor = at.nn.EmbeddingExtractor()
    visual_embedding = extractor.extract(visual_hidden)
    
    visual_concept = at.create_concept_node(space, "perceived_object")
    visual_concept.set_embedding(visual_embedding.squeeze(0))
    time_server.record_creation(visual_concept)
    
    print("  Created visual concept node")
    
    # Phase 2: Language Understanding
    print("\n[Phase 2: Language Understanding]")
    bert_config = at.nn.ModelConfig("bert-base", "bert")
    bert = registry.load_model(bert_config)
    
    text_tokens = torch.tensor([[101, 4937, 102]])  # "cat"
    text_hidden = bert.forward(text_tokens)
    text_embedding = extractor.extract(text_hidden)
    
    linguistic_concept = at.create_concept_node(space, "cat")
    linguistic_concept.set_embedding(text_embedding.squeeze(0))
    
    print("  Created linguistic concept node")
    
    # Phase 3: Multi-modal Grounding
    print("\n[Phase 3: Multi-modal Grounding]")
    grounding_link = at.create_similarity_link(
        space, visual_concept, linguistic_concept
    )
    
    visual_emb = visual_concept.get_embedding()
    ling_emb = linguistic_concept.get_embedding()
    similarity = torch.cosine_similarity(visual_emb, ling_emb, dim=0)
    
    grounding_link.set_truth_value(
        torch.tensor([similarity.item(), 0.8])
    )
    
    print(f"  Grounded visual to language")
    print(f"  Similarity: {similarity.item():.4f}")
    
    # Phase 4: Attention Allocation
    print("\n[Phase 4: Attention Allocation]")
    attention_bank.set_attention_value(
        visual_concept, at.AttentionValue(80.0, 60.0, 20.0)
    )
    attention_bank.set_attention_value(
        linguistic_concept, at.AttentionValue(90.0, 70.0, 25.0)
    )
    attention_bank.set_attention_value(
        grounding_link, at.AttentionValue(85.0, 65.0, 22.0)
    )
    
    focus = attention_bank.get_attentional_focus()
    print(f"  Attentional focus: {len(focus)} atoms")
    
    # Phase 5: Symbolic Reasoning
    print("\n[Phase 5: Symbolic Reasoning]")
    mammal = at.create_concept_node(space, "mammal")
    inheritance = at.create_inheritance_link(space, linguistic_concept, mammal)
    inheritance.set_truth_value(torch.tensor([0.95, 0.9]))
    
    print("  Added symbolic knowledge")
    
    # Summary
    print("\n[Workflow Summary]")
    print(f"  Total atoms: {space.get_num_atoms()}")
    print("  Visual concepts: 1")
    print("  Linguistic concepts: 2")
    print("  Grounding links: 1")
    print("  Reasoning links: 1")
    print("  Cognitive integration: Complete ✓")


def example6_performance_monitoring():
    """Example 6: Performance Monitoring"""
    print("\n=== Example 6: Performance Monitoring ===")
    
    space = at.AtomSpace()
    monitor = at.nn.PerformanceMonitor()
    
    # Register models
    at.nn.register_pretrained_models()
    registry = at.nn.ModelRegistry.get_instance()
    
    bert_config = at.nn.ModelConfig("bert-base", "bert")
    bert = registry.load_model(bert_config)
    
    print("Running performance benchmark...")
    
    # Run multiple inferences
    num_runs = 10
    batch_size = 4
    
    for i in range(num_runs):
        tokens = torch.randint(0, 30522, (batch_size, 128))
        
        monitor.start_inference()
        outputs = bert.forward(tokens)
        monitor.end_inference(num_tokens=batch_size * 128)
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    print(f"\nPerformance Metrics:")
    print(f"  Total inferences: {metrics.num_inferences}")
    print(f"  Avg inference time: {metrics.avg_inference_time_ms:.2f} ms")
    print(f"  Total tokens: {metrics.total_tokens_processed}")
    print(f"  Throughput: {metrics.throughput_tokens_per_sec:.2f} tokens/sec")
    
    monitor.print_summary()


def main():
    """Run all examples"""
    print("ATenNN Python Integration Examples")
    print("=" * 50)
    
    try:
        example1_bert_embeddings()
        example2_multimodal_grounding()
        example3_attention_bridging()
        example4_neurosymbolic_reasoning()
        example5_cognitive_workflow()
        example6_performance_monitoring()
        
        print("\n" + "=" * 50)
        print("All Examples Complete ✓")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
