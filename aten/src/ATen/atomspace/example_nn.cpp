/**
 * ATenNN Examples - Neural Network Integration with ATenSpace
 * 
 * This example demonstrates how to integrate pre-trained neural models
 * with the cognitive architecture for neuro-symbolic AI.
 * 
 * Scenarios covered:
 * 1. BERT for concept embeddings
 * 2. GPT for knowledge generation
 * 3. ViT for visual grounding
 * 4. YOLO for object detection
 * 5. Neuro-symbolic reasoning
 * 6. Attention bridging
 * 7. End-to-end cognitive workflows
 */

#include <ATen/atomspace/ATenSpace.h>
#include <iostream>
#include <vector>

using namespace at::atomspace;
using namespace at::atomspace::nn;

// ============================================================================
// Example 1: BERT for Concept Embeddings
// ============================================================================

void example1_bert_embeddings() {
    std::cout << "\n=== Example 1: BERT for Concept Embeddings ===" << std::endl;
    
    // Create AtomSpace
    AtomSpace space;
    
    // Configure BERT model
    ModelConfig bert_config("bert-base-uncased", "bert");
    bert_config.hidden_size = 768;
    bert_config.num_layers = 12;
    bert_config.num_heads = 12;
    bert_config.device = torch::kCPU;
    
    // Register pre-trained models
    registerPretrainedModels();
    
    // Load BERT model
    auto& registry = ModelRegistry::getInstance();
    auto bert = registry.loadModel(bert_config);
    
    std::cout << "Loaded model: " << bert->getName() << std::endl;
    
    // Create concept nodes
    auto cat = createConceptNode(space, "cat");
    auto dog = createConceptNode(space, "dog");
    auto bird = createConceptNode(space, "bird");
    
    // Simulate tokenization (in practice, use proper tokenizer)
    // Token IDs for "cat", "dog", "bird"
    auto cat_tokens = torch::tensor({{101, 4937, 102}});  // [CLS] cat [SEP]
    auto dog_tokens = torch::tensor({{101, 3899, 102}});  // [CLS] dog [SEP]
    auto bird_tokens = torch::tensor({{101, 4743, 102}}); // [CLS] bird [SEP]
    
    // Extract embeddings using BERT
    PerformanceMonitor monitor;
    EmbeddingExtractor extractor(EmbeddingExtractor::Strategy::CLS_TOKEN);
    
    monitor.startInference();
    auto cat_hidden = bert->forward(cat_tokens);
    auto cat_embedding = extractor.extract(cat_hidden);
    monitor.endInference(3);
    
    monitor.startInference();
    auto dog_hidden = bert->forward(dog_tokens);
    auto dog_embedding = extractor.extract(dog_hidden);
    monitor.endInference(3);
    
    monitor.startInference();
    auto bird_hidden = bert->forward(bird_tokens);
    auto bird_embedding = extractor.extract(bird_hidden);
    monitor.endInference(3);
    
    // Attach embeddings to nodes
    extractor.attachToNode(cat, cat_embedding.squeeze(0));
    extractor.attachToNode(dog, dog_embedding.squeeze(0));
    extractor.attachToNode(bird, bird_embedding.squeeze(0));
    
    std::cout << "Attached BERT embeddings to concepts" << std::endl;
    
    // Query similar concepts using neural embeddings
    auto query_embedding = cat->getEmbedding();
    auto similar = space.querySimilar(query_embedding, 2);
    
    std::cout << "\nConcepts similar to 'cat':" << std::endl;
    for (const auto& [atom, similarity] : similar) {
        std::cout << "  " << atom->toString() << " (similarity: " 
                  << similarity << ")" << std::endl;
    }
    
    monitor.printSummary();
}

// ============================================================================
// Example 2: GPT for Knowledge Generation
// ============================================================================

void example2_gpt_generation() {
    std::cout << "\n=== Example 2: GPT for Knowledge Generation ===" << std::endl;
    
    AtomSpace space;
    
    // Configure GPT model
    ModelConfig gpt_config("gpt2", "gpt");
    gpt_config.hidden_size = 768;
    gpt_config.num_layers = 12;
    gpt_config.device = torch::kCPU;
    
    // Load GPT model
    auto& registry = ModelRegistry::getInstance();
    auto gpt = std::static_pointer_cast<GPTModel>(registry.loadModel(gpt_config));
    
    std::cout << "Loaded model: " << gpt->getName() << std::endl;
    
    // Generate text from prompt
    auto prompt_tokens = torch::tensor({{464, 2258, 318}});  // "The cat is"
    
    PerformanceMonitor monitor;
    monitor.startInference();
    auto generated = gpt->generate(prompt_tokens, 10);
    monitor.endInference(10);
    
    std::cout << "Generated tokens shape: " << generated.sizes() << std::endl;
    
    // In practice, decode tokens back to text and create knowledge
    // For now, just demonstrate the generation capability
    std::cout << "Text generation complete" << std::endl;
    
    monitor.printSummary();
}

// ============================================================================
// Example 3: ViT for Visual Grounding
// ============================================================================

void example3_vit_visual_grounding() {
    std::cout << "\n=== Example 3: ViT for Visual Grounding ===" << std::endl;
    
    AtomSpace space;
    
    // Configure ViT model
    ModelConfig vit_config("vit-base", "vit");
    vit_config.hidden_size = 768;
    vit_config.num_layers = 12;
    vit_config.device = torch::kCPU;
    
    // Load ViT model
    auto& registry = ModelRegistry::getInstance();
    auto vit = std::static_pointer_cast<ViTModel>(registry.loadModel(vit_config));
    
    std::cout << "Loaded model: " << vit->getName() << std::endl;
    
    // Create visual concept nodes
    auto image_concept = createConceptNode(space, "cat_image");
    
    // Simulate image input [1, 3, 224, 224]
    auto image = torch::randn({1, 3, 224, 224});
    
    PerformanceMonitor monitor;
    monitor.startInference();
    auto visual_embedding = vit->extractEmbeddings(image);
    monitor.endInference();
    
    // Attach visual embedding to concept
    image_concept->setEmbedding(visual_embedding.squeeze(0));
    
    std::cout << "Attached visual embedding to concept" << std::endl;
    std::cout << "Embedding shape: " << visual_embedding.sizes() << std::endl;
    
    // Create similarity link between visual and linguistic concepts
    auto cat_text = createConceptNode(space, "cat");
    auto similarity_link = createSimilarityLink(space, image_concept, cat_text);
    
    std::cout << "Created neuro-symbolic grounding link" << std::endl;
    
    monitor.printSummary();
}

// ============================================================================
// Example 4: YOLO for Object Detection
// ============================================================================

void example4_yolo_detection() {
    std::cout << "\n=== Example 4: YOLO for Object Detection ===" << std::endl;
    
    AtomSpace space;
    
    // Configure YOLO model
    ModelConfig yolo_config("yolov5", "yolo");
    yolo_config.device = torch::kCPU;
    
    // Load YOLO model
    auto& registry = ModelRegistry::getInstance();
    auto yolo = std::static_pointer_cast<YOLOModel>(registry.loadModel(yolo_config));
    
    std::cout << "Loaded model: " << yolo->getName() << std::endl;
    
    // Simulate image input
    auto image = torch::randn({3, 640, 640});
    
    PerformanceMonitor monitor;
    monitor.startInference();
    auto detected_objects = yolo->detectObjects(image, space, 0.5);
    monitor.endInference();
    
    std::cout << "Detected " << detected_objects.size() << " objects" << std::endl;
    
    // In practice, this would create concept nodes for each detected object
    // and spatial relation links for their locations
    
    monitor.printSummary();
}

// ============================================================================
// Example 5: Neuro-Symbolic Reasoning
// ============================================================================

void example5_neurosymbolic_reasoning() {
    std::cout << "\n=== Example 5: Neuro-Symbolic Reasoning ===" << std::endl;
    
    AtomSpace space;
    
    // Create symbolic knowledge
    auto mammal = createConceptNode(space, "mammal");
    auto cat = createConceptNode(space, "cat");
    auto dog = createConceptNode(space, "dog");
    
    auto inheritance1 = createInheritanceLink(space, cat, mammal);
    auto inheritance2 = createInheritanceLink(space, dog, mammal);
    
    // Add PLN truth values
    inheritance1->setTruthValue(torch::tensor({0.95f, 0.9f}));
    inheritance2->setTruthValue(torch::tensor({0.95f, 0.9f}));
    
    std::cout << "Created symbolic knowledge base" << std::endl;
    
    // Add neural embeddings from BERT
    ModelConfig bert_config("bert-base", "bert");
    registerPretrainedModels();
    auto& registry = ModelRegistry::getInstance();
    auto bert = registry.loadModel(bert_config);
    
    // Extract and attach embeddings
    EmbeddingExtractor extractor(EmbeddingExtractor::Strategy::CLS_TOKEN);
    
    auto mammal_tokens = torch::tensor({{101, 15718, 102}});
    auto mammal_hidden = bert->forward(mammal_tokens);
    auto mammal_embedding = extractor.extract(mammal_hidden);
    mammal->setEmbedding(mammal_embedding.squeeze(0));
    
    auto cat_tokens = torch::tensor({{101, 4937, 102}});
    auto cat_hidden = bert->forward(cat_tokens);
    auto cat_embedding = extractor.extract(cat_hidden);
    cat->setEmbedding(cat_embedding.squeeze(0));
    
    std::cout << "Attached neural embeddings" << std::endl;
    
    // Perform hybrid reasoning
    // 1. Use symbolic PLN for logical inference
    ForwardChainer chainer(space);
    // Add inference rules here...
    
    // 2. Use neural similarity for concept discovery
    auto similar_to_cat = space.querySimilar(cat->getEmbedding(), 5);
    
    std::cout << "\nNeuro-symbolic integration complete:" << std::endl;
    std::cout << "- Symbolic reasoning: " << space.getNumAtoms() << " atoms" << std::endl;
    std::cout << "- Neural similarity: " << similar_to_cat.size() << " results" << std::endl;
}

// ============================================================================
// Example 6: Attention Bridging
// ============================================================================

void example6_attention_bridging() {
    std::cout << "\n=== Example 6: Attention Bridging ===" << std::endl;
    
    AtomSpace space;
    AttentionBank attention_bank;
    
    // Create concepts
    auto cat = createConceptNode(space, "cat");
    auto dog = createConceptNode(space, "dog");
    auto bird = createConceptNode(space, "bird");
    auto fish = createConceptNode(space, "fish");
    
    std::vector<std::shared_ptr<Atom>> atoms = {cat, dog, bird, fish};
    
    // Initialize attention values
    for (auto atom : atoms) {
        attention_bank.setAttentionValue(atom, AttentionValue(50.0f, 50.0f, 10.0f));
    }
    
    std::cout << "Initialized attention bank" << std::endl;
    
    // Simulate neural attention scores from a model
    auto attention_scores = torch::tensor({0.7f, 0.8f, 0.3f, 0.2f});
    
    // Bridge neural attention to ECAN
    AttentionBridge bridge(attention_bank);
    bridge.mapAttentionToSTI(atoms, attention_scores, 100.0f);
    
    std::cout << "\nAttention values after neural bridging:" << std::endl;
    for (auto atom : atoms) {
        auto av = attention_bank.getAttentionValue(atom);
        std::cout << "  " << atom->toString() << ": STI=" << av.sti << std::endl;
    }
    
    // Extract attentional focus
    auto focus = bridge.extractFocus(atoms, attention_scores, 2);
    
    std::cout << "\nAttentional focus (top 2):" << std::endl;
    for (auto atom : focus) {
        std::cout << "  " << atom->toString() << std::endl;
    }
}

// ============================================================================
// Example 7: End-to-End Cognitive Workflow
// ============================================================================

void example7_cognitive_workflow() {
    std::cout << "\n=== Example 7: End-to-End Cognitive Workflow ===" << std::endl;
    
    // Initialize components
    AtomSpace space;
    AttentionBank attention_bank;
    TimeServer time_server;
    
    std::cout << "Initialized cognitive architecture" << std::endl;
    
    // Register models
    registerPretrainedModels();
    
    // Phase 1: Visual Perception (ViT)
    std::cout << "\n[Phase 1: Visual Perception]" << std::endl;
    ModelConfig vit_config("vit-base", "vit");
    auto& registry = ModelRegistry::getInstance();
    auto vit = std::static_pointer_cast<ViTModel>(registry.loadModel(vit_config));
    
    auto image = torch::randn({1, 3, 224, 224});
    auto visual_embedding = vit->extractEmbeddings(image);
    
    auto visual_concept = createConceptNode(space, "perceived_object");
    visual_concept->setEmbedding(visual_embedding.squeeze(0));
    time_server.recordCreation(visual_concept);
    
    std::cout << "  Created visual concept node" << std::endl;
    
    // Phase 2: Language Understanding (BERT)
    std::cout << "\n[Phase 2: Language Understanding]" << std::endl;
    ModelConfig bert_config("bert-base", "bert");
    auto bert = registry.loadModel(bert_config);
    
    auto text_tokens = torch::tensor({{101, 4937, 102}});  // "cat"
    auto text_hidden = bert->forward(text_tokens);
    EmbeddingExtractor extractor;
    auto text_embedding = extractor.extract(text_hidden);
    
    auto linguistic_concept = createConceptNode(space, "cat");
    linguistic_concept->setEmbedding(text_embedding.squeeze(0));
    
    std::cout << "  Created linguistic concept node" << std::endl;
    
    // Phase 3: Multi-modal Grounding
    std::cout << "\n[Phase 3: Multi-modal Grounding]" << std::endl;
    auto grounding_link = createSimilarityLink(space, visual_concept, linguistic_concept);
    
    // Calculate similarity between visual and linguistic embeddings
    auto visual_emb = visual_concept->getEmbedding();
    auto ling_emb = linguistic_concept->getEmbedding();
    auto similarity = torch::cosine_similarity(visual_emb, ling_emb, 0);
    
    grounding_link->setTruthValue(torch::tensor({similarity.item<float>(), 0.8f}));
    
    std::cout << "  Grounded visual perception to language" << std::endl;
    std::cout << "  Similarity: " << similarity.item<float>() << std::endl;
    
    // Phase 4: Attention Allocation
    std::cout << "\n[Phase 4: Attention Allocation]" << std::endl;
    attention_bank.setAttentionValue(visual_concept, AttentionValue(80.0f, 60.0f, 20.0f));
    attention_bank.setAttentionValue(linguistic_concept, AttentionValue(90.0f, 70.0f, 25.0f));
    attention_bank.setAttentionValue(grounding_link, AttentionValue(85.0f, 65.0f, 22.0f));
    
    auto focus = attention_bank.getAttentionalFocus();
    std::cout << "  Attentional focus: " << focus.size() << " atoms" << std::endl;
    
    // Phase 5: Reasoning
    std::cout << "\n[Phase 5: Symbolic Reasoning]" << std::endl;
    auto mammal = createConceptNode(space, "mammal");
    auto inheritance = createInheritanceLink(space, linguistic_concept, mammal);
    inheritance->setTruthValue(torch::tensor({0.95f, 0.9f}));
    
    std::cout << "  Added symbolic knowledge" << std::endl;
    
    // Summary
    std::cout << "\n[Workflow Summary]" << std::endl;
    std::cout << "  Total atoms: " << space.getNumAtoms() << std::endl;
    std::cout << "  Visual concepts: 1" << std::endl;
    std::cout << "  Linguistic concepts: 2" << std::endl;
    std::cout << "  Grounding links: 1" << std::endl;
    std::cout << "  Reasoning links: 1" << std::endl;
    std::cout << "  Cognitive integration: Complete" << std::endl;
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << "ATenNN Integration Examples" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        example1_bert_embeddings();
        example2_gpt_generation();
        example3_vit_visual_grounding();
        example4_yolo_detection();
        example5_neurosymbolic_reasoning();
        example6_attention_bridging();
        example7_cognitive_workflow();
        
        std::cout << "\n=== All Examples Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
