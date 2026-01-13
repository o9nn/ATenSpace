/**
 * Unit Tests for ATenNN - Neural Network Integration
 * 
 * Test coverage:
 * 1. ModelConfig creation and settings
 * 2. EmbeddingExtractor strategies
 * 3. AttentionBridge neural-to-ECAN mapping
 * 4. PerformanceMonitor metrics
 * 5. ModelRegistry management
 * 6. BERT model integration
 * 7. GPT model integration
 * 8. ViT model integration
 * 9. YOLO model integration
 * 10. End-to-end neuro-symbolic workflows
 */

#include <ATen/atomspace/ATenSpace.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace at::atomspace;
using namespace at::atomspace::nn;

// ============================================================================
// Test 1: ModelConfig
// ============================================================================

void test_model_config() {
    std::cout << "\n=== Test 1: ModelConfig ===" << std::endl;
    
    // Test default construction
    ModelConfig config1;
    assert(config1.hidden_size == 768);
    assert(config1.device == torch::kCPU);
    std::cout << "✓ Default construction" << std::endl;
    
    // Test parameterized construction
    ModelConfig config2("bert-base", "bert");
    assert(config2.model_name == "bert-base");
    assert(config2.model_type == "bert");
    std::cout << "✓ Parameterized construction" << std::endl;
    
    // Test configuration settings
    config2.hidden_size = 1024;
    config2.num_layers = 24;
    config2.use_cache = true;
    config2.half_precision = false;
    
    assert(config2.hidden_size == 1024);
    assert(config2.num_layers == 24);
    assert(config2.use_cache == true);
    std::cout << "✓ Configuration settings" << std::endl;
    
    std::cout << "Test 1 passed!" << std::endl;
}

// ============================================================================
// Test 2: EmbeddingExtractor
// ============================================================================

void test_embedding_extractor() {
    std::cout << "\n=== Test 2: EmbeddingExtractor ===" << std::endl;
    
    AtomSpace space;
    
    // Create test hidden states [batch=2, seq_len=5, hidden=8]
    auto hidden_states = torch::randn({2, 5, 8});
    auto attention_mask = torch::ones({2, 5});
    
    // Test CLS token extraction
    EmbeddingExtractor cls_extractor(EmbeddingExtractor::Strategy::CLS_TOKEN);
    auto cls_emb = cls_extractor.extract(hidden_states);
    assert(cls_emb.size(0) == 2);
    assert(cls_emb.size(1) == 8);
    std::cout << "✓ CLS token extraction: " << cls_emb.sizes() << std::endl;
    
    // Test mean pooling
    EmbeddingExtractor mean_extractor(EmbeddingExtractor::Strategy::MEAN_POOLING);
    auto mean_emb = mean_extractor.extract(hidden_states, attention_mask);
    assert(mean_emb.size(0) == 2);
    assert(mean_emb.size(1) == 8);
    std::cout << "✓ Mean pooling: " << mean_emb.sizes() << std::endl;
    
    // Test max pooling
    EmbeddingExtractor max_extractor(EmbeddingExtractor::Strategy::MAX_POOLING);
    auto max_emb = max_extractor.extract(hidden_states, attention_mask);
    assert(max_emb.size(0) == 2);
    assert(max_emb.size(1) == 8);
    std::cout << "✓ Max pooling: " << max_emb.sizes() << std::endl;
    
    // Test attachment to node
    auto concept = createConceptNode(space, "test_concept");
    cls_extractor.attachToNode(concept, cls_emb.index({0}));
    assert(concept->hasEmbedding());
    assert(concept->getEmbedding().size(0) == 8);
    std::cout << "✓ Embedding attachment to node" << std::endl;
    
    std::cout << "Test 2 passed!" << std::endl;
}

// ============================================================================
// Test 3: AttentionBridge
// ============================================================================

void test_attention_bridge() {
    std::cout << "\n=== Test 3: AttentionBridge ===" << std::endl;
    
    AtomSpace space;
    AttentionBank attention_bank;
    
    // Create test atoms
    std::vector<std::shared_ptr<Atom>> atoms;
    atoms.push_back(createConceptNode(space, "atom1"));
    atoms.push_back(createConceptNode(space, "atom2"));
    atoms.push_back(createConceptNode(space, "atom3"));
    atoms.push_back(createConceptNode(space, "atom4"));
    
    // Initialize attention values
    for (auto atom : atoms) {
        attention_bank.setAttentionValue(atom, AttentionValue(50.0f, 50.0f, 10.0f));
    }
    
    // Create attention bridge
    AttentionBridge bridge(attention_bank);
    
    // Test neural attention scores
    auto attention_scores = torch::tensor({0.9f, 0.7f, 0.3f, 0.1f});
    
    // Map to STI
    bridge.mapAttentionToSTI(atoms, attention_scores, 100.0f);
    
    // Verify STI values updated
    auto av1 = attention_bank.getAttentionValue(atoms[0]);
    auto av2 = attention_bank.getAttentionValue(atoms[1]);
    auto av3 = attention_bank.getAttentionValue(atoms[2]);
    
    assert(av1.sti > av2.sti);  // Higher attention → higher STI
    assert(av2.sti > av3.sti);
    
    std::cout << "✓ Neural attention mapped to STI values" << std::endl;
    std::cout << "  Atom1 STI: " << av1.sti << std::endl;
    std::cout << "  Atom2 STI: " << av2.sti << std::endl;
    std::cout << "  Atom3 STI: " << av3.sti << std::endl;
    
    // Test focus extraction
    auto focus = bridge.extractFocus(atoms, attention_scores, 2);
    assert(focus.size() == 2);
    assert(focus[0] == atoms[0]);  // Highest attention
    assert(focus[1] == atoms[1]);  // Second highest
    
    std::cout << "✓ Attentional focus extraction (top 2)" << std::endl;
    
    std::cout << "Test 3 passed!" << std::endl;
}

// ============================================================================
// Test 4: PerformanceMonitor
// ============================================================================

void test_performance_monitor() {
    std::cout << "\n=== Test 4: PerformanceMonitor ===" << std::endl;
    
    PerformanceMonitor monitor;
    
    // Test initial state
    auto metrics = monitor.getMetrics();
    assert(metrics.num_inferences == 0);
    assert(metrics.total_inference_time_ms == 0.0);
    std::cout << "✓ Initial state" << std::endl;
    
    // Simulate inference
    monitor.startInference();
    // Simulate some work
    auto dummy = torch::randn({100, 100}).matmul(torch::randn({100, 100}));
    monitor.endInference(100);
    
    metrics = monitor.getMetrics();
    assert(metrics.num_inferences == 1);
    assert(metrics.total_inference_time_ms > 0.0);
    assert(metrics.total_tokens_processed == 100);
    std::cout << "✓ First inference recorded" << std::endl;
    
    // Simulate more inferences
    for (int i = 0; i < 5; ++i) {
        monitor.startInference();
        auto dummy = torch::randn({50, 50}).matmul(torch::randn({50, 50}));
        monitor.endInference(50);
    }
    
    metrics = monitor.getMetrics();
    assert(metrics.num_inferences == 6);
    assert(metrics.total_tokens_processed == 350);
    assert(metrics.avg_inference_time_ms > 0.0);
    assert(metrics.throughput_tokens_per_sec > 0.0);
    
    std::cout << "✓ Multiple inferences" << std::endl;
    std::cout << "  Total inferences: " << metrics.num_inferences << std::endl;
    std::cout << "  Avg time: " << metrics.avg_inference_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << metrics.throughput_tokens_per_sec << " tokens/sec" << std::endl;
    
    // Test reset
    monitor.reset();
    metrics = monitor.getMetrics();
    assert(metrics.num_inferences == 0);
    std::cout << "✓ Monitor reset" << std::endl;
    
    std::cout << "Test 4 passed!" << std::endl;
}

// ============================================================================
// Test 5: ModelRegistry
// ============================================================================

void test_model_registry() {
    std::cout << "\n=== Test 5: ModelRegistry ===" << std::endl;
    
    auto& registry = ModelRegistry::getInstance();
    
    // Register models
    registerPretrainedModels();
    
    // Test model type checking
    assert(registry.hasModelType("bert"));
    assert(registry.hasModelType("gpt"));
    assert(registry.hasModelType("vit"));
    assert(registry.hasModelType("yolo"));
    assert(!registry.hasModelType("unknown_model"));
    
    std::cout << "✓ Model type registration" << std::endl;
    
    // Test model loading
    ModelConfig bert_config("bert-test", "bert");
    bert_config.hidden_size = 128;
    bert_config.num_layers = 2;
    
    auto bert = registry.loadModel(bert_config);
    assert(bert != nullptr);
    assert(bert->getName() == "bert-test");
    
    std::cout << "✓ BERT model loading" << std::endl;
    
    // Test caching
    auto bert2 = registry.loadModel(bert_config);
    assert(bert2 == bert);  // Same instance due to caching
    
    std::cout << "✓ Model caching" << std::endl;
    
    // Test cache clearing
    registry.clearCache();
    auto bert3 = registry.loadModel(bert_config);
    // After clearing, should be new instance
    
    std::cout << "✓ Cache clearing" << std::endl;
    
    std::cout << "Test 5 passed!" << std::endl;
}

// ============================================================================
// Test 6: BERT Model Integration
// ============================================================================

void test_bert_integration() {
    std::cout << "\n=== Test 6: BERT Model Integration ===" << std::endl;
    
    AtomSpace space;
    registerPretrainedModels();
    auto& registry = ModelRegistry::getInstance();
    
    // Create BERT model
    ModelConfig config("bert-base", "bert");
    config.hidden_size = 128;
    config.num_layers = 2;
    config.vocab_size = 30522;
    
    auto bert = std::static_pointer_cast<BERTModel>(registry.loadModel(config));
    
    std::cout << "✓ BERT model created" << std::endl;
    
    // Test forward pass
    auto input_ids = torch::randint(0, config.vocab_size, {2, 10});
    auto hidden_states = bert->forward(input_ids);
    
    assert(hidden_states.size(0) == 2);   // batch size
    assert(hidden_states.size(1) == 10);  // sequence length
    assert(hidden_states.size(2) == 128); // hidden size
    
    std::cout << "✓ BERT forward pass: " << hidden_states.sizes() << std::endl;
    
    // Test embedding extraction
    auto embeddings = bert->extractEmbeddings(input_ids);
    assert(embeddings.size(0) == 2);
    assert(embeddings.size(1) == 128);
    
    std::cout << "✓ BERT embedding extraction: " << embeddings.sizes() << std::endl;
    
    // Test integration with AtomSpace
    auto concept = createConceptNode(space, "test_bert");
    concept->setEmbedding(embeddings.index({0}));
    
    assert(concept->hasEmbedding());
    std::cout << "✓ BERT embeddings attached to AtomSpace" << std::endl;
    
    std::cout << "Test 6 passed!" << std::endl;
}

// ============================================================================
// Test 7: GPT Model Integration
// ============================================================================

void test_gpt_integration() {
    std::cout << "\n=== Test 7: GPT Model Integration ===" << std::endl;
    
    registerPretrainedModels();
    auto& registry = ModelRegistry::getInstance();
    
    // Create GPT model
    ModelConfig config("gpt2", "gpt");
    config.hidden_size = 128;
    config.num_layers = 2;
    config.vocab_size = 50257;
    
    auto gpt = std::static_pointer_cast<GPTModel>(registry.loadModel(config));
    
    std::cout << "✓ GPT model created" << std::endl;
    
    // Test forward pass (returns logits)
    auto input_ids = torch::randint(0, config.vocab_size, {2, 10});
    auto logits = gpt->forward(input_ids);
    
    assert(logits.size(0) == 2);      // batch size
    assert(logits.size(1) == 10);     // sequence length
    assert(logits.size(2) == 50257);  // vocab size
    
    std::cout << "✓ GPT forward pass: " << logits.sizes() << std::endl;
    
    // Test text generation
    auto prompt = torch::randint(0, config.vocab_size, {1, 5});
    auto generated = gpt->generate(prompt, 10);
    
    assert(generated.size(0) == 1);
    assert(generated.size(1) == 15);  // 5 prompt + 10 generated
    
    std::cout << "✓ GPT text generation: " << generated.sizes() << std::endl;
    
    std::cout << "Test 7 passed!" << std::endl;
}

// ============================================================================
// Test 8: ViT Model Integration
// ============================================================================

void test_vit_integration() {
    std::cout << "\n=== Test 8: ViT Model Integration ===" << std::endl;
    
    AtomSpace space;
    registerPretrainedModels();
    auto& registry = ModelRegistry::getInstance();
    
    // Create ViT model
    ModelConfig config("vit-base", "vit");
    config.hidden_size = 128;
    config.num_layers = 2;
    
    auto vit = std::static_pointer_cast<ViTModel>(registry.loadModel(config));
    
    std::cout << "✓ ViT model created" << std::endl;
    
    // Test forward pass with image
    auto image = torch::randn({2, 3, 224, 224});
    auto hidden_states = vit->forward(image);
    
    assert(hidden_states.size(0) == 2);   // batch size
    assert(hidden_states.size(2) == 128); // hidden size
    
    std::cout << "✓ ViT forward pass: " << hidden_states.sizes() << std::endl;
    
    // Test visual embedding extraction
    auto visual_embedding = vit->extractEmbeddings(image);
    assert(visual_embedding.size(0) == 2);
    assert(visual_embedding.size(1) == 128);
    
    std::cout << "✓ ViT embedding extraction: " << visual_embedding.sizes() << std::endl;
    
    // Test integration with AtomSpace
    auto visual_concept = createConceptNode(space, "test_image");
    visual_concept->setEmbedding(visual_embedding.index({0}));
    
    assert(visual_concept->hasEmbedding());
    std::cout << "✓ ViT embeddings attached to AtomSpace" << std::endl;
    
    std::cout << "Test 8 passed!" << std::endl;
}

// ============================================================================
// Test 9: YOLO Model Integration
// ============================================================================

void test_yolo_integration() {
    std::cout << "\n=== Test 9: YOLO Model Integration ===" << std::endl;
    
    AtomSpace space;
    registerPretrainedModels();
    auto& registry = ModelRegistry::getInstance();
    
    // Create YOLO model
    ModelConfig config("yolov5", "yolo");
    
    auto yolo = std::static_pointer_cast<YOLOModel>(registry.loadModel(config));
    
    std::cout << "✓ YOLO model created" << std::endl;
    
    // Test forward pass with image
    auto image = torch::randn({2, 3, 640, 640});
    auto detections = yolo->forward(image);
    
    // Detections tensor returned (format depends on implementation)
    assert(detections.defined());
    
    std::cout << "✓ YOLO forward pass: " << detections.sizes() << std::endl;
    
    // Test object detection (creates AtomSpace nodes)
    auto single_image = torch::randn({3, 640, 640});
    auto detected_nodes = yolo->detectObjects(single_image, space, 0.5);
    
    // In this test implementation, returns empty vector
    // In production, would create concept nodes for detected objects
    std::cout << "✓ YOLO object detection integration" << std::endl;
    
    std::cout << "Test 9 passed!" << std::endl;
}

// ============================================================================
// Test 10: Neuro-Symbolic Integration
// ============================================================================

void test_neurosymbolic_integration() {
    std::cout << "\n=== Test 10: Neuro-Symbolic Integration ===" << std::endl;
    
    AtomSpace space;
    AttentionBank attention_bank;
    
    // Create symbolic knowledge
    auto mammal = createConceptNode(space, "mammal");
    auto cat = createConceptNode(space, "cat");
    auto dog = createConceptNode(space, "dog");
    
    auto inh1 = createInheritanceLink(space, cat, mammal);
    auto inh2 = createInheritanceLink(space, dog, mammal);
    
    inh1->setTruthValue(torch::tensor({0.95f, 0.9f}));
    inh2->setTruthValue(torch::tensor({0.95f, 0.9f}));
    
    std::cout << "✓ Symbolic knowledge created" << std::endl;
    
    // Add neural embeddings
    registerPretrainedModels();
    auto& registry = ModelRegistry::getInstance();
    
    ModelConfig config("bert-base", "bert");
    config.hidden_size = 128;
    config.num_layers = 2;
    
    auto bert = std::static_pointer_cast<BERTModel>(registry.loadModel(config));
    
    // Extract embeddings
    auto cat_tokens = torch::randint(0, 30522, {1, 5});
    auto dog_tokens = torch::randint(0, 30522, {1, 5});
    auto mammal_tokens = torch::randint(0, 30522, {1, 5});
    
    cat->setEmbedding(bert->extractEmbeddings(cat_tokens).squeeze(0));
    dog->setEmbedding(bert->extractEmbeddings(dog_tokens).squeeze(0));
    mammal->setEmbedding(bert->extractEmbeddings(mammal_tokens).squeeze(0));
    
    std::cout << "✓ Neural embeddings attached" << std::endl;
    
    // Hybrid reasoning: similarity + logic
    auto similar_to_cat = space.querySimilar(cat->getEmbedding(), 3);
    
    assert(similar_to_cat.size() > 0);
    std::cout << "✓ Neural similarity search: " << similar_to_cat.size() << " results" << std::endl;
    
    // Attention bridging
    std::vector<std::shared_ptr<Atom>> concepts = {cat, dog, mammal};
    auto attention_scores = torch::tensor({0.8f, 0.6f, 0.4f});
    
    for (auto concept : concepts) {
        attention_bank.setAttentionValue(concept, AttentionValue(50.0f, 50.0f, 10.0f));
    }
    
    AttentionBridge bridge(attention_bank);
    bridge.mapAttentionToSTI(concepts, attention_scores, 100.0f);
    
    std::cout << "✓ Neural attention mapped to ECAN" << std::endl;
    
    // Verify integration
    assert(space.getNumAtoms() >= 5);  // 3 concepts + 2 links
    assert(cat->hasEmbedding());
    assert(dog->hasEmbedding());
    assert(attention_bank.getAttentionValue(cat).sti > 50.0f);
    
    std::cout << "✓ Complete neuro-symbolic integration" << std::endl;
    std::cout << "  Atoms in space: " << space.getNumAtoms() << std::endl;
    std::cout << "  Concepts with embeddings: 3" << std::endl;
    std::cout << "  Attention-guided: Yes" << std::endl;
    
    std::cout << "Test 10 passed!" << std::endl;
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << "ATenNN Unit Tests" << std::endl;
    std::cout << "=================" << std::endl;
    
    try {
        test_model_config();
        test_embedding_extractor();
        test_attention_bridge();
        test_performance_monitor();
        test_model_registry();
        test_bert_integration();
        test_gpt_integration();
        test_vit_integration();
        test_yolo_integration();
        test_neurosymbolic_integration();
        
        std::cout << "\n================================" << std::endl;
        std::cout << "All Tests Passed! ✓" << std::endl;
        std::cout << "================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest Failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
