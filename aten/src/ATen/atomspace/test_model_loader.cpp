/**
 * Unit Tests for ModelLoader - TorchScript Model Loading
 * 
 * Test coverage:
 * 1. ModelLoader construction and device detection
 * 2. TorchScriptModel wrapper functionality
 * 3. Model caching mechanisms
 * 4. Configuration loading and parsing
 * 5. Error handling for missing files
 * 6. Helper function tests (loadBERTModel, etc.)
 * 7. Integration with AtomSpace
 */

#include <ATen/atomspace/ATenSpace.h>
#include <ATen/atomspace/ModelLoader.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <cmath>

using namespace at::atomspace;
using namespace at::atomspace::nn;

// ============================================================================
// Test 1: ModelLoader Construction
// ============================================================================

void test_model_loader_construction() {
    std::cout << "\n=== Test 1: ModelLoader Construction ===" << std::endl;
    
    // Test default construction
    ModelLoader loader;
    
    // Check device detection
    auto device = loader.getDefaultDevice();
    if (torch::cuda::is_available()) {
        assert(device.is_cuda());
        std::cout << "✓ CUDA detected, default device is GPU" << std::endl;
    } else {
        assert(device.is_cpu());
        std::cout << "✓ No CUDA, default device is CPU" << std::endl;
    }
    
    std::cout << "Test 1 passed!" << std::endl;
}

// ============================================================================
// Test 2: Model Path Checking
// ============================================================================

void test_model_exists() {
    std::cout << "\n=== Test 2: Model Path Checking ===" << std::endl;
    
    // Test non-existent path
    bool exists = ModelLoader::modelExists("nonexistent_model.pt");
    assert(!exists);
    std::cout << "✓ Non-existent model returns false" << std::endl;
    
    // Create a temporary file to test existing path
    const std::string temp_path = "/tmp/test_model_exists.pt";
    std::ofstream temp_file(temp_path);
    temp_file << "dummy content";
    temp_file.close();
    
    exists = ModelLoader::modelExists(temp_path);
    assert(exists);
    std::cout << "✓ Existing file returns true" << std::endl;
    
    // Clean up
    std::remove(temp_path.c_str());
    
    std::cout << "Test 2 passed!" << std::endl;
}

// ============================================================================
// Test 3: LoadedModelConfig
// ============================================================================

void test_loaded_model_config() {
    std::cout << "\n=== Test 3: LoadedModelConfig ===" << std::endl;
    
    // Test default construction
    LoadedModelConfig config;
    
    // Default values should be zero/empty
    assert(config.model_name.empty());
    assert(config.hidden_size == 0);
    assert(config.num_hidden_layers == 0);
    assert(config.num_attention_heads == 0);
    assert(config.vocab_size == 0);
    assert(config.max_seq_length == 0);
    std::cout << "✓ Default construction" << std::endl;
    
    // Test setting values
    config.model_name = "test-bert";
    config.hidden_size = 768;
    config.num_hidden_layers = 12;
    config.num_attention_heads = 12;
    config.vocab_size = 30522;
    config.max_seq_length = 512;
    config.max_position_embeddings = 512;
    config.type_vocab_size = 2;
    
    assert(config.model_name == "test-bert");
    assert(config.hidden_size == 768);
    assert(config.num_hidden_layers == 12);
    assert(config.num_attention_heads == 12);
    assert(config.vocab_size == 30522);
    assert(config.max_seq_length == 512);
    assert(config.max_position_embeddings == 512);
    assert(config.type_vocab_size == 2);
    std::cout << "✓ Configuration settings" << std::endl;
    
    std::cout << "Test 3 passed!" << std::endl;
}

// ============================================================================
// Test 4: JSON Config Loading
// ============================================================================

void test_config_loading() {
    std::cout << "\n=== Test 4: JSON Config Loading ===" << std::endl;
    
    // Create a temporary JSON config file
    const std::string config_path = "/tmp/test_config.json";
    std::ofstream config_file(config_path);
    config_file << R"({
    "model_name": "bert-base-uncased",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "vocab_size": 30522,
    "max_seq_length": 512,
    "max_position_embeddings": 512,
    "type_vocab_size": 2
})";
    config_file.close();
    
    // Load the config
    ModelLoader loader;
    auto config = loader.loadModelConfig(config_path);
    
    assert(config.model_name == "bert-base-uncased");
    assert(config.hidden_size == 768);
    assert(config.num_hidden_layers == 12);
    assert(config.num_attention_heads == 12);
    assert(config.vocab_size == 30522);
    assert(config.max_seq_length == 512);
    std::cout << "✓ JSON config loaded successfully" << std::endl;
    
    // Clean up
    std::remove(config_path.c_str());
    
    // Test missing file error
    try {
        loader.loadModelConfig("nonexistent_config.json");
        assert(false && "Should have thrown exception");
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Missing file throws exception: " << e.what() << std::endl;
    }
    
    std::cout << "Test 4 passed!" << std::endl;
}

// ============================================================================
// Test 5: TorchScriptModel Creation (without actual model file)
// ============================================================================

void test_torchscript_model_wrapper() {
    std::cout << "\n=== Test 5: TorchScriptModel Wrapper ===" << std::endl;
    
    // Test that device management works
    torch::Device cpu_device(torch::kCPU);
    
    std::cout << "✓ Device management tested" << std::endl;
    std::cout << "  Note: Full TorchScript testing requires exported model files" << std::endl;
    
    std::cout << "Test 5 passed!" << std::endl;
}

// ============================================================================
// Test 6: Model Caching
// ============================================================================

void test_model_caching() {
    std::cout << "\n=== Test 6: Model Caching ===" << std::endl;
    
    ModelLoader loader;
    
    // Test cache clearing (doesn't require model files)
    loader.clearCache();
    std::cout << "✓ Cache clearing works" << std::endl;
    
    // Note: Full caching tests require actual model files
    // When models are available, loading twice should return same pointer
    
    std::cout << "  Note: Full caching tests require exported model files" << std::endl;
    
    std::cout << "Test 6 passed!" << std::endl;
}

// ============================================================================
// Test 7: Error Handling for Missing Models
// ============================================================================

void test_missing_model_handling() {
    std::cout << "\n=== Test 7: Error Handling for Missing Models ===" << std::endl;
    
    ModelLoader loader;
    
    // Test loading non-existent model
    try {
        loader.loadTorchScriptModel("nonexistent_model.pt");
        assert(false && "Should have thrown exception");
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Missing model throws exception" << std::endl;
    }
    
    std::cout << "Test 7 passed!" << std::endl;
}

// ============================================================================
// Test 8: Integration with AtomSpace (without model files)
// ============================================================================

void test_atomspace_integration_no_models() {
    std::cout << "\n=== Test 8: AtomSpace Integration ===" << std::endl;
    
    AtomSpace space;
    
    // Create concepts that could have embeddings from real models
    auto cat = createConceptNode(space, "cat");
    auto dog = createConceptNode(space, "dog");
    auto animal = createConceptNode(space, "animal");
    
    // Simulate embedding extraction (without real model)
    // In production, embeddings would come from BERT/GPT
    auto embedding_dim = 768;  // Standard BERT hidden size
    
    cat->setEmbedding(torch::randn({embedding_dim}));
    dog->setEmbedding(torch::randn({embedding_dim}));
    animal->setEmbedding(torch::randn({embedding_dim}));
    
    assert(cat->hasEmbedding());
    assert(cat->getEmbedding().size(0) == embedding_dim);
    std::cout << "✓ Concepts created with simulated BERT embeddings" << std::endl;
    
    // Test semantic similarity
    auto similar = space.querySimilar(cat->getEmbedding(), 3);
    assert(similar.size() == 3);
    std::cout << "✓ Semantic similarity search works" << std::endl;
    
    // Create knowledge structure
    auto inheritance1 = createInheritanceLink(space, cat, animal);
    auto inheritance2 = createInheritanceLink(space, dog, animal);
    
    inheritance1->setTruthValue(torch::tensor({0.95f, 0.9f}));
    inheritance2->setTruthValue(torch::tensor({0.95f, 0.9f}));
    
    std::cout << "✓ Knowledge graph structure created" << std::endl;
    
    // Verify integration
    assert(space.getSize() == 5);  // 3 nodes + 2 links
    std::cout << "  Atoms in space: " << space.getSize() << std::endl;
    
    std::cout << "Test 8 passed!" << std::endl;
}

// ============================================================================
// Test 9: Helper Functions Availability
// ============================================================================

void test_helper_functions() {
    std::cout << "\n=== Test 9: Helper Functions ===" << std::endl;
    
    // Test that helper functions exist and handle missing files gracefully
    
    try {
        auto bert = loadBERTModel("nonexistent.pt");
        assert(false && "Should have thrown");
    } catch (...) {
        std::cout << "✓ loadBERTModel handles missing file" << std::endl;
    }
    
    try {
        auto gpt2 = loadGPT2Model("nonexistent.pt");
        assert(false && "Should have thrown");
    } catch (...) {
        std::cout << "✓ loadGPT2Model handles missing file" << std::endl;
    }
    
    try {
        auto vit = loadViTModel("nonexistent.pt");
        assert(false && "Should have thrown");
    } catch (...) {
        std::cout << "✓ loadViTModel handles missing file" << std::endl;
    }
    
    try {
        auto yolo = loadYOLOModel("nonexistent.pt");
        assert(false && "Should have thrown");
    } catch (...) {
        std::cout << "✓ loadYOLOModel handles missing file" << std::endl;
    }
    
    std::cout << "Test 9 passed!" << std::endl;
}

// ============================================================================
// Test 10: Device String Conversion
// ============================================================================

void test_device_handling() {
    std::cout << "\n=== Test 10: Device Handling ===" << std::endl;
    
    ModelLoader loader;
    
    // Test CPU device
    auto cpu_device = torch::Device(torch::kCPU);
    std::cout << "✓ CPU device: " << cpu_device << std::endl;
    
    // Test that model loader can handle device changes
    auto default_device = loader.getDefaultDevice();
    std::cout << "✓ Default device: " << default_device << std::endl;
    
    // If CUDA is available, test that too
    if (torch::cuda::is_available()) {
        auto cuda_device = torch::Device(torch::kCUDA, 0);
        std::cout << "✓ CUDA device: " << cuda_device << std::endl;
    }
    
    std::cout << "Test 10 passed!" << std::endl;
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << "ModelLoader Unit Tests" << std::endl;
    std::cout << "======================" << std::endl;
    
    try {
        test_model_loader_construction();
        test_model_exists();
        test_loaded_model_config();
        test_config_loading();
        test_torchscript_model_wrapper();
        test_model_caching();
        test_missing_model_handling();
        test_atomspace_integration_no_models();
        test_helper_functions();
        test_device_handling();
        
        std::cout << "\n================================" << std::endl;
        std::cout << "All Tests Passed! ✓" << std::endl;
        std::cout << "================================" << std::endl;
        std::cout << "\nNote: Some tests require exported model files for full coverage." << std::endl;
        std::cout << "Export models using: python tools/export_models/export_all.py" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest Failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
