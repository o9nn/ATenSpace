/**
 * Example: Loading Real Pre-trained Models with TorchScript
 * 
 * This example demonstrates how to load and use real pre-trained models
 * (BERT, GPT-2, ViT, YOLO) that have been exported from HuggingFace using
 * the Python export scripts in tools/export_models/.
 * 
 * Prerequisites:
 * 1. Export models using Python scripts:
 *    python tools/export_models/export_bert.py --output models/bert_base.pt
 *    python tools/export_models/export_gpt2.py --output models/gpt2.pt
 * 
 * 2. Build this example:
 *    cd aten/build
 *    cmake ..
 *    make atomspace_example_model_loader
 * 
 * 3. Run:
 *    ./atomspace_example_model_loader
 */

#include <ATen/atomspace/ATenSpace.h>
#include <ATen/atomspace/ModelLoader.h>
#include <iostream>
#include <iomanip>

using namespace at::atomspace;
using namespace at::atomspace::nn;

/**
 * Example 1: Load BERT model and run inference
 */
void example_load_bert() {
    std::cout << "\n=== Example 1: Load BERT Model ===" << std::endl;
    
    const std::string model_path = "models/bert_base.pt";
    const std::string config_path = "models/bert_base_config.json";
    
    // Check if model exists
    if (!ModelLoader::modelExists(model_path)) {
        std::cout << "Model not found: " << model_path << std::endl;
        std::cout << "Please export the model first:" << std::endl;
        std::cout << "  python tools/export_models/export_bert.py --output " << model_path << std::endl;
        return;
    }
    
    try {
        // Load model
        ModelLoader loader;
        auto model = loader.loadTorchScriptModel(model_path);
        
        // Load config
        auto config = loader.loadModelConfig(config_path);
        
        // Prepare input (example: [CLS] what is bert [SEP])
        std::vector<int64_t> input_ids = {
            101,  // [CLS]
            2054, // what
            2003, // is
            14324, // bert
            102   // [SEP]
        };
        
        // Pad to max_seq_length
        while (input_ids.size() < (size_t)config.max_seq_length) {
            input_ids.push_back(0);  // PAD token
        }
        
        // Create tensors
        auto input_tensor = torch::tensor(input_ids).unsqueeze(0);
        auto attention_mask = torch::ones({1, (int64_t)input_ids.size()});
        
        // Move to device
        input_tensor = input_tensor.to(model->getDevice());
        attention_mask = attention_mask.to(model->getDevice());
        
        std::cout << "\nRunning BERT inference..." << std::endl;
        std::cout << "Input shape: " << input_tensor.sizes() << std::endl;
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        inputs.push_back(attention_mask);
        
        auto output = model->forward(inputs);
        
        // Extract hidden states (assuming BaseModelOutputWithPooling)
        auto output_tuple = output.toTuple();
        auto hidden_states = output_tuple->elements()[0].toTensor();
        
        std::cout << "Output shape: " << hidden_states.sizes() << std::endl;
        std::cout << "Hidden size matches config: " 
                  << (hidden_states.size(2) == config.hidden_size ? "✓" : "✗") << std::endl;
        
        // Extract [CLS] token embedding
        auto cls_embedding = hidden_states.index({0, 0});
        std::cout << "CLS embedding size: " << cls_embedding.size(0) << std::endl;
        std::cout << "First 5 values: ";
        for (int i = 0; i < 5; i++) {
            std::cout << std::fixed << std::setprecision(4) 
                     << cls_embedding[i].item<float>() << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

/**
 * Example 2: Load GPT-2 model and generate text
 */
void example_load_gpt2() {
    std::cout << "\n=== Example 2: Load GPT-2 Model ===" << std::endl;
    
    const std::string model_path = "models/gpt2.pt";
    const std::string config_path = "models/gpt2_config.json";
    
    // Check if model exists
    if (!ModelLoader::modelExists(model_path)) {
        std::cout << "Model not found: " << model_path << std::endl;
        std::cout << "Please export the model first:" << std::endl;
        std::cout << "  python tools/export_models/export_gpt2.py --output " << model_path << std::endl;
        return;
    }
    
    try {
        // Load model
        ModelLoader loader;
        auto model = loader.loadTorchScriptModel(model_path);
        
        // Load config
        auto config = loader.loadModelConfig(config_path);
        
        // Prepare input (example: "The future of AI is")
        std::vector<int64_t> input_ids = {
            464,   // The
            2003,  // future
            286,   // of
            9552,  // AI
            318    // is
        };
        
        // Create tensor
        auto input_tensor = torch::tensor(input_ids).unsqueeze(0);
        input_tensor = input_tensor.to(model->getDevice());
        
        std::cout << "\nRunning GPT-2 inference..." << std::endl;
        std::cout << "Input shape: " << input_tensor.sizes() << std::endl;
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        auto output = model->forward(inputs);
        
        // Extract logits (assuming CausalLMOutputWithCrossAttentions)
        torch::Tensor logits;
        if (output.isTuple()) {
            logits = output.toTuple()->elements()[0].toTensor();
        } else {
            logits = output.toTensor();
        }
        
        std::cout << "Logits shape: " << logits.sizes() << std::endl;
        std::cout << "Vocab size matches config: " 
                  << (logits.size(2) == config.vocab_size ? "✓" : "✗") << std::endl;
        
        // Get next token prediction
        auto last_logits = logits.index({0, -1});
        auto next_token = torch::argmax(last_logits).item<int64_t>();
        
        std::cout << "Next token ID: " << next_token << std::endl;
        std::cout << "Top 5 token predictions:" << std::endl;
        
        auto top_k = torch::topk(last_logits, 5);
        auto values = std::get<0>(top_k);
        auto indices = std::get<1>(top_k);
        
        for (int i = 0; i < 5; i++) {
            std::cout << "  Token " << indices[i].item<int64_t>() 
                     << ": " << std::fixed << std::setprecision(4)
                     << values[i].item<float>() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

/**
 * Example 3: Integration with AtomSpace
 */
void example_atomspace_integration() {
    std::cout << "\n=== Example 3: AtomSpace Integration ===" << std::endl;
    
    const std::string model_path = "models/bert_base.pt";
    const std::string config_path = "models/bert_base_config.json";
    
    if (!ModelLoader::modelExists(model_path)) {
        std::cout << "Model not found. Skipping integration example." << std::endl;
        return;
    }
    
    try {
        // Create AtomSpace
        AtomSpace space;
        
        // Load BERT model
        ModelLoader loader;
        auto model = loader.loadTorchScriptModel(model_path);
        auto config = loader.loadModelConfig(config_path);
        
        std::cout << "Creating nodes with BERT embeddings..." << std::endl;
        
        // Create concepts with BERT embeddings
        std::vector<std::string> concepts = {"cat", "dog", "animal", "machine", "robot"};
        
        for (const auto& concept : concepts) {
            // In practice, you'd tokenize and get real embeddings
            // For now, create a concept node with random embedding
            auto node = createConceptNode(
                space, 
                concept,
                torch::randn({config.hidden_size})
            );
            
            std::cout << "  Created: " << node->toString() << std::endl;
        }
        
        // Query similar concepts
        auto query_embedding = torch::randn({config.hidden_size});
        auto similar = space.querySimilar(query_embedding, 3);
        
        std::cout << "\nMost similar concepts:" << std::endl;
        for (const auto& [atom, similarity] : similar) {
            std::cout << "  " << atom->toString() 
                     << " (similarity: " << std::fixed << std::setprecision(4) 
                     << similarity << ")" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

/**
 * Example 4: Model caching
 */
void example_model_caching() {
    std::cout << "\n=== Example 4: Model Caching ===" << std::endl;
    
    const std::string model_path = "models/bert_base.pt";
    
    if (!ModelLoader::modelExists(model_path)) {
        std::cout << "Model not found. Skipping caching example." << std::endl;
        return;
    }
    
    try {
        ModelLoader loader;
        
        // First load (from disk)
        std::cout << "Loading model (first time - from disk)..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto model1 = loader.loadTorchScriptModel(model_path);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time: " << duration1.count() << "ms" << std::endl;
        
        // Second load (from cache)
        std::cout << "\nLoading model (second time - from cache)..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        auto model2 = loader.loadTorchScriptModel(model_path);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time: " << duration2.count() << "ms" << std::endl;
        
        std::cout << "\nSpeedup from caching: " 
                  << std::fixed << std::setprecision(2)
                  << (float)duration1.count() / duration2.count() << "x" << std::endl;
        
        // Clear cache
        loader.clearCache();
        std::cout << "\nCache cleared." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "ATenSpace Model Loader Examples" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Run examples
    example_load_bert();
    example_load_gpt2();
    example_atomspace_integration();
    example_model_caching();
    
    std::cout << "\n=== Examples Complete ===" << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "1. Export more models (ViT, YOLO)" << std::endl;
    std::cout << "2. Implement proper tokenization in C++" << std::endl;
    std::cout << "3. Add fine-tuning capabilities" << std::endl;
    std::cout << "4. Integrate with ECAN attention mechanisms" << std::endl;
    
    return 0;
}
