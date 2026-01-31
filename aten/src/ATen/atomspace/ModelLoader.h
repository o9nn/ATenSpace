/**
 * ModelLoader - TorchScript Model Loading Utilities
 * 
 * This module provides utilities for loading pre-trained models exported
 * to TorchScript format from Python/HuggingFace. It integrates with the
 * ATenNN framework to provide seamless loading of BERT, GPT-2, ViT, and
 * YOLO models.
 * 
 * Features:
 * - Load TorchScript models (.pt files)
 * - Model caching and path management
 * - Automatic device placement (CPU/GPU)
 * - Configuration loading from JSON
 * - Integration with ModelRegistry
 * 
 * Usage:
 *   ModelLoader loader;
 *   auto model = loader.loadTorchScriptModel("models/bert_base.pt");
 *   auto config = loader.loadModelConfig("models/bert_base_config.json");
 */

#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <stdexcept>

// JSON parsing library (simple header-only)
// For production, consider using nlohmann/json or rapidjson
#include <sstream>

namespace at {
namespace atomspace {
namespace nn {

/**
 * Model configuration loaded from JSON.
 * Stores hyperparameters and metadata about a model.
 */
struct LoadedModelConfig {
    std::string model_name;
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int vocab_size;
    int max_seq_length;
    
    // Optional fields
    int max_position_embeddings = 0;
    int type_vocab_size = 0;
    
    LoadedModelConfig() = default;
};

/**
 * Wrapper around a loaded TorchScript model.
 * Provides convenient inference interface.
 */
class TorchScriptModel {
public:
    TorchScriptModel(torch::jit::script::Module module, 
                     const torch::Device& device)
        : module_(std::move(module)), device_(device) {
        module_.to(device_);
        module_.eval();
    }
    
    /**
     * Run forward pass on the model.
     */
    torch::jit::IValue forward(const std::vector<torch::jit::IValue>& inputs) {
        torch::NoGradGuard no_grad;
        return module_.forward(inputs);
    }
    
    /**
     * Get the underlying TorchScript module.
     */
    torch::jit::script::Module& getModule() {
        return module_;
    }
    
    /**
     * Move model to a different device.
     */
    void to(const torch::Device& device) {
        device_ = device;
        module_.to(device);
    }
    
    /**
     * Get current device.
     */
    torch::Device getDevice() const {
        return device_;
    }
    
private:
    torch::jit::script::Module module_;
    torch::Device device_;
};

/**
 * ModelLoader - Loads and manages TorchScript models.
 * 
 * This class handles loading pre-trained models from disk,
 * caching them, and managing their lifecycle.
 */
class ModelLoader {
public:
    ModelLoader() : default_device_(torch::kCPU) {
        // Check if CUDA is available
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available, setting default device to GPU" << std::endl;
            default_device_ = torch::Device(torch::kCUDA, 0);
        }
    }
    
    /**
     * Load a TorchScript model from disk.
     * 
     * @param model_path Path to the .pt file
     * @param device Device to load model on (CPU/GPU)
     * @param use_cache Whether to cache the loaded model
     * @return Shared pointer to the loaded model
     */
    std::shared_ptr<TorchScriptModel> loadTorchScriptModel(
        const std::string& model_path,
        const torch::Device& device = torch::kCPU,
        bool use_cache = true
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check cache
        auto cache_key = model_path + "_" + deviceToString(device);
        if (use_cache && model_cache_.find(cache_key) != model_cache_.end()) {
            std::cout << "Loading model from cache: " << model_path << std::endl;
            return model_cache_[cache_key];
        }
        
        std::cout << "Loading TorchScript model: " << model_path << std::endl;
        
        try {
            // Load the model
            torch::jit::script::Module module = torch::jit::load(model_path);
            
            // Create wrapper
            auto model = std::make_shared<TorchScriptModel>(std::move(module), device);
            
            // Cache if requested
            if (use_cache) {
                model_cache_[cache_key] = model;
            }
            
            std::cout << "Model loaded successfully" << std::endl;
            return model;
            
        } catch (const c10::Error& e) {
            throw std::runtime_error("Failed to load model from " + model_path + 
                                   ": " + e.what());
        }
    }
    
    /**
     * Load model configuration from JSON file.
     * 
     * @param config_path Path to the _config.json file
     * @return Model configuration struct
     */
    LoadedModelConfig loadModelConfig(const std::string& config_path) {
        std::cout << "Loading model config: " << config_path << std::endl;
        
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + config_path);
        }
        
        // Read file content
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        
        // Parse JSON (simple parsing for demonstration)
        // In production, use a proper JSON library
        LoadedModelConfig config;
        config = parseSimpleJson(content);
        
        std::cout << "Config loaded:" << std::endl;
        std::cout << "  - Model: " << config.model_name << std::endl;
        std::cout << "  - Hidden size: " << config.hidden_size << std::endl;
        std::cout << "  - Layers: " << config.num_hidden_layers << std::endl;
        std::cout << "  - Vocab size: " << config.vocab_size << std::endl;
        
        return config;
    }
    
    /**
     * Clear the model cache.
     */
    void clearCache() {
        std::lock_guard<std::mutex> lock(mutex_);
        model_cache_.clear();
        std::cout << "Model cache cleared" << std::endl;
    }
    
    /**
     * Check if a model file exists.
     */
    static bool modelExists(const std::string& model_path) {
        std::ifstream file(model_path);
        return file.good();
    }
    
    /**
     * Get default device (GPU if available, else CPU).
     */
    torch::Device getDefaultDevice() const {
        return default_device_;
    }
    
private:
    torch::Device default_device_;
    std::unordered_map<std::string, std::shared_ptr<TorchScriptModel>> model_cache_;
    std::mutex mutex_;
    
    /**
     * Convert device to string for cache key.
     */
    std::string deviceToString(const torch::Device& device) const {
        if (device.is_cuda()) {
            return "cuda:" + std::to_string(device.index());
        }
        return "cpu";
    }
    
    /**
     * Simple JSON parser for model configs.
     * In production, use nlohmann/json or similar.
     */
    LoadedModelConfig parseSimpleJson(const std::string& json) {
        LoadedModelConfig config;
        
        // Extract fields using simple string parsing
        config.model_name = extractStringField(json, "model_name");
        config.hidden_size = extractIntField(json, "hidden_size");
        config.num_hidden_layers = extractIntField(json, "num_hidden_layers");
        config.num_attention_heads = extractIntField(json, "num_attention_heads");
        config.vocab_size = extractIntField(json, "vocab_size");
        config.max_seq_length = extractIntField(json, "max_seq_length");
        
        // Optional fields
        try {
            config.max_position_embeddings = extractIntField(json, "max_position_embeddings");
        } catch (...) {}
        
        try {
            config.type_vocab_size = extractIntField(json, "type_vocab_size");
        } catch (...) {}
        
        return config;
    }
    
    std::string extractStringField(const std::string& json, const std::string& field) {
        auto pos = json.find("\"" + field + "\"");
        if (pos == std::string::npos) {
            throw std::runtime_error("Field not found: " + field);
        }
        
        // Find the value (after the colon)
        auto colon_pos = json.find(":", pos);
        auto quote_start = json.find("\"", colon_pos);
        auto quote_end = json.find("\"", quote_start + 1);
        
        return json.substr(quote_start + 1, quote_end - quote_start - 1);
    }
    
    int extractIntField(const std::string& json, const std::string& field) {
        auto pos = json.find("\"" + field + "\"");
        if (pos == std::string::npos) {
            throw std::runtime_error("Field not found: " + field);
        }
        
        // Find the value (after the colon)
        auto colon_pos = json.find(":", pos);
        auto value_start = colon_pos + 1;
        
        // Skip whitespace
        while (value_start < json.length() && 
               (json[value_start] == ' ' || json[value_start] == '\t')) {
            value_start++;
        }
        
        // Extract number
        std::string num_str;
        while (value_start < json.length() && 
               (std::isdigit(json[value_start]) || json[value_start] == '-')) {
            num_str += json[value_start++];
        }
        
        return std::stoi(num_str);
    }
};

/**
 * Helper function to load BERT model from TorchScript.
 */
inline std::shared_ptr<TorchScriptModel> loadBERTModel(
    const std::string& model_path = "models/bert_base.pt"
) {
    ModelLoader loader;
    return loader.loadTorchScriptModel(model_path, loader.getDefaultDevice());
}

/**
 * Helper function to load GPT-2 model from TorchScript.
 */
inline std::shared_ptr<TorchScriptModel> loadGPT2Model(
    const std::string& model_path = "models/gpt2.pt"
) {
    ModelLoader loader;
    return loader.loadTorchScriptModel(model_path, loader.getDefaultDevice());
}

/**
 * Helper function to load ViT model from TorchScript.
 */
inline std::shared_ptr<TorchScriptModel> loadViTModel(
    const std::string& model_path = "models/vit_base.pt"
) {
    ModelLoader loader;
    return loader.loadTorchScriptModel(model_path, loader.getDefaultDevice());
}

/**
 * Helper function to load YOLO model from TorchScript.
 */
inline std::shared_ptr<TorchScriptModel> loadYOLOModel(
    const std::string& model_path = "models/yolov5.pt"
) {
    ModelLoader loader;
    return loader.loadTorchScriptModel(model_path, loader.getDefaultDevice());
}

} // namespace nn
} // namespace atomspace
} // namespace at
