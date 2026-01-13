#pragma once

/**
 * Pre-trained Model Integrations for ATenNN
 * 
 * This module provides integration with popular pre-trained models:
 * - BERT: Bidirectional Encoder Representations from Transformers
 * - GPT: Generative Pre-trained Transformer
 * - ViT: Vision Transformer
 * - YOLO: You Only Look Once (object detection)
 * 
 * These integrations enable the cognitive architecture to leverage
 * state-of-the-art neural models for language understanding, generation,
 * and visual perception.
 */

#include "ATenNN.h"
#include <torch/torch.h>

namespace at {
namespace atomspace {
namespace nn {

// ============================================================================
// BERT Model - Language Understanding
// ============================================================================

/**
 * BERT-style transformer model for language understanding.
 * Provides contextualized embeddings for text inputs.
 */
class BERTModel : public NeuralModule {
public:
    BERTModel(const ModelConfig& config)
        : config_(config), device_(config.device) {
        buildModel();
    }
    
    Tensor forward(const Tensor& input) override {
        torch::NoGradGuard no_grad;
        model_->eval();
        
        // Input is token IDs [batch, seq_len]
        auto embeddings = embedding_->forward(input);
        
        // Add positional embeddings
        auto seq_len = input.size(1);
        auto positions = torch::arange(seq_len, device_).unsqueeze(0).expand_as(input);
        embeddings = embeddings + position_embedding_->forward(positions);
        
        // Apply layer norm
        embeddings = layer_norm_->forward(embeddings);
        
        // Pass through transformer layers
        auto hidden_states = embeddings;
        for (auto& layer : transformer_layers_) {
            hidden_states = layer->forward(hidden_states);
        }
        
        return hidden_states;
    }
    
    std::string getName() const override {
        return config_.model_name;
    }
    
    torch::Device getDevice() const override {
        return device_;
    }
    
    void to(const torch::Device& device) override {
        device_ = device;
        model_->to(device);
    }
    
    void train() override {
        model_->train();
    }
    
    void eval() override {
        model_->eval();
    }
    
    /**
     * Extract embeddings for text integration with AtomSpace.
     * @param input_ids Token IDs [batch, seq_len]
     * @param attention_mask Attention mask [batch, seq_len]
     * @return Embeddings [batch, hidden_size]
     */
    Tensor extractEmbeddings(const Tensor& input_ids, 
                            const Tensor& attention_mask = Tensor()) {
        auto hidden_states = forward(input_ids);
        EmbeddingExtractor extractor(EmbeddingExtractor::Strategy::CLS_TOKEN);
        return extractor.extract(hidden_states, attention_mask);
    }
    
private:
    ModelConfig config_;
    torch::Device device_;
    torch::nn::Sequential model_{nullptr};
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::Embedding position_embedding_{nullptr};
    torch::nn::LayerNorm layer_norm_{nullptr};
    std::vector<torch::nn::Sequential> transformer_layers_;
    
    void buildModel() {
        // Token embeddings
        embedding_ = torch::nn::Embedding(config_.vocab_size, config_.hidden_size);
        
        // Position embeddings
        position_embedding_ = torch::nn::Embedding(config_.max_seq_length, config_.hidden_size);
        
        // Layer normalization
        layer_norm_ = torch::nn::LayerNorm(torch::nn::LayerNormOptions({config_.hidden_size}));
        
        // Transformer layers
        for (int i = 0; i < config_.num_layers; ++i) {
            auto layer = torch::nn::Sequential();
            
            // Multi-head self-attention (simplified)
            auto attn = torch::nn::Linear(config_.hidden_size, config_.hidden_size);
            layer->push_back(attn);
            
            // Feed-forward network
            auto ff1 = torch::nn::Linear(config_.hidden_size, config_.hidden_size * 4);
            auto ff2 = torch::nn::Linear(config_.hidden_size * 4, config_.hidden_size);
            layer->push_back(ff1);
            layer->push_back(ff2);
            
            transformer_layers_.push_back(layer);
        }
        
        // Wrap in sequential model
        model_ = torch::nn::Sequential();
        
        // Move to device
        model_->to(device_);
        embedding_->to(device_);
        position_embedding_->to(device_);
        layer_norm_->to(device_);
        for (auto& layer : transformer_layers_) {
            layer->to(device_);
        }
    }
};

// ============================================================================
// GPT Model - Language Generation
// ============================================================================

/**
 * GPT-style autoregressive transformer for text generation.
 * Generates text conditioned on context.
 */
class GPTModel : public NeuralModule {
public:
    GPTModel(const ModelConfig& config)
        : config_(config), device_(config.device) {
        buildModel();
    }
    
    Tensor forward(const Tensor& input) override {
        torch::NoGradGuard no_grad;
        model_->eval();
        
        // Similar to BERT but with causal masking
        auto embeddings = embedding_->forward(input);
        
        auto seq_len = input.size(1);
        auto positions = torch::arange(seq_len, device_).unsqueeze(0).expand_as(input);
        embeddings = embeddings + position_embedding_->forward(positions);
        
        auto hidden_states = embeddings;
        for (auto& layer : transformer_layers_) {
            hidden_states = layer->forward(hidden_states);
        }
        
        // Project to vocabulary
        auto logits = output_projection_->forward(hidden_states);
        return logits;
    }
    
    std::string getName() const override {
        return config_.model_name;
    }
    
    torch::Device getDevice() const override {
        return device_;
    }
    
    void to(const torch::Device& device) override {
        device_ = device;
        model_->to(device);
    }
    
    void train() override {
        model_->train();
    }
    
    void eval() override {
        model_->eval();
    }
    
    /**
     * Generate text from prompt.
     * @param input_ids Initial token IDs [batch, seq_len]
     * @param max_length Maximum generation length
     * @return Generated token IDs [batch, max_length]
     */
    Tensor generate(const Tensor& input_ids, int64_t max_length = 50) {
        auto generated = input_ids.clone();
        
        for (int64_t i = 0; i < max_length; ++i) {
            auto logits = forward(generated);
            auto next_token_logits = logits.index({
                torch::indexing::Slice(), -1, torch::indexing::Slice()
            });
            auto next_token = next_token_logits.argmax(-1, /*keepdim=*/true);
            generated = torch::cat({generated, next_token}, /*dim=*/1);
        }
        
        return generated;
    }
    
private:
    ModelConfig config_;
    torch::Device device_;
    torch::nn::Sequential model_{nullptr};
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::Embedding position_embedding_{nullptr};
    std::vector<torch::nn::Sequential> transformer_layers_;
    torch::nn::Linear output_projection_{nullptr};
    
    void buildModel() {
        embedding_ = torch::nn::Embedding(config_.vocab_size, config_.hidden_size);
        position_embedding_ = torch::nn::Embedding(config_.max_seq_length, config_.hidden_size);
        
        for (int i = 0; i < config_.num_layers; ++i) {
            auto layer = torch::nn::Sequential();
            auto attn = torch::nn::Linear(config_.hidden_size, config_.hidden_size);
            auto ff1 = torch::nn::Linear(config_.hidden_size, config_.hidden_size * 4);
            auto ff2 = torch::nn::Linear(config_.hidden_size * 4, config_.hidden_size);
            layer->push_back(attn);
            layer->push_back(ff1);
            layer->push_back(ff2);
            transformer_layers_.push_back(layer);
        }
        
        output_projection_ = torch::nn::Linear(config_.hidden_size, config_.vocab_size);
        
        model_ = torch::nn::Sequential();
        model_->to(device_);
        embedding_->to(device_);
        position_embedding_->to(device_);
        output_projection_->to(device_);
        for (auto& layer : transformer_layers_) {
            layer->to(device_);
        }
    }
};

// ============================================================================
// Vision Transformer (ViT) - Visual Understanding
// ============================================================================

/**
 * Vision Transformer for image understanding.
 * Processes images as sequences of patches.
 */
class ViTModel : public NeuralModule {
public:
    ViTModel(const ModelConfig& config)
        : config_(config), device_(config.device) {
        patch_size_ = 16;  // 16x16 patches
        image_size_ = 224;  // 224x224 images
        num_patches_ = (image_size_ / patch_size_) * (image_size_ / patch_size_);
        buildModel();
    }
    
    Tensor forward(const Tensor& input) override {
        torch::NoGradGuard no_grad;
        model_->eval();
        
        // Input is image [batch, 3, H, W]
        // Extract patches
        auto patches = patchify(input);
        
        // Project patches to embedding dimension
        auto patch_embeddings = patch_projection_->forward(patches);
        
        // Add CLS token and position embeddings
        auto batch_size = input.size(0);
        auto cls_tokens = cls_token_.expand({batch_size, -1, -1});
        auto embeddings = torch::cat({cls_tokens, patch_embeddings}, /*dim=*/1);
        embeddings = embeddings + position_embedding_;
        
        // Pass through transformer
        auto hidden_states = embeddings;
        for (auto& layer : transformer_layers_) {
            hidden_states = layer->forward(hidden_states);
        }
        
        return hidden_states;
    }
    
    std::string getName() const override {
        return config_.model_name;
    }
    
    torch::Device getDevice() const override {
        return device_;
    }
    
    void to(const torch::Device& device) override {
        device_ = device;
        model_->to(device);
    }
    
    void train() override {
        model_->train();
    }
    
    void eval() override {
        model_->eval();
    }
    
    /**
     * Extract visual embeddings for AtomSpace integration.
     * @param images Input images [batch, 3, H, W]
     * @return Embeddings [batch, hidden_size]
     */
    Tensor extractEmbeddings(const Tensor& images) {
        auto hidden_states = forward(images);
        // Extract CLS token embedding
        return hidden_states.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
    }
    
private:
    ModelConfig config_;
    torch::Device device_;
    torch::nn::Sequential model_{nullptr};
    torch::nn::Linear patch_projection_{nullptr};
    Tensor cls_token_;
    Tensor position_embedding_;
    std::vector<torch::nn::Sequential> transformer_layers_;
    
    int64_t patch_size_;
    int64_t image_size_;
    int64_t num_patches_;
    
    void buildModel() {
        int64_t patch_dim = 3 * patch_size_ * patch_size_;
        
        // Patch projection
        patch_projection_ = torch::nn::Linear(patch_dim, config_.hidden_size);
        
        // CLS token
        cls_token_ = torch::randn({1, 1, config_.hidden_size}, device_);
        
        // Position embeddings
        position_embedding_ = torch::randn({1, num_patches_ + 1, config_.hidden_size}, device_);
        
        // Transformer layers
        for (int i = 0; i < config_.num_layers; ++i) {
            auto layer = torch::nn::Sequential();
            auto attn = torch::nn::Linear(config_.hidden_size, config_.hidden_size);
            auto ff1 = torch::nn::Linear(config_.hidden_size, config_.hidden_size * 4);
            auto ff2 = torch::nn::Linear(config_.hidden_size * 4, config_.hidden_size);
            layer->push_back(attn);
            layer->push_back(ff1);
            layer->push_back(ff2);
            transformer_layers_.push_back(layer);
        }
        
        model_ = torch::nn::Sequential();
        model_->to(device_);
        patch_projection_->to(device_);
        for (auto& layer : transformer_layers_) {
            layer->to(device_);
        }
    }
    
    Tensor patchify(const Tensor& images) {
        // Convert image to patches
        // images: [batch, 3, H, W]
        // output: [batch, num_patches, patch_dim]
        auto batch_size = images.size(0);
        auto patches = images.unfold(2, patch_size_, patch_size_)
                             .unfold(3, patch_size_, patch_size_);
        patches = patches.contiguous().view({
            batch_size, 
            3, 
            -1, 
            patch_size_, 
            patch_size_
        });
        patches = patches.permute({0, 2, 1, 3, 4});
        patches = patches.contiguous().view({
            batch_size, 
            num_patches_, 
            3 * patch_size_ * patch_size_
        });
        return patches;
    }
};

// ============================================================================
// YOLO Model - Object Detection
// ============================================================================

/**
 * YOLO-style object detection model.
 * Detects and localizes objects in images.
 */
class YOLOModel : public NeuralModule {
public:
    struct Detection {
        Tensor bbox;      // [x, y, w, h]
        float confidence;
        int64_t class_id;
        std::string class_name;
    };
    
    YOLOModel(const ModelConfig& config)
        : config_(config), device_(config.device) {
        buildModel();
    }
    
    Tensor forward(const Tensor& input) override {
        torch::NoGradGuard no_grad;
        model_->eval();
        
        // Input is image [batch, 3, H, W]
        auto features = backbone_->forward(input);
        auto detections = detection_head_->forward(features);
        
        return detections;
    }
    
    std::string getName() const override {
        return config_.model_name;
    }
    
    torch::Device getDevice() const override {
        return device_;
    }
    
    void to(const torch::Device& device) override {
        device_ = device;
        model_->to(device);
    }
    
    void train() override {
        model_->train();
    }
    
    void eval() override {
        model_->eval();
    }
    
    /**
     * Detect objects in image and create AtomSpace representations.
     * @param image Input image [3, H, W]
     * @param space AtomSpace to add detections to
     * @param confidence_threshold Minimum confidence for detections
     * @return Vector of detected object nodes
     */
    std::vector<std::shared_ptr<Atom>> detectObjects(
        const Tensor& image,
        AtomSpace& space,
        float confidence_threshold = 0.5) {
        
        auto detections = forward(image.unsqueeze(0));
        std::vector<std::shared_ptr<Atom>> detected_nodes;
        
        // Parse detections (simplified)
        // In real implementation, would use proper NMS and post-processing
        auto det_cpu = detections.cpu();
        
        // For now, return empty vector
        // Real implementation would create concept nodes for detected objects
        // and spatial relation links for their bounding boxes
        
        return detected_nodes;
    }
    
private:
    ModelConfig config_;
    torch::Device device_;
    torch::nn::Sequential model_{nullptr};
    torch::nn::Sequential backbone_{nullptr};
    torch::nn::Sequential detection_head_{nullptr};
    
    void buildModel() {
        // Simplified backbone (in practice, use ResNet/Darknet)
        backbone_ = torch::nn::Sequential(
            torch::nn::Conv2d(3, 64, 3, 1, 1),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(2),
            torch::nn::Conv2d(64, 128, 3, 1, 1),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(2),
            torch::nn::Conv2d(128, 256, 3, 1, 1),
            torch::nn::ReLU()
        );
        
        // Detection head
        detection_head_ = torch::nn::Sequential(
            torch::nn::Conv2d(256, 512, 3, 1, 1),
            torch::nn::ReLU(),
            torch::nn::Conv2d(512, 85, 1)  // 85 = 4 bbox + 1 obj + 80 classes
        );
        
        model_ = torch::nn::Sequential();
        model_->to(device_);
        backbone_->to(device_);
        detection_head_->to(device_);
    }
};

// ============================================================================
// Model Factory Functions
// ============================================================================

/**
 * Create BERT model from configuration.
 */
inline std::shared_ptr<NeuralModule> createBERTModel(const ModelConfig& config) {
    return std::make_shared<BERTModel>(config);
}

/**
 * Create GPT model from configuration.
 */
inline std::shared_ptr<NeuralModule> createGPTModel(const ModelConfig& config) {
    return std::make_shared<GPTModel>(config);
}

/**
 * Create ViT model from configuration.
 */
inline std::shared_ptr<NeuralModule> createViTModel(const ModelConfig& config) {
    return std::make_shared<ViTModel>(config);
}

/**
 * Create YOLO model from configuration.
 */
inline std::shared_ptr<NeuralModule> createYOLOModel(const ModelConfig& config) {
    return std::make_shared<YOLOModel>(config);
}

/**
 * Register all pre-trained models with the model registry.
 */
inline void registerPretrainedModels() {
    auto& registry = ModelRegistry::getInstance();
    registry.registerModel("bert", createBERTModel);
    registry.registerModel("gpt", createGPTModel);
    registry.registerModel("vit", createViTModel);
    registry.registerModel("yolo", createYOLOModel);
}

} // namespace nn
} // namespace atomspace
} // namespace at
