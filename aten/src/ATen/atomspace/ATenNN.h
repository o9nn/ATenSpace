#pragma once

/**
 * ATenNN - Neural Networks Framework for ATenSpace
 * 
 * This module provides neural network architectures and components that integrate
 * with the symbolic cognitive architecture. It enables neuro-symbolic AI by bridging
 * deep learning models with knowledge graph reasoning.
 * 
 * Key Features:
 * - Pre-trained model integration (BERT, GPT, ViT, YOLO)
 * - Embedding extraction to AtomSpace nodes
 * - Neural-symbolic bridging
 * - Model registry and caching
 * - Configuration management
 * - Performance monitoring
 * 
 * Architecture:
 * - NeuralModule: Base class for all neural components
 * - ModelRegistry: Centralized model management
 * - EmbeddingExtractor: Extract embeddings from models
 * - AttentionBridge: Map neural attention to ECAN
 * - ModelConfig: Configuration management
 * - PerformanceMonitor: Track metrics and performance
 */

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include <mutex>
#include <chrono>

#include "Atom.h"
#include "AtomSpace.h"
#include "AttentionBank.h"

namespace at {
namespace atomspace {
namespace nn {

using Tensor = at::Tensor;

// ============================================================================
// Neural Module Base Class
// ============================================================================

/**
 * Base class for all neural network modules in ATenNN.
 * Provides common interface for forward pass, parameter management,
 * and integration with AtomSpace.
 */
class NeuralModule {
public:
    virtual ~NeuralModule() = default;
    
    /**
     * Forward pass through the neural module.
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor forward(const Tensor& input) = 0;
    
    /**
     * Get module name/identifier.
     */
    virtual std::string getName() const = 0;
    
    /**
     * Get device where module is located (CPU/GPU).
     */
    virtual torch::Device getDevice() const = 0;
    
    /**
     * Move module to specified device.
     */
    virtual void to(const torch::Device& device) = 0;
    
    /**
     * Set module to training mode.
     */
    virtual void train() = 0;
    
    /**
     * Set module to evaluation mode.
     */
    virtual void eval() = 0;
};

// ============================================================================
// Model Configuration
// ============================================================================

/**
 * Configuration for neural models.
 * Manages hyperparameters, paths, and runtime settings.
 */
struct ModelConfig {
    std::string model_name;
    std::string model_path;
    std::string model_type;  // "bert", "gpt", "vit", "yolo"
    
    // Device configuration
    torch::Device device = torch::kCPU;
    
    // Model parameters
    int64_t hidden_size = 768;
    int64_t num_layers = 12;
    int64_t num_heads = 12;
    int64_t vocab_size = 30522;
    int64_t max_seq_length = 512;
    
    // Runtime settings
    bool use_cache = true;
    bool half_precision = false;
    int64_t batch_size = 32;
    
    ModelConfig() = default;
    
    ModelConfig(const std::string& name, const std::string& type)
        : model_name(name), model_type(type) {}
};

// ============================================================================
// Embedding Extractor
// ============================================================================

/**
 * Extract embeddings from neural models and attach to AtomSpace nodes.
 * Supports various embedding strategies: CLS token, mean pooling, etc.
 */
class EmbeddingExtractor {
public:
    enum class Strategy {
        CLS_TOKEN,      // Use [CLS] token embedding (BERT-style)
        MEAN_POOLING,   // Average all token embeddings
        MAX_POOLING,    // Max pool over token embeddings
        LAST_HIDDEN,    // Use last hidden state
        WEIGHTED_MEAN   // Weighted average (attention-based)
    };
    
    EmbeddingExtractor(Strategy strategy = Strategy::CLS_TOKEN)
        : strategy_(strategy) {}
    
    /**
     * Extract embedding from model output.
     * @param hidden_states Model hidden states [batch, seq_len, hidden_size]
     * @param attention_mask Optional attention mask [batch, seq_len]
     * @return Extracted embeddings [batch, hidden_size]
     */
    Tensor extract(const Tensor& hidden_states, 
                   const Tensor& attention_mask = Tensor()) {
        switch (strategy_) {
            case Strategy::CLS_TOKEN:
                return extractCLS(hidden_states);
            case Strategy::MEAN_POOLING:
                return extractMean(hidden_states, attention_mask);
            case Strategy::MAX_POOLING:
                return extractMax(hidden_states, attention_mask);
            case Strategy::LAST_HIDDEN:
                return extractLast(hidden_states);
            case Strategy::WEIGHTED_MEAN:
                return extractWeightedMean(hidden_states, attention_mask);
            default:
                return extractCLS(hidden_states);
        }
    }
    
    /**
     * Attach embedding to AtomSpace node.
     */
    void attachToNode(std::shared_ptr<Atom> node, const Tensor& embedding) {
        if (node) {
            node->setEmbedding(embedding);
        }
    }
    
private:
    Strategy strategy_;
    
    Tensor extractCLS(const Tensor& hidden_states) {
        // Extract first token (CLS) embedding
        return hidden_states.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
    }
    
    Tensor extractMean(const Tensor& hidden_states, const Tensor& attention_mask) {
        if (attention_mask.defined()) {
            // Masked mean pooling
            auto mask = attention_mask.unsqueeze(-1).expand_as(hidden_states);
            auto sum = (hidden_states * mask).sum(1);
            auto count = mask.sum(1).clamp_min(1e-9);
            return sum / count;
        } else {
            // Simple mean pooling
            return hidden_states.mean(1);
        }
    }
    
    Tensor extractMax(const Tensor& hidden_states, const Tensor& attention_mask) {
        if (attention_mask.defined()) {
            // Masked max pooling
            auto mask = attention_mask.unsqueeze(-1).expand_as(hidden_states);
            auto masked = hidden_states.masked_fill(mask == 0, -1e9);
            return std::get<0>(masked.max(1));
        } else {
            return std::get<0>(hidden_states.max(1));
        }
    }
    
    Tensor extractLast(const Tensor& hidden_states) {
        // Extract last token embedding
        int64_t seq_len = hidden_states.size(1);
        return hidden_states.index({torch::indexing::Slice(), seq_len - 1, torch::indexing::Slice()});
    }
    
    Tensor extractWeightedMean(const Tensor& hidden_states, const Tensor& attention_mask) {
        // Use attention weights for weighted averaging
        // For simplicity, using mean pooling here
        return extractMean(hidden_states, attention_mask);
    }
};

// ============================================================================
// Attention Bridge
// ============================================================================

/**
 * Bridge neural attention mechanisms to ECAN attention system.
 * Maps attention scores to STI/LTI values in AttentionBank.
 */
class AttentionBridge {
public:
    AttentionBridge(AttentionBank& attention_bank)
        : attention_bank_(attention_bank) {}
    
    /**
     * Map attention scores to STI values.
     * @param atoms Vector of atoms to update
     * @param attention_scores Attention scores [batch_size, num_atoms]
     * @param scale Scaling factor for attention to STI conversion
     */
    void mapAttentionToSTI(const std::vector<std::shared_ptr<Atom>>& atoms,
                           const Tensor& attention_scores,
                           float scale = 100.0f) {
        auto scores_cpu = attention_scores.cpu();
        auto scores_accessor = scores_cpu.accessor<float, 1>();
        
        for (size_t i = 0; i < atoms.size() && i < scores_cpu.size(0); ++i) {
            float attention = scores_accessor[i];
            float sti = attention * scale;
            
            // Get current attention value
            auto current_av = attention_bank_.getAttentionValue(atoms[i]);
            
            // Update STI while preserving LTI and VLTI
            AttentionValue new_av(sti, current_av.lti, current_av.vlti);
            attention_bank_.setAttentionValue(atoms[i], new_av);
        }
    }
    
    /**
     * Extract attention focus from neural model.
     * Identifies most attended atoms based on attention scores.
     */
    std::vector<std::shared_ptr<Atom>> extractFocus(
        const std::vector<std::shared_ptr<Atom>>& atoms,
        const Tensor& attention_scores,
        int top_k = 10) {
        
        auto scores_cpu = attention_scores.cpu();
        auto topk_result = torch::topk(scores_cpu, std::min(top_k, (int)atoms.size()));
        auto indices = std::get<1>(topk_result);
        auto indices_accessor = indices.accessor<int64_t, 1>();
        
        std::vector<std::shared_ptr<Atom>> focus;
        for (int i = 0; i < indices.size(0); ++i) {
            int64_t idx = indices_accessor[i];
            if (idx < atoms.size()) {
                focus.push_back(atoms[idx]);
            }
        }
        
        return focus;
    }
    
private:
    AttentionBank& attention_bank_;
};

// ============================================================================
// Performance Monitor
// ============================================================================

/**
 * Monitor neural model performance and resource usage.
 * Tracks inference time, memory usage, and throughput.
 */
class PerformanceMonitor {
public:
    struct Metrics {
        int64_t num_inferences = 0;
        double total_inference_time_ms = 0.0;
        double avg_inference_time_ms = 0.0;
        int64_t total_tokens_processed = 0;
        double throughput_tokens_per_sec = 0.0;
        size_t peak_memory_bytes = 0;
    };
    
    PerformanceMonitor() = default;
    
    /**
     * Start timing an inference operation.
     */
    void startInference() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * End timing and record metrics.
     * @param num_tokens Number of tokens processed
     */
    void endInference(int64_t num_tokens = 0) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        double time_ms = duration.count() / 1000.0;
        
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.num_inferences++;
        metrics_.total_inference_time_ms += time_ms;
        metrics_.avg_inference_time_ms = 
            metrics_.total_inference_time_ms / metrics_.num_inferences;
        
        if (num_tokens > 0) {
            metrics_.total_tokens_processed += num_tokens;
            double total_time_sec = metrics_.total_inference_time_ms / 1000.0;
            metrics_.throughput_tokens_per_sec = 
                metrics_.total_tokens_processed / total_time_sec;
        }
    }
    
    /**
     * Get current metrics.
     */
    Metrics getMetrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return metrics_;
    }
    
    /**
     * Reset all metrics.
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_ = Metrics();
    }
    
    /**
     * Print metrics summary.
     */
    void printSummary() const {
        auto m = getMetrics();
        std::cout << "\n=== Performance Metrics ===" << std::endl;
        std::cout << "Total inferences: " << m.num_inferences << std::endl;
        std::cout << "Avg inference time: " << m.avg_inference_time_ms << " ms" << std::endl;
        std::cout << "Total tokens: " << m.total_tokens_processed << std::endl;
        std::cout << "Throughput: " << m.throughput_tokens_per_sec << " tokens/sec" << std::endl;
        std::cout << "=========================" << std::endl;
    }
    
private:
    mutable std::mutex mutex_;
    Metrics metrics_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

// ============================================================================
// Model Registry
// ============================================================================

/**
 * Central registry for managing neural models.
 * Provides model loading, caching, and lifecycle management.
 */
class ModelRegistry {
public:
    using ModelFactory = std::function<std::shared_ptr<NeuralModule>(const ModelConfig&)>;
    
    static ModelRegistry& getInstance() {
        static ModelRegistry instance;
        return instance;
    }
    
    /**
     * Register a model factory.
     * @param model_type Model type identifier (e.g., "bert", "gpt")
     * @param factory Factory function to create model
     */
    void registerModel(const std::string& model_type, ModelFactory factory) {
        std::lock_guard<std::mutex> lock(mutex_);
        factories_[model_type] = factory;
    }
    
    /**
     * Load a model by configuration.
     * Uses caching if enabled in config.
     */
    std::shared_ptr<NeuralModule> loadModel(const ModelConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check cache if enabled
        if (config.use_cache) {
            auto cache_key = config.model_name + ":" + config.model_type;
            auto it = cache_.find(cache_key);
            if (it != cache_.end()) {
                return it->second;
            }
        }
        
        // Create new model
        auto factory_it = factories_.find(config.model_type);
        if (factory_it == factories_.end()) {
            throw std::runtime_error("Unknown model type: " + config.model_type);
        }
        
        auto model = factory_it->second(config);
        
        // Cache if enabled
        if (config.use_cache) {
            auto cache_key = config.model_name + ":" + config.model_type;
            cache_[cache_key] = model;
        }
        
        return model;
    }
    
    /**
     * Check if model type is registered.
     */
    bool hasModelType(const std::string& model_type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return factories_.find(model_type) != factories_.end();
    }
    
    /**
     * Clear model cache.
     */
    void clearCache() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }
    
private:
    ModelRegistry() = default;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ModelFactory> factories_;
    std::unordered_map<std::string, std::shared_ptr<NeuralModule>> cache_;
};

} // namespace nn
} // namespace atomspace
} // namespace at
