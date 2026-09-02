#pragma once

#include "Atom.h"
#include "AtomSpace.h"
#include "TruthValue.h"
#include "PatternMatcher.h"
#include <ATen/ATen.h>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

namespace at {
namespace atomspace {

/**
 * TensorLogicEngine - GPU-accelerated batch logical inference
 * 
 * This engine implements tensor-based logical operations, enabling
 * batch processing of truth values and parallel inference across
 * multiple atoms simultaneously. It leverages GPU acceleration for
 * large-scale reasoning tasks.
 * 
 * Key features:
 * - Batch truth value computation
 * - Parallel pattern matching
 * - GPU-accelerated logical operations
 * - Vectorized inference rules
 * - Efficient memory layout for tensor operations
 */
class TensorLogicEngine {
public:
    /**
     * LogicalOperation - Types of logical operations supported
     */
    enum class LogicalOperation {
        AND,
        OR,
        NOT,
        IMPLIES,
        EQUIVALENT,
        XOR
    };
    
    /**
     * InferenceMode - Execution mode for inference
     */
    enum class InferenceMode {
        CPU,      // CPU-only execution
        GPU,      // GPU-accelerated execution
        AUTO      // Automatic selection based on data size
    };
    
    /**
     * Construct a TensorLogicEngine
     * 
     * @param mode Inference mode (CPU, GPU, or AUTO)
     * @param batchSize Optimal batch size for GPU operations
     */
    explicit TensorLogicEngine(
        InferenceMode mode = InferenceMode::AUTO,
        size_t batchSize = 256)
        : mode_(mode), batchSize_(batchSize) {}
    
    /**
     * Batch apply logical operation to multiple atom pairs
     * 
     * Computes logical operations on truth values in parallel using tensors.
     * 
     * @param atoms1 First set of atoms
     * @param atoms2 Second set of atoms
     * @param operation Logical operation to apply
     * @return Tensor of result truth values [n, 2] where n = atoms.size()
     */
    Tensor batchLogicalOperation(
        const std::vector<Atom::Handle>& atoms1,
        const std::vector<Atom::Handle>& atoms2,
        LogicalOperation operation) const {
        
        if (atoms1.size() != atoms2.size()) {
            throw std::runtime_error("Atom vectors must have same size");
        }
        
        if (atoms1.empty()) {
            return torch::empty({0, 2});
        }
        
        // Extract truth values into tensor
        auto tv1 = extractTruthValues(atoms1);
        auto tv2 = extractTruthValues(atoms2);
        
        // Apply operation
        return applyLogicalOperation(tv1, tv2, operation);
    }
    
    /**
     * Batch apply unary logical operation
     * 
     * @param atoms Set of atoms
     * @param operation Logical operation (e.g., NOT)
     * @return Tensor of result truth values [n, 2]
     */
    Tensor batchUnaryOperation(
        const std::vector<Atom::Handle>& atoms,
        LogicalOperation operation) const {
        
        if (atoms.empty()) {
            return torch::empty({0, 2});
        }
        
        auto tv = extractTruthValues(atoms);
        
        if (operation == LogicalOperation::NOT) {
            return applyNegation(tv);
        }
        
        throw std::runtime_error("Unsupported unary operation");
    }
    
    /**
     * Batch deduction inference: (A→B, B→C) ⊢ A→C
     *
     * Given two sets of implication links, compute all resulting implications
     * in parallel using tensor operations.
     *
     * @param premises1 First set of implications (A→B)
     * @param premises2 Second set of implications (B→C)
     * @return Tensor of deduced truth values [n, 2]
     */
    Tensor batchDeduction(
        const std::vector<Atom::Handle>& premises1,
        const std::vector<Atom::Handle>& premises2) const {

        if (premises1.size() != premises2.size()) {
            throw std::runtime_error("Premise vectors must have same size");
        }

        if (premises1.empty()) {
            return torch::empty({0, 2});
        }

        auto tv1 = extractTruthValues(premises1);
        auto tv2 = extractTruthValues(premises2);

        return batchDeductionTV(tv1, tv2);
    }

    /**
     * Batch deduction tensor contraction over raw [n, 2] truth-value lanes.
     *
     * This is the tensor-contraction form of the PLN deduction rule
     * (Iteration 2, FR-2.1): a single element-wise contraction over
     * strength/confidence lanes, numerically equivalent to the scalar
     * TruthValue::deduction reference within 1e-4.
     *
     * @param tv1 Truth-value tensor for the A→B premises, shape [n, 2]
     * @param tv2 Truth-value tensor for the B→C premises, shape [n, 2]
     * @return Tensor of deduced truth values [n, 2]
     */
    Tensor batchDeductionTV(const Tensor& tv1, const Tensor& tv2) const {
        checkTVPair(tv1, tv2);
        return dispatchTV(tv1, tv2,
            [this](const Tensor& a, const Tensor& b) {
                return applyDeduction(a, b);
            });
    }

    /**
     * Batch induction inference: (A→B, A→C) ⊢ B→C
     *
     * When B and C follow from the same antecedent A, induces a link B→C.
     * Computed as a single batched tensor contraction (Iteration 2, FR-2.2).
     *
     * @param premises1 First set of implications (A→B)
     * @param premises2 Second set of implications (A→C)
     * @return Tensor of induced truth values [n, 2]
     */
    Tensor batchInduction(
        const std::vector<Atom::Handle>& premises1,
        const std::vector<Atom::Handle>& premises2) const {

        if (premises1.size() != premises2.size()) {
            throw std::runtime_error("Premise vectors must have same size");
        }

        if (premises1.empty()) {
            return torch::empty({0, 2});
        }

        return batchInductionTV(extractTruthValues(premises1),
                                extractTruthValues(premises2));
    }

    /**
     * Batch induction tensor contraction over raw [n, 2] truth-value lanes.
     *
     * @param tv1 Truth-value tensor for the A→B premises, shape [n, 2]
     * @param tv2 Truth-value tensor for the A→C premises, shape [n, 2]
     * @return Tensor of induced truth values [n, 2]
     */
    Tensor batchInductionTV(const Tensor& tv1, const Tensor& tv2) const {
        checkTVPair(tv1, tv2);
        return dispatchTV(tv1, tv2,
            [this](const Tensor& a, const Tensor& b) {
                return applyInduction(a, b);
            });
    }

    /**
     * Batch abduction inference: (A→C, B→C) ⊢ A→B
     *
     * When A and B both imply the same consequent C, abduces a link A→B
     * ("reasoning to best explanation", discounted for uncertainty).
     * Computed as a single batched tensor contraction (Iteration 2, FR-2.2).
     *
     * @param premises1 First set of implications (A→C)
     * @param premises2 Second set of implications (B→C)
     * @return Tensor of abduced truth values [n, 2]
     */
    Tensor batchAbduction(
        const std::vector<Atom::Handle>& premises1,
        const std::vector<Atom::Handle>& premises2) const {

        if (premises1.size() != premises2.size()) {
            throw std::runtime_error("Premise vectors must have same size");
        }

        if (premises1.empty()) {
            return torch::empty({0, 2});
        }

        return batchAbductionTV(extractTruthValues(premises1),
                                extractTruthValues(premises2));
    }

    /**
     * Batch abduction tensor contraction over raw [n, 2] truth-value lanes.
     *
     * @param tv1 Truth-value tensor for the A→C premises, shape [n, 2]
     * @param tv2 Truth-value tensor for the B→C premises, shape [n, 2]
     * @return Tensor of abduced truth values [n, 2]
     */
    Tensor batchAbductionTV(const Tensor& tv1, const Tensor& tv2) const {
        checkTVPair(tv1, tv2);
        return dispatchTV(tv1, tv2,
            [this](const Tensor& a, const Tensor& b) {
                return applyAbduction(a, b);
            });
    }

    /**
     * Batch revision: merge two independent evidence sets lane-wise
     *
     * Combines pairs of truth-value estimates of the same propositions
     * using the PLN revision formula, applied across all tensor lanes in
     * parallel (Iteration 2, FR-2.3).
     *
     * @param atoms1 First set of evidence atoms
     * @param atoms2 Second set of evidence atoms
     * @return Tensor of revised truth values [n, 2]
     */
    Tensor batchRevision(
        const std::vector<Atom::Handle>& atoms1,
        const std::vector<Atom::Handle>& atoms2) const {

        if (atoms1.size() != atoms2.size()) {
            throw std::runtime_error("Atom vectors must have same size");
        }

        if (atoms1.empty()) {
            return torch::empty({0, 2});
        }

        return batchRevisionTV(extractTruthValues(atoms1),
                               extractTruthValues(atoms2));
    }

    /**
     * Batch revision tensor contraction over raw [n, 2] truth-value lanes.
     *
     * @param tv1 First evidence tensor, shape [n, 2]
     * @param tv2 Second evidence tensor, shape [n, 2]
     * @return Tensor of revised truth values [n, 2]
     */
    Tensor batchRevisionTV(const Tensor& tv1, const Tensor& tv2) const {
        checkTVPair(tv1, tv2);
        return dispatchTV(tv1, tv2,
            [this](const Tensor& a, const Tensor& b) {
                return applyRevision(a, b);
            });
    }

    /**
     * Batch similarity computation using embeddings
     * 
     * Compute semantic similarity between atoms using their embeddings.
     * This is GPU-accelerated and handles large batches efficiently.
     * 
     * @param atoms1 First set of atoms
     * @param atoms2 Second set of atoms
     * @return Tensor of similarity scores [n]
     */
    Tensor batchSimilarity(
        const std::vector<Atom::Handle>& atoms1,
        const std::vector<Atom::Handle>& atoms2) const {
        
        if (atoms1.size() != atoms2.size()) {
            throw std::runtime_error("Atom vectors must have same size");
        }
        
        if (atoms1.empty()) {
            return torch::empty({0});
        }
        
        auto emb1 = extractEmbeddings(atoms1);
        auto emb2 = extractEmbeddings(atoms2);
        
        // Compute cosine similarity
        return torch::cosine_similarity(emb1, emb2, /*dim=*/1);
    }
    
    /**
     * Batch pattern matching
     * 
     * Find all matches for a pattern across multiple target atoms in parallel.
     * 
     * @param space AtomSpace to search
     * @param pattern Pattern to match
     * @param targets Candidate atoms to match against
     * @return Vector of (atom, bindings) pairs for successful matches
     */
    std::vector<std::pair<Atom::Handle, PatternMatcher::VariableBinding>>
    batchPatternMatch(
        AtomSpace& space,
        Atom::Handle pattern,
        const std::vector<Atom::Handle>& targets) const {
        
        std::vector<std::pair<Atom::Handle, PatternMatcher::VariableBinding>> results;
        
        // For each target, try to match pattern
        for (const auto& target : targets) {
            PatternMatcher::VariableBinding bindings;
            if (PatternMatcher::match(pattern, target, bindings)) {
                results.emplace_back(target, bindings);
            }
        }
        
        return results;
    }
    
    /**
     * Compute truth value distributions
     * 
     * Given a set of atoms, compute statistical distributions of their
     * truth values (mean, variance, etc.)
     * 
     * @param atoms Set of atoms
     * @return Tensor with [mean_strength, mean_confidence, var_strength, var_confidence]
     */
    Tensor computeTruthValueDistribution(
        const std::vector<Atom::Handle>& atoms) const {
        
        if (atoms.empty()) {
            return torch::zeros({4});
        }
        
        auto tv = extractTruthValues(atoms);
        
        // Compute statistics
        auto mean = torch::mean(tv, /*dim=*/0);
        auto var = torch::var(tv, /*dim=*/0, /*unbiased=*/false);
        
        return torch::cat({mean, var});
    }
    
    /**
     * Filter atoms by truth value threshold
     * 
     * Efficiently filter large sets of atoms based on truth value criteria.
     * 
     * @param atoms Set of atoms to filter
     * @param minStrength Minimum strength threshold
     * @param minConfidence Minimum confidence threshold
     * @return Filtered vector of atoms
     */
    std::vector<Atom::Handle> filterByTruthValue(
        const std::vector<Atom::Handle>& atoms,
        float minStrength = 0.0f,
        float minConfidence = 0.0f) const {
        
        std::vector<Atom::Handle> filtered;
        
        for (const auto& atom : atoms) {
            auto tv = atom->getTruthValue();
            float strength = TruthValue::getStrength(tv);
            float confidence = TruthValue::getConfidence(tv);
            
            if (strength >= minStrength && confidence >= minConfidence) {
                filtered.push_back(atom);
            }
        }
        
        return filtered;
    }
    
    /**
     * Batch inference rule application
     * 
     * Apply an inference rule to all valid premise combinations in parallel.
     * 
     * @param atoms Set of atoms to consider as premises
     * @param ruleFunction Function implementing the inference rule
     * @return Vector of inferred truth values
     */
    std::vector<Tensor> batchInferenceRule(
        const std::vector<Atom::Handle>& atoms,
        std::function<Tensor(const Tensor&, const Tensor&)> ruleFunction) const {
        
        std::vector<Tensor> results;
        
        // Apply rule to all pairs
        for (size_t i = 0; i < atoms.size(); ++i) {
            for (size_t j = i + 1; j < atoms.size(); ++j) {
                auto tv1 = atoms[i]->getTruthValue();
                auto tv2 = atoms[j]->getTruthValue();
                results.push_back(ruleFunction(tv1, tv2));
            }
        }
        
        return results;
    }
    
    /**
     * Set inference mode
     */
    void setInferenceMode(InferenceMode mode) {
        mode_ = mode;
    }
    
    /**
     * Get current inference mode
     */
    InferenceMode getInferenceMode() const {
        return mode_;
    }
    
    /**
     * Set batch size for GPU operations
     *
     * In AUTO mode this also acts as the dispatch heuristic (T2.5):
     * workloads smaller than the batch size stay on the CPU, larger
     * workloads are dispatched to the GPU when one is available.
     */
    void setBatchSize(size_t batchSize) {
        batchSize_ = batchSize;
    }

    /**
     * Get current batch size
     */
    size_t getBatchSize() const {
        return batchSize_;
    }

    /**
     * Minimum number of lanes before AUTO mode dispatches to the GPU.
     */
    void setGPUThreshold(size_t threshold) { gpuThreshold_ = threshold; }
    size_t getGPUThreshold() const { return gpuThreshold_; }

    /**
     * Force/disable GPU usage. Defaults to automatic: GPU is used when
     * mode_ == GPU, or when mode_ == AUTO, a CUDA device is available,
     * and the batch is large enough. CPU fallback is automatic (T2.5).
     */
    void setUseGPU(bool useGPU) { useGPU_ = useGPU; }
    bool getUseGPU() const { return useGPU_; }

    /**
     * Device selected for a batch of the given size under the current
     * mode/heuristic. Public so callers (e.g. benchmarks, diagnostics)
     * can inspect dispatch decisions.
     */
    torch::Device deviceFor(size_t batchSize) const {
        bool wantGPU = useGPU_ && torch::cuda::is_available() &&
            (mode_ == InferenceMode::GPU ||
             (mode_ == InferenceMode::AUTO && batchSize >= gpuThreshold_));
        return wantGPU ? torch::Device(torch::kCUDA)
                       : torch::Device(torch::kCPU);
    }

private:
    InferenceMode mode_;
    size_t batchSize_;
    size_t gpuThreshold_ = 1024;
    bool useGPU_ = true;

    /**
     * Resolve the device for a pair of truth-value tensors (T2.5).
     *
     * Device selection heuristic:
     *  - If either input is already on a CUDA device, stay on that device.
     *  - In GPU mode, use CUDA when available, otherwise fall back to CPU.
     *  - In AUTO mode, use CUDA when available and the batch is large
     *    enough (>= gpuThreshold_, which defaults to 1024 and tracks any
     *    explicitly-set batch size), otherwise stay on the CPU.
     *  - In CPU mode, always use the CPU.
     */
    torch::Device resolveDevice(const Tensor& tv1, const Tensor& tv2) const {
        if (tv1.is_cuda()) return tv1.device();
        if (tv2.is_cuda()) return tv2.device();
        return deviceFor(static_cast<size_t>(tv1.size(0)));
    }

    /**
     * Validate that two tensors are matching [n, 2] truth-value lanes.
     */
    static void checkTVPair(const Tensor& tv1, const Tensor& tv2) {
        if (tv1.dim() != 2 || tv1.size(1) != 2 ||
            tv2.dim() != 2 || tv2.size(1) != 2) {
            throw std::runtime_error(
                "Truth value tensors must have shape [n, 2]");
        }
        if (tv1.size(0) != tv2.size(0)) {
            throw std::runtime_error(
                "Truth value tensors must have same number of lanes");
        }
    }

    /**
     * Dispatch a batched truth-value contraction to the resolved device
     * (GPU with automatic CPU fallback) and run fn on the [n, 2] lanes.
     */
    template <typename Fn>
    Tensor dispatchTV(const Tensor& tv1, const Tensor& tv2, Fn&& fn) const {
        auto options = tv1.options().dtype(torch::kFloat32);
        auto empty = torch::empty({0, 2}, options);
        if (tv1.size(0) == 0) {
            return empty;
        }

        auto device = resolveDevice(tv1, tv2);
        auto t1 = tv1.to(options.device(device));
        auto t2 = tv2.to(options.device(device));

        Tensor result = fn(t1, t2);
        // Keep results on the same device as the first input so the batch
        // path is drop-in compatible with the scalar reference.
        return result.to(tv1.device());
    }

    
    /**
     * Extract truth values from atoms into tensor
     */
    Tensor extractTruthValues(const std::vector<Atom::Handle>& atoms) const {
        std::vector<float> values;
        values.reserve(atoms.size() * 2);
        
        for (const auto& atom : atoms) {
            auto tv = atom->getTruthValue();
            values.push_back(TruthValue::getStrength(tv));
            values.push_back(TruthValue::getConfidence(tv));
        }
        
        return torch::tensor(values).reshape({static_cast<int64_t>(atoms.size()), 2});
    }
    
    /**
     * Extract embeddings from atoms into tensor
     */
    Tensor extractEmbeddings(const std::vector<Atom::Handle>& atoms) const {
        std::vector<Tensor> embeddings;
        
        for (const auto& atom : atoms) {
            if (atom->hasEmbedding()) {
                embeddings.push_back(atom->getEmbedding());
            } else {
                // Use zero embedding if not available
                int64_t dim = embeddings.empty() ? 128 : embeddings[0].size(0);
                embeddings.push_back(torch::zeros({dim}));
            }
        }
        
        return torch::stack(embeddings);
    }
    
    /**
     * Apply logical operation to truth value tensors
     */
    Tensor applyLogicalOperation(
        const Tensor& tv1,
        const Tensor& tv2,
        LogicalOperation operation) const {
        
        auto s1 = tv1.index({torch::indexing::Slice(), 0});
        auto c1 = tv1.index({torch::indexing::Slice(), 1});
        auto s2 = tv2.index({torch::indexing::Slice(), 0});
        auto c2 = tv2.index({torch::indexing::Slice(), 1});
        
        Tensor strength, confidence;
        
        switch (operation) {
            case LogicalOperation::AND:
                strength = s1 * s2;
                confidence = (c1 * c2 * (s1 + s2)) / (1.0f + s1 * s2 + 1e-7f);
                break;
                
            case LogicalOperation::OR:
                strength = s1 + s2 - s1 * s2;
                confidence = (c1 * c2) / (c1 + c2 - c1 * c2 + 1e-7f);
                break;
                
            case LogicalOperation::IMPLIES:
                strength = 1.0f - s1 + s1 * s2;
                confidence = (c1 * c2) / (1.0f + torch::abs(s1 - s2) + 1e-7f);
                break;
                
            case LogicalOperation::EQUIVALENT:
                strength = 1.0f - torch::abs(s1 - s2);
                confidence = (c1 * c2) / (1.0f + torch::abs(s1 - s2) + 1e-7f);
                break;
                
            case LogicalOperation::XOR:
                strength = (s1 + s2 - 2.0f * s1 * s2).abs();
                confidence = torch::min(c1, c2);
                break;
                
            default:
                strength = s1 * s2;
                confidence = torch::min(c1, c2);
        }
        
        // Clamp to [0, 1]
        strength = torch::clamp(strength, 0.0f, 1.0f);
        confidence = torch::clamp(confidence, 0.0f, 1.0f);
        
        return torch::stack({strength, confidence}, /*dim=*/1);
    }
    
    /**
     * Apply negation to truth values
     */
    Tensor applyNegation(const Tensor& tv) const {
        auto s = tv.index({torch::indexing::Slice(), 0});
        auto c = tv.index({torch::indexing::Slice(), 1});
        
        auto negStrength = 1.0f - s;
        
        return torch::stack({negStrength, c}, /*dim=*/1);
    }
    
    /**
     * Apply deduction formula in batch
     *
     * Tensor-contraction form of the scalar TruthValue::deduction
     * reference (FR-2.1, FR-2.5): the lanes are contracted element-wise
     * so the result matches the scalar formula on every lane.
     */
    Tensor applyDeduction(const Tensor& tv1, const Tensor& tv2) const {
        auto s1 = tv1.index({torch::indexing::Slice(), 0});
        auto c1 = tv1.index({torch::indexing::Slice(), 1});
        auto s2 = tv2.index({torch::indexing::Slice(), 0});
        auto c2 = tv2.index({torch::indexing::Slice(), 1});

        // Deduction strength: product of strengths
        auto strength = s1 * s2;

        // Confidence: geometric mean of confidences, weighted by strengths
        // (same formula as the scalar TruthValue::deduction reference)
        auto confidence =
            (c1 * c2 * (s1 + s2)) / (1.0f + s1 * s2 + TruthValue::EPSILON);

        // Clamp to [0, 1]
        strength = torch::clamp(strength, 0.0f, 1.0f);
        confidence = torch::clamp(confidence, 0.0f, 1.0f);

        return torch::stack({strength, confidence}, /*dim=*/1);
    }

    /**
     * Apply induction formula in batch: (A→B, A→C) ⊢ B→C (FR-2.2).
     *
     * Strength: P(B→C) ≈ s1 * s2 (both consequents follow from A).
     * Confidence: symmetric counterpart of the deduction confidence,
     * discounted by the TruthValue::INDUCTION_DISCOUNT uncertainty factor.
     */
    Tensor applyInduction(const Tensor& tv1, const Tensor& tv2) const {
        auto s1 = tv1.index({torch::indexing::Slice(), 0});
        auto c1 = tv1.index({torch::indexing::Slice(), 1});
        auto s2 = tv2.index({torch::indexing::Slice(), 0});
        auto c2 = tv2.index({torch::indexing::Slice(), 1});

        auto strength = s1 * s2;

        auto confidence = (c1 * c2 * (s1 + s2)) /
            (1.0f + s1 * s2 + TruthValue::EPSILON);
        confidence = confidence * TruthValue::INDUCTION_DISCOUNT;

        strength = torch::clamp(strength, 0.0f, 1.0f);
        confidence = torch::clamp(confidence, 0.0f, 1.0f);

        return torch::stack({strength, confidence}, /*dim=*/1);
    }

    /**
     * Apply abduction formula in batch: (A→C, B→C) ⊢ A→B (FR-2.2).
     *
     * Strength: s1 * s2 (both antecedents explain the same consequent).
     * Confidence: matches the scalar TruthValue::abduction reference,
     * discounted by the default 0.8 "reasoning to best explanation"
     * uncertainty factor.
     */
    Tensor applyAbduction(const Tensor& tv1, const Tensor& tv2) const {
        auto s1 = tv1.index({torch::indexing::Slice(), 0});
        auto c1 = tv1.index({torch::indexing::Slice(), 1});
        auto s2 = tv2.index({torch::indexing::Slice(), 0});
        auto c2 = tv2.index({torch::indexing::Slice(), 1});

        auto strength = s1 * s2;

        // Scalar reference: confidence = (c1 * c2 * s2) / (1 + s1) * 0.8
        auto confidence = (c1 * c2 * s2) / (1.0f + s1 + TruthValue::EPSILON);
        confidence = confidence * TruthValue::ABDUCTION_DISCOUNT;

        strength = torch::clamp(strength, 0.0f, 1.0f);
        confidence = torch::clamp(confidence, 0.0f, 1.0f);

        return torch::stack({strength, confidence}, /*dim=*/1);
    }

    /**
     * Apply revision formula in batch: merge evidence lane-wise (FR-2.3).
     *
     * Strength: confidence-weighted average of the lane strengths.
     * Confidence: combined (noisy-OR style) evidence strength, capped at
     * 0.99 like the scalar TruthValue::revision reference.
     */
    Tensor applyRevision(const Tensor& tv1, const Tensor& tv2) const {
        auto s1 = tv1.index({torch::indexing::Slice(), 0});
        auto c1 = tv1.index({torch::indexing::Slice(), 1});
        auto s2 = tv2.index({torch::indexing::Slice(), 0});
        auto c2 = tv2.index({torch::indexing::Slice(), 1});

        auto totalConfidence = c1 + c2;
        auto strength = (s1 * c1 + s2 * c2) /
            (totalConfidence + TruthValue::EPSILON);

        auto confidence = (c1 * c2) /
            (c1 + c2 - c1 * c2 + TruthValue::EPSILON);
        confidence = torch::clamp(confidence, 0.0f, 0.99f);

        // Lanes with no evidence on either side fall back to the default
        // (maximum uncertainty) truth value, mirroring the scalar formula.
        auto noEvidence = totalConfidence <= TruthValue::EPSILON;
        strength = torch::where(noEvidence, torch::full_like(strength, 0.5f),
                                strength);
        confidence = torch::where(noEvidence,
                                  torch::zeros_like(confidence), confidence);

        strength = torch::clamp(strength, 0.0f, 1.0f);

        return torch::stack({strength, confidence}, /*dim=*/1);
    }
};

} // namespace atomspace
} // namespace at
