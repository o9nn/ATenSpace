#pragma once

#include "Atom.h"
#include "AtomSpace.h"
#include "TimeServer.h"
#include "AttentionBank.h"
#include "PatternMatcher.h"
#include "TruthValue.h"
#include "ForwardChainer.h"
#include "BackwardChainer.h"
#include "ECAN.h"
#include "TensorLogicEngine.h"
#include <memory>
#include <vector>
#include <functional>
#include <chrono>

namespace at {
namespace atomspace {

/**
 * CognitiveEngine - Master Algorithm Integration Framework
 * 
 * The CognitiveEngine orchestrates all cognitive subsystems (PLN, ECAN,
 * pattern matching, forward/backward chaining) into a unified cognitive
 * architecture. It implements the master algorithm that coordinates:
 * 
 * - Attention allocation (ECAN)
 * - Inference (Forward/Backward chaining)
 * - Pattern recognition
 * - Temporal reasoning
 * - Learning and adaptation
 * 
 * This engine embodies the OpenCog cognitive synergy principle: intelligence
 * emerges from the interaction of multiple cognitive processes.
 */
class CognitiveEngine {
public:
    /**
     * CognitiveMode - Operating mode for the cognitive engine
     */
    enum class CognitiveMode {
        REACTIVE,      // Respond to queries only
        PROACTIVE,     // Continuous background inference
        GOAL_DIRECTED, // Focus on achieving specific goals
        EXPLORATORY,   // Explore knowledge space
        BALANCED       // Mix of all modes
    };
    
    /**
     * CognitiveMetrics - Performance metrics for the cognitive engine
     */
    struct CognitiveMetrics {
        size_t atomsProcessed = 0;
        size_t inferencesPerformed = 0;
        size_t patternsMatched = 0;
        size_t attentionUpdates = 0;
        double totalProcessingTime = 0.0;
        size_t newKnowledgeGenerated = 0;
        double averageConfidence = 0.0;
        
        void reset() {
            atomsProcessed = 0;
            inferencesPerformed = 0;
            patternsMatched = 0;
            attentionUpdates = 0;
            totalProcessingTime = 0.0;
            newKnowledgeGenerated = 0;
            averageConfidence = 0.0;
        }
    };
    
    /**
     * Construct a CognitiveEngine
     * 
     * @param space The AtomSpace to operate on
     * @param mode Operating mode
     */
    explicit CognitiveEngine(
        AtomSpace& space,
        CognitiveMode mode = CognitiveMode::BALANCED)
        : space_(space)
        , mode_(mode)
        , timeServer_(std::make_shared<TimeServer>())
        , attentionBank_(std::make_shared<AttentionBank>())
        , forwardChainer_(std::make_shared<ForwardChainer>(space))
        , backwardChainer_(std::make_shared<BackwardChainer>(space))
        , ecan_(std::make_shared<ECAN>(*attentionBank_))
        , tensorLogic_(std::make_shared<TensorLogicEngine>())
        , cycleCount_(0)
        , isRunning_(false) {
        
        // Configure subsystems
        configureSubsystems();
    }
    
    /**
     * Run a single cognitive cycle
     * 
     * A cognitive cycle consists of:
     * 1. Attention allocation (ECAN)
     * 2. Pattern matching on focused atoms
     * 3. Forward inference on important atoms
     * 4. Backward chaining for active goals
     * 5. Temporal updates
     * 6. Metric collection
     * 
     * @return Number of new atoms created this cycle
     */
    size_t runCycle() {
        auto startTime = std::chrono::high_resolution_clock::now();
        size_t initialAtomCount = space_.getAtomCount();
        
        // 1. Attention allocation
        updateAttention();
        
        // 2. Pattern matching on focused atoms
        performPatternMatching();
        
        // 3. Forward inference
        performForwardInference();
        
        // 4. Backward chaining for goals
        performBackwardChaining();
        
        // 5. Temporal updates
        updateTemporal();
        
        // Update metrics
        cycleCount_++;
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        metrics_.totalProcessingTime += elapsed.count();
        
        size_t newAtoms = space_.getAtomCount() - initialAtomCount;
        metrics_.newKnowledgeGenerated += newAtoms;
        
        return newAtoms;
    }
    
    /**
     * Run multiple cognitive cycles
     * 
     * @param numCycles Number of cycles to run
     * @return Total number of new atoms created
     */
    size_t runCycles(size_t numCycles) {
        size_t totalNew = 0;
        isRunning_ = true;
        
        for (size_t i = 0; i < numCycles && isRunning_; ++i) {
            totalNew += runCycle();
        }
        
        isRunning_ = false;
        return totalNew;
    }
    
    /**
     * Stop running cycles (for async operation)
     */
    void stop() {
        isRunning_ = false;
    }
    
    /**
     * Add a goal for the cognitive engine to pursue
     * 
     * Goals guide backward chaining and attention allocation.
     * 
     * @param goal Goal atom to achieve
     * @param priority Priority level (higher = more important)
     */
    void addGoal(Atom::Handle goal, float priority = 1.0f) {
        goals_.push_back({goal, priority});
        
        // Boost attention for goal-related atoms
        attentionBank_->stimulate(goal, priority * 50.0f);
    }
    
    /**
     * Remove a goal
     */
    void removeGoal(Atom::Handle goal) {
        goals_.erase(
            std::remove_if(goals_.begin(), goals_.end(),
                [&goal](const auto& g) { return g.first == goal; }),
            goals_.end()
        );
    }
    
    /**
     * Get current goals
     */
    const std::vector<std::pair<Atom::Handle, float>>& getGoals() const {
        return goals_;
    }
    
    /**
     * Register a pattern to watch for
     * 
     * When the pattern is matched, the callback is invoked.
     * 
     * @param pattern Pattern to watch
     * @param callback Function to call on match
     */
    void registerPattern(
        Atom::Handle pattern,
        std::function<void(Atom::Handle, const PatternMatcher::VariableBinding&)> callback) {
        
        patterns_.push_back({pattern, callback});
    }
    
    /**
     * Clear all registered patterns
     */
    void clearPatterns() {
        patterns_.clear();
    }
    
    /**
     * Add an inference rule to the forward chainer
     */
    void addInferenceRule(std::shared_ptr<InferenceRule> rule) {
        forwardChainer_->addRule(rule);
    }
    
    /**
     * Query the cognitive engine
     * 
     * Performs attention-guided inference to answer a query.
     * 
     * @param query Query pattern
     * @param maxSteps Maximum inference steps
     * @return Vector of matching atoms with bindings
     */
    std::vector<std::pair<Atom::Handle, PatternMatcher::VariableBinding>>
    query(Atom::Handle query, size_t maxSteps = 10) {
        // Get attentionally focused atoms
        auto focusAtoms = attentionBank_->getAttentionalFocus();
        
        // Try pattern matching first
        auto matches = tensorLogic_->batchPatternMatch(space_, query, focusAtoms);
        
        if (!matches.empty()) {
            return matches;
        }
        
        // If no direct matches, try backward chaining
        backwardChainer_->setMaxDepth(maxSteps);
        auto proofs = backwardChainer_->prove(query);
        
        // Convert proofs to matches
        std::vector<std::pair<Atom::Handle, PatternMatcher::VariableBinding>> results;
        for (const auto& proof : proofs) {
            if (proof) {
                results.push_back({proof->goal, {}});
            }
        }
        
        return results;
    }
    
    /**
     * Learn from examples
     * 
     * Given positive and negative examples, induce rules.
     * 
     * @param positiveExamples Examples of the pattern to learn
     * @param negativeExamples Counter-examples
     * @return Induced rule (if any)
     */
    Atom::Handle learn(
        const std::vector<Atom::Handle>& positiveExamples,
        const std::vector<Atom::Handle>& negativeExamples = {}) {
        
        if (positiveExamples.empty()) {
            return nullptr;
        }
        
        // Compute truth value from examples
        auto tv = TruthValue::induction(
            positiveExamples.size(),
            positiveExamples.size() + negativeExamples.size()
        );
        
        // Create generalized pattern (simplified - could be more sophisticated)
        // For now, just mark the first example as a learned pattern
        auto pattern = positiveExamples[0];
        pattern->setTruthValue(tv);
        
        // Boost attention for learned knowledge
        attentionBank_->stimulate(pattern, 100.0f);
        
        metrics_.atomsProcessed += positiveExamples.size() + negativeExamples.size();
        
        return pattern;
    }
    
    /**
     * Batch inference using tensor logic engine
     * 
     * Performs parallel inference across multiple atoms.
     * 
     * @param atoms Atoms to perform inference on
     * @param operation Logical operation
     * @return Vector of result truth values
     */
    std::vector<Tensor> batchInference(
        const std::vector<Atom::Handle>& atoms,
        TensorLogicEngine::LogicalOperation operation) {
        
        std::vector<Tensor> results;
        
        // Process in batches
        for (size_t i = 0; i < atoms.size(); i += tensorLogic_->getBatchSize()) {
            size_t end = std::min(i + tensorLogic_->getBatchSize(), atoms.size());
            
            std::vector<Atom::Handle> batch1(atoms.begin() + i, atoms.begin() + end);
            std::vector<Atom::Handle> batch2(atoms.begin() + i, atoms.begin() + end);
            
            auto result = tensorLogic_->batchLogicalOperation(batch1, batch2, operation);
            
            // Convert back to individual tensors
            for (int64_t j = 0; j < result.size(0); ++j) {
                results.push_back(result[j]);
            }
        }
        
        metrics_.inferencesPerformed += results.size();
        
        return results;
    }
    
    /**
     * Get cognitive metrics
     */
    const CognitiveMetrics& getMetrics() const {
        return metrics_;
    }
    
    /**
     * Reset metrics
     */
    void resetMetrics() {
        metrics_.reset();
    }
    
    /**
     * Get cycle count
     */
    size_t getCycleCount() const {
        return cycleCount_;
    }
    
    /**
     * Set cognitive mode
     */
    void setCognitiveMode(CognitiveMode mode) {
        mode_ = mode;
        configureSubsystems();
    }
    
    /**
     * Get cognitive mode
     */
    CognitiveMode getCognitiveMode() const {
        return mode_;
    }
    
    /**
     * Get TimeServer
     */
    std::shared_ptr<TimeServer> getTimeServer() {
        return timeServer_;
    }
    
    /**
     * Get AttentionBank
     */
    std::shared_ptr<AttentionBank> getAttentionBank() {
        return attentionBank_;
    }
    
    /**
     * Get ForwardChainer
     */
    std::shared_ptr<ForwardChainer> getForwardChainer() {
        return forwardChainer_;
    }
    
    /**
     * Get BackwardChainer
     */
    std::shared_ptr<BackwardChainer> getBackwardChainer() {
        return backwardChainer_;
    }
    
    /**
     * Get ECAN
     */
    std::shared_ptr<ECAN> getECAN() {
        return ecan_;
    }
    
    /**
     * Get TensorLogicEngine
     */
    std::shared_ptr<TensorLogicEngine> getTensorLogicEngine() {
        return tensorLogic_;
    }
    
    /**
     * Get AtomSpace
     */
    AtomSpace& getAtomSpace() {
        return space_;
    }

private:
    AtomSpace& space_;
    CognitiveMode mode_;
    
    // Subsystems
    std::shared_ptr<TimeServer> timeServer_;
    std::shared_ptr<AttentionBank> attentionBank_;
    std::shared_ptr<ForwardChainer> forwardChainer_;
    std::shared_ptr<BackwardChainer> backwardChainer_;
    std::shared_ptr<ECAN> ecan_;
    std::shared_ptr<TensorLogicEngine> tensorLogic_;
    
    // Goals and patterns
    std::vector<std::pair<Atom::Handle, float>> goals_;
    std::vector<std::pair<Atom::Handle, 
        std::function<void(Atom::Handle, const PatternMatcher::VariableBinding&)>>> patterns_;
    
    // State
    size_t cycleCount_;
    bool isRunning_;
    CognitiveMetrics metrics_;
    
    /**
     * Configure subsystems based on cognitive mode
     */
    void configureSubsystems() {
        switch (mode_) {
            case CognitiveMode::REACTIVE:
                // Minimal proactive inference
                forwardChainer_->setMaxIterations(1);
                ecan_->setForgettingThreshold(-100.0f); // Rarely forget
                break;
                
            case CognitiveMode::PROACTIVE:
                // Aggressive inference
                forwardChainer_->setMaxIterations(10);
                ecan_->setForgettingThreshold(0.0f); // Forget unused
                break;
                
            case CognitiveMode::GOAL_DIRECTED:
                // Focus on goals
                forwardChainer_->setMaxIterations(3);
                backwardChainer_->setMaxDepth(10);
                ecan_->setForgettingThreshold(-50.0f);
                break;
                
            case CognitiveMode::EXPLORATORY:
                // Explore broadly
                forwardChainer_->setMaxIterations(5);
                backwardChainer_->setMaxDepth(5);
                ecan_->setForgettingThreshold(10.0f); // Forget aggressively
                break;
                
            case CognitiveMode::BALANCED:
            default:
                // Balanced settings
                forwardChainer_->setMaxIterations(5);
                backwardChainer_->setMaxDepth(7);
                ecan_->setForgettingThreshold(0.0f);
                break;
        }
    }
    
    /**
     * Update attention values using ECAN
     */
    void updateAttention() {
        // Get all atoms
        auto atoms = space_.getAllAtoms();
        
        // Run ECAN cycle
        ecan_->runCycle(atoms);
        
        metrics_.attentionUpdates++;
        metrics_.atomsProcessed += atoms.size();
    }
    
    /**
     * Perform pattern matching on focused atoms
     */
    void performPatternMatching() {
        if (patterns_.empty()) {
            return;
        }
        
        auto focusAtoms = attentionBank_->getAttentionalFocus();
        
        for (const auto& [pattern, callback] : patterns_) {
            auto matches = tensorLogic_->batchPatternMatch(space_, pattern, focusAtoms);
            
            for (const auto& [atom, bindings] : matches) {
                callback(atom, bindings);
            }
            
            metrics_.patternsMatched += matches.size();
        }
    }
    
    /**
     * Perform forward inference
     */
    void performForwardInference() {
        if (mode_ == CognitiveMode::REACTIVE) {
            return; // Skip in reactive mode
        }
        
        // Use attention to guide inference
        int newAtoms = forwardChainer_->run(attentionBank_.get());
        
        metrics_.inferencesPerformed += newAtoms;
        
        // Record creation time for new atoms
        auto atoms = space_.getAllAtoms();
        for (const auto& atom : atoms) {
            if (!timeServer_->hasCreationTime(atom)) {
                timeServer_->recordCreation(atom);
            }
        }
    }
    
    /**
     * Perform backward chaining for active goals
     */
    void performBackwardChaining() {
        if (goals_.empty()) {
            return;
        }
        
        // Sort goals by priority
        auto sortedGoals = goals_;
        std::sort(sortedGoals.begin(), sortedGoals.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Try to prove top priority goals
        size_t maxGoals = (mode_ == CognitiveMode::GOAL_DIRECTED) ? sortedGoals.size() : 
                          std::min(sortedGoals.size(), size_t(3));
        
        for (size_t i = 0; i < maxGoals; ++i) {
            auto goal = sortedGoals[i].first;
            auto proofs = backwardChainer_->prove(goal);
            
            if (!proofs.empty()) {
                // Goal achieved!
                removeGoal(goal);
                metrics_.inferencesPerformed += proofs.size();
            }
        }
    }
    
    /**
     * Update temporal information
     */
    void updateTemporal() {
        // Mark focused atoms as accessed
        auto focusAtoms = attentionBank_->getAttentionalFocus();
        
        for (const auto& atom : focusAtoms) {
            timeServer_->recordAccess(atom);
        }
    }
};

} // namespace atomspace
} // namespace at
