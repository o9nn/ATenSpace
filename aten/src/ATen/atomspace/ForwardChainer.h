#pragma once

#include "Atom.h"
#include "AtomSpace.h"
#include "PatternMatcher.h"
#include "TruthValue.h"
#include "TensorLogicEngine.h"
#include "AttentionBank.h"
#include <vector>
#include <functional>
#include <memory>
#include <utility>

namespace at {
namespace atomspace {

/**
 * InferenceRule - Base class for PLN inference rules
 * 
 * An inference rule takes premises (input atoms) and produces
 * conclusions (new atoms) with computed truth values.
 */
class InferenceRule {
public:
    virtual ~InferenceRule() = default;
    
    /**
     * Get the rule name
     */
    virtual std::string getName() const = 0;
    
    /**
     * Check if the rule can be applied to the given premises
     */
    virtual bool canApply(const std::vector<Atom::Handle>& premises) const = 0;
    
    /**
     * Apply the rule to generate conclusions
     * 
     * @param premises Input atoms
     * @param space AtomSpace to create new atoms in
     * @return Vector of newly created conclusion atoms
     */
    virtual std::vector<Atom::Handle> apply(const std::vector<Atom::Handle>& premises,
                                            AtomSpace& space) = 0;
};

/**
 * DeductionRule - Implements A→B, B→C ⊢ A→C
 */
class DeductionRule : public InferenceRule {
public:
    std::string getName() const override {
        return "Deduction";
    }
    
    bool canApply(const std::vector<Atom::Handle>& premises) const override {
        if (premises.size() != 2) return false;
        
        // Both premises must be inheritance links (or other implication links)
        return premises[0]->isLink() && premises[1]->isLink() &&
               (premises[0]->getType() == Atom::Type::INHERITANCE_LINK ||
                premises[0]->getType() == Atom::Type::IMPLICATION_LINK) &&
               (premises[1]->getType() == Atom::Type::INHERITANCE_LINK ||
                premises[1]->getType() == Atom::Type::IMPLICATION_LINK);
    }
    
    std::vector<Atom::Handle> apply(const std::vector<Atom::Handle>& premises,
                                    AtomSpace& space) override {
        std::vector<Atom::Handle> conclusions;
        
        const Link* link1 = static_cast<const Link*>(premises[0].get());
        const Link* link2 = static_cast<const Link*>(premises[1].get());
        
        // Check if they chain: A→B and B→C
        if (link1->getArity() != 2 || link2->getArity() != 2) {
            return conclusions;
        }
        
        auto A = link1->getOutgoingAtom(0);
        auto B1 = link1->getOutgoingAtom(1);
        auto B2 = link2->getOutgoingAtom(0);
        auto C = link2->getOutgoingAtom(1);
        
        // Check if B matches
        if (!B1->equals(*B2)) {
            return conclusions;
        }
        
        // Create A→C
        auto conclusion = space.addLink(premises[0]->getType(), {A, C});
        
        // Compute truth value using deduction formula
        Tensor tv = TruthValue::deduction(premises[0]->getTruthValue(),
                                         premises[1]->getTruthValue());
        conclusion->setTruthValue(tv);
        
        conclusions.push_back(conclusion);
        return conclusions;
    }

    /**
     * Batched deduction offload (Iteration 2, FR-2.1 / FR-2.4).
     *
     * Applies the deduction rule to many premise pairs in one pass: the
     * structural chaining check (A→B, B→C) is still per-pair, but all
     * PLN truth-value math is delegated to a single
     * TensorLogicEngine::batchDeduction tensor contraction instead of a
     * scalar loop.
     *
     * @param premisePairs Pairs of (A→B, B→C) implication links
     * @param space AtomSpace to create conclusions in
     * @param engine TensorLogicEngine used for the batched contraction
     * @return Vector of newly created conclusion atoms (A→C)
     */
    std::vector<Atom::Handle> applyBatch(
        const std::vector<std::pair<Atom::Handle, Atom::Handle>>& premisePairs,
        AtomSpace& space,
        const TensorLogicEngine& engine) {
        std::vector<Atom::Handle> conclusions;

        // Structural filtering: keep only chaining (A→B, B→C) pairs.
        std::vector<std::pair<Atom::Handle, Atom::Handle>> valid;
        for (const auto& pair : premisePairs) {
            std::vector<Atom::Handle> premises = {pair.first, pair.second};
            if (!canApply(premises)) continue;

            const Link* link1 = static_cast<const Link*>(pair.first.get());
            const Link* link2 = static_cast<const Link*>(pair.second.get());
            if (link1->getArity() != 2 || link2->getArity() != 2) continue;
            if (!link1->getOutgoingAtom(1)->equals(*link2->getOutgoingAtom(0)))
                continue;

            valid.push_back(pair);
        }
        if (valid.empty()) {
            return conclusions;
        }

        // Gather conclusion atoms; addLink is idempotent so repeated
        // derivations of the same A→C collapse to a single atom whose
        // truth value is revised (evidence merged) below.
        std::vector<Atom::Handle> atoms1, atoms2, conclusionAtoms;
        atoms1.reserve(valid.size());
        atoms2.reserve(valid.size());
        conclusionAtoms.reserve(valid.size());
        for (const auto& pair : valid) {
            const Link* link1 = static_cast<const Link*>(pair.first.get());
            const Link* link2 = static_cast<const Link*>(pair.second.get());
            auto A = link1->getOutgoingAtom(0);
            auto C = link2->getOutgoingAtom(1);

            atoms1.push_back(pair.first);
            atoms2.push_back(pair.second);
            conclusionAtoms.push_back(
                space.addLink(pair.first->getType(), {A, C}));
        }

        // Single batched tensor contraction for all truth values.
        Tensor tvs = engine.batchDeduction(atoms1, atoms2);

        // Scatter results back; merge repeated evidence via revision.
        for (size_t i = 0; i < conclusionAtoms.size(); ++i) {
            auto& conclusion = conclusionAtoms[i];
            Tensor tv = tvs[static_cast<int64_t>(i)].clone();
            bool alreadySeen = false;
            for (const auto& existing : conclusions) {
                if (existing == conclusion) { alreadySeen = true; break; }
            }
            if (alreadySeen) {
                conclusion->setTruthValue(TruthValue::revision(
                    conclusion->getTruthValue(), tv));
            } else {
                conclusion->setTruthValue(tv);
                conclusions.push_back(conclusion);
            }
        }

        return conclusions;
    }
};

/**
 * InductionRule - Generalize from specific instances
 */
class InductionRule : public InferenceRule {
public:
    std::string getName() const override {
        return "Induction";
    }
    
    bool canApply(const std::vector<Atom::Handle>& premises) const override {
        // Need at least one evaluation link
        if (premises.empty()) return false;
        
        for (const auto& premise : premises) {
            if (premise->getType() != Atom::Type::EVALUATION_LINK) {
                return false;
            }
        }
        return true;
    }
    
    std::vector<Atom::Handle> apply(const std::vector<Atom::Handle>& premises,
                                    AtomSpace& space) override {
        std::vector<Atom::Handle> conclusions;
        
        // Group premises by predicate
        std::map<Atom::Handle, std::vector<Atom::Handle>> byPredicate;
        
        for (const auto& premise : premises) {
            const Link* eval = static_cast<const Link*>(premise.get());
            if (eval->getArity() >= 1) {
                auto predicate = eval->getOutgoingAtom(0);
                byPredicate[predicate].push_back(premise);
            }
        }
        
        // For each predicate, create an induced rule if we have enough evidence
        for (const auto& [predicate, instances] : byPredicate) {
            if (instances.size() >= 2) {
                // Count positive instances (high truth value)
                int positiveCount = 0;
                for (const auto& instance : instances) {
                    float strength = TruthValue::getStrength(instance->getTruthValue());
                    if (strength > 0.5f) {
                        positiveCount++;
                    }
                }
                
                // Create a general rule about this predicate
                // This is simplified - real induction would extract the pattern
                Tensor tv = TruthValue::induction(positiveCount, instances.size());
                
                // We don't create a specific conclusion here without more context
                // This is a placeholder for more sophisticated induction
            }
        }
        
        return conclusions;
    }
};

/**
 * AbductionRule - Reason to best explanation: B, A→B ⊢ A
 */
class AbductionRule : public InferenceRule {
public:
    std::string getName() const override {
        return "Abduction";
    }
    
    bool canApply(const std::vector<Atom::Handle>& premises) const override {
        // Need an observation and a rule
        return premises.size() == 2 &&
               premises[1]->isLink() &&
               (premises[1]->getType() == Atom::Type::INHERITANCE_LINK ||
                premises[1]->getType() == Atom::Type::IMPLICATION_LINK);
    }
    
    std::vector<Atom::Handle> apply(const std::vector<Atom::Handle>& premises,
                                    AtomSpace& space) override {
        std::vector<Atom::Handle> conclusions;
        
        auto observation = premises[0];
        const Link* rule = static_cast<const Link*>(premises[1].get());
        
        if (rule->getArity() != 2) {
            return conclusions;
        }
        
        auto A = rule->getOutgoingAtom(0);
        auto B = rule->getOutgoingAtom(1);
        
        // Check if observation matches B
        if (!observation->equals(*B)) {
            return conclusions;
        }
        
        // Abduce A
        auto conclusion = A;
        
        // Compute truth value using abduction formula
        Tensor tv = TruthValue::abduction(observation->getTruthValue(),
                                         rule->getTruthValue());
        conclusion->setTruthValue(tv);
        
        conclusions.push_back(conclusion);
        return conclusions;
    }
};

/**
 * ForwardChainer - Forward chaining inference engine
 * 
 * Applies inference rules to derive new knowledge from existing atoms.
 * Uses attention values to prioritize which inferences to perform.
 */
class ForwardChainer {
public:
    ForwardChainer(AtomSpace& space)
        : space_(space), maxIterations_(100), confidenceThreshold_(0.1f),
          maxSteps_(10000),
          tensorLogic_(std::make_shared<TensorLogicEngine>()) {
        // Register default rules
        addRule(std::make_shared<DeductionRule>());
        addRule(std::make_shared<InductionRule>());
        addRule(std::make_shared<AbductionRule>());
    }

    /**
     * Add an inference rule
     */
    void addRule(std::shared_ptr<InferenceRule> rule) {
        rules_.push_back(rule);
    }

    /**
     * Set maximum iterations
     */
    void setMaxIterations(int maxIter) {
        maxIterations_ = maxIter;
    }

    /**
     * Set minimum confidence threshold for new conclusions
     */
    void setConfidenceThreshold(float threshold) {
        confidenceThreshold_ = threshold;
    }

    /**
     * Set the global step budget (Iteration 1, FR-1.4).
     *
     * A hard cap on the total number of rule applications across a run,
     * guaranteeing termination even on cyclic or densely connected
     * knowledge bases.
     */
    void setMaxSteps(int maxSteps) { maxSteps_ = maxSteps; }
    int getMaxSteps() const { return maxSteps_; }

    /**
     * Enable/disable batching eligible deduction premise pairs through
     * TensorLogicEngine (Iteration 2, FR-2.4). Enabled by default; the
     * scalar per-pair path is kept as an opt-out fallback.
     */
    void setBatchOffload(bool enable) { batchOffload_ = enable; }
    bool getBatchOffload() const { return batchOffload_; }

    /**
     * Minimum number of eligible deduction pairs per iteration before the
     * batch path is used (batchSize heuristic, T2.5).
     */
    void setMinBatchSize(size_t n) { minBatchSize_ = n; }
    size_t getMinBatchSize() const { return minBatchSize_; }

    /**
     * Set the TensorLogicEngine used for batch rule offload.
     */
    void setTensorLogicEngine(std::shared_ptr<TensorLogicEngine> engine) {
        if (engine) tensorLogic_ = std::move(engine);
    }
    std::shared_ptr<TensorLogicEngine> getTensorLogicEngine() const {
        return tensorLogic_;
    }

    /**
     * Run forward chaining to exhaustion, max iterations, or step budget.
     *
     * @param attentionBank Optional attention bank for priority guidance
     * @return Number of new atoms created
     */
    int run(AttentionBank* attentionBank = nullptr) {
        int totalNewAtoms = 0;
        int stepsUsed = 0;

        for (int iteration = 0; iteration < maxIterations_; ++iteration) {
            int newAtomsThisIteration =
                performIteration(attentionBank, nullptr, stepsUsed);
            totalNewAtoms += newAtomsThisIteration;

            // Stop if no new atoms were created (fixpoint / cycle guard)
            if (newAtomsThisIteration == 0) {
                break;
            }
            // Stop if the global step budget is exhausted
            if (stepsUsed >= maxSteps_) {
                break;
            }
        }

        return totalNewAtoms;
    }
    
    /**
     * Perform a single forward chaining step
     * 
     * @param target Optional target atom to focus inference on
     * @param attentionBank Optional attention bank for priority
     * @return Number of new atoms created
     */
    int step(Atom::Handle target = nullptr, AttentionBank* attentionBank = nullptr) {
        int stepsUsed = 0;
        return performIteration(attentionBank, target, stepsUsed);
    }
    
    /**
     * Apply all applicable rules to a set of premises
     * 
     * @param premises Input atoms
     * @return New conclusions
     */
    std::vector<Atom::Handle> applyRules(const std::vector<Atom::Handle>& premises) {
        std::vector<Atom::Handle> allConclusions;
        
        for (const auto& rule : rules_) {
            if (rule->canApply(premises)) {
                auto conclusions = rule->apply(premises, space_);
                
                // Filter by confidence threshold
                for (const auto& conclusion : conclusions) {
                    float conf = TruthValue::getConfidence(conclusion->getTruthValue());
                    if (conf >= confidenceThreshold_) {
                        allConclusions.push_back(conclusion);
                    }
                }
            }
        }
        
        return allConclusions;
    }
    
private:
    /**
     * Perform one iteration of forward chaining
     */
    int performIteration(AttentionBank* attentionBank,
                         Atom::Handle target,
                         int& stepsUsed) {
        size_t initialSize = space_.size();

        // Get all atoms, optionally sorted by attention
        std::vector<Atom::Handle> atoms;
        if (attentionBank) {
            atoms = attentionBank->getAttentionalFocus();
            // Also include some additional atoms
            auto atomSet = space_.getAtoms();
            size_t sampleSize = std::min(size_t(50), atomSet.size());
            size_t count = 0;
            for (const auto& a : atomSet) {
                if (count >= sampleSize || atoms.size() >= 100) break;
                atoms.push_back(a);
                ++count;
            }
        } else {
            auto atomSet = space_.getAtoms();
            atoms.assign(atomSet.begin(), atomSet.end());
        }

        // Try to apply rules to pairs of atoms.  Two guards ensure
        // termination (FR-1.4):
        //  - a per-iteration cap on newly created atoms, and
        //  - the shared global step budget `stepsUsed` / maxSteps_.
        // Because addLink is idempotent, cyclic rule applications stop
        // producing new atoms once every derivable link already exists.
        //
        // Iteration 2 (FR-2.4): eligible deduction premise pairs are
        // collected per iteration and offloaded to TensorLogicEngine as a
        // single batchDeduction tensor contraction instead of firing the
        // scalar rule once per pair. Other rules keep the scalar path.
        std::shared_ptr<DeductionRule> deductionRule;
        if (batchOffload_) {
            for (const auto& rule : rules_) {
                auto candidate = std::dynamic_pointer_cast<DeductionRule>(rule);
                if (candidate) { deductionRule = candidate; break; }
            }
        }

        int newAtoms = 0;
        std::vector<std::pair<Atom::Handle, Atom::Handle>> deductionPairs;
        for (size_t i = 0; i < atoms.size() && newAtoms < 100; ++i) {
            for (size_t j = i + 1; j < atoms.size() && newAtoms < 100; ++j) {
                if (stepsUsed >= maxSteps_) break;
                ++stepsUsed;

                std::vector<Atom::Handle> premises = {atoms[i], atoms[j]};
                for (const auto& rule : rules_) {
                    // Deduction pairs are deferred to the batch path.
                    if (deductionRule && rule == deductionRule) {
                        if (rule->canApply(premises)) {
                            // Try both orientations: deduction only fires
                            // when the pair chains as (A→B, B→C), which
                            // depends on iteration order over the atom set.
                            deductionPairs.emplace_back(atoms[i], atoms[j]);
                            deductionPairs.emplace_back(atoms[j], atoms[i]);
                        }
                        continue;
                    }
                    if (!rule->canApply(premises)) continue;

                    auto conclusions = rule->apply(premises, space_);
                    for (const auto& conclusion : conclusions) {
                        float conf = TruthValue::getConfidence(
                            conclusion->getTruthValue());
                        if (conf >= confidenceThreshold_) {
                            ++newAtoms;
                        }
                    }
                }
            }
        }

        // Offload the collected deduction pairs to the vectorised path.
        // The iteration cap is driven by the number of genuinely new atoms
        // (space size delta), not by how many conclusions pass the
        // confidence filter, so cyclic rule applications still terminate.
        if (deductionRule &&
            deductionPairs.size() >= minBatchSize_) {
            size_t before = space_.size();
            deductionRule->applyBatch(deductionPairs, space_, *tensorLogic_);
            newAtoms += static_cast<int>(space_.size() - before);
        } else {
            // Scalar fallback for small batches or when offload is disabled.
            for (const auto& pair : deductionPairs) {
                std::vector<Atom::Handle> premises = {pair.first,
                                                      pair.second};
                size_t before = space_.size();
                auto conclusions = deductionRule->apply(premises, space_);
                for (const auto& conclusion : conclusions) {
                    float conf = TruthValue::getConfidence(
                        conclusion->getTruthValue());
                    if (conf >= confidenceThreshold_ &&
                        space_.size() > before) {
                        ++newAtoms;
                    }
                }
            }
        }

        return static_cast<int>(space_.size() - initialSize);
    }

    AtomSpace& space_;
    std::vector<std::shared_ptr<InferenceRule>> rules_;
    int maxIterations_;
    float confidenceThreshold_;
    int maxSteps_;
    bool batchOffload_ = true;
    size_t minBatchSize_ = 1;
    std::shared_ptr<TensorLogicEngine> tensorLogic_;
};

} // namespace atomspace
} // namespace at
