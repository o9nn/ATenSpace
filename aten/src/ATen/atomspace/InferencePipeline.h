#pragma once

#include "Atom.h"
#include "AtomSpace.h"
#include "PatternMatcher.h"
#include "ForwardChainer.h"
#include "BackwardChainer.h"
#include "TruthValue.h"
#include "AttentionBank.h"
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <chrono>
#include <optional>
#include <unordered_map>

namespace at {
namespace atomspace {

// ======================================================================== //
//  Pipeline building blocks                                                  //
// ======================================================================== //

/**
 * InferenceStep - Base class for a single step in an inference pipeline.
 *
 * A step receives a set of atoms (the "working set") and may:
 *  - Apply inference rules to derive new atoms
 *  - Filter, transform, or annotate existing atoms
 *  - Query the AtomSpace for additional context
 *
 * Steps are stateless: all required state lives in the AtomSpace and in
 * the InferencePipelineContext passed to each invocation.
 */
class InferenceStep {
public:
    virtual ~InferenceStep() = default;

    /** Human-readable name of this step */
    virtual std::string getName() const = 0;

    /**
     * Execute the step.
     *
     * @param workingSet  Atoms being processed (may be modified in place)
     * @param space       Shared AtomSpace
     * @return            true if the step produced or modified at least one atom
     */
    virtual bool execute(std::vector<Atom::Handle>& workingSet,
                         AtomSpace& space) = 0;
};

// ======================================================================== //
//  Concrete step implementations                                             //
// ======================================================================== //

/**
 * ForwardChainingStep - Run one round of forward chaining over the working set.
 */
class ForwardChainingStep : public InferenceStep {
public:
    explicit ForwardChainingStep(int maxRounds = 1)
        : maxRounds_(maxRounds) {}

    std::string getName() const override { return "ForwardChaining"; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& space) override {
        ForwardChainer chainer(space);

        size_t before = space.size();
        for (int r = 0; r < maxRounds_; ++r) {
            chainer.run();
        }

        // Collect newly derived atoms into the working set
        auto allAtoms = space.getAtoms();
        for (const auto& a : allAtoms) {
            if (std::find(workingSet.begin(), workingSet.end(), a) ==
                workingSet.end()) {
                workingSet.push_back(a);
            }
        }
        return space.size() > before;
    }

private:
    int maxRounds_;
};

/**
 * BackwardChainingStep - Use backward chaining to prove a goal.
 */
class BackwardChainingStep : public InferenceStep {
public:
    explicit BackwardChainingStep(Atom::Handle goal, int maxDepth = 5)
        : goal_(std::move(goal)), maxDepth_(maxDepth) {}

    std::string getName() const override { return "BackwardChaining"; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& space) override {
        BackwardChainer chainer(space);
        auto proofs = chainer.prove(goal_);
        (void)maxDepth_; // depth configurable in future

        bool derived = false;
        for (const auto& proof : proofs) {
            if (!proof) continue;
            // Add the proved goal to the working set
            if (std::find(workingSet.begin(), workingSet.end(), proof->goal) ==
                workingSet.end()) {
                workingSet.push_back(proof->goal);
                derived = true;
            }
        }
        return derived;
    }

private:
    Atom::Handle goal_;
    int maxDepth_;
};

/**
 * FilterStep - Retain only atoms satisfying a predicate.
 */
class FilterStep : public InferenceStep {
public:
    using Predicate = std::function<bool(const Atom::Handle&)>;

    FilterStep(std::string name, Predicate pred)
        : name_(std::move(name)), pred_(std::move(pred)) {}

    std::string getName() const override { return "Filter[" + name_ + "]"; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& /*space*/) override {
        size_t before = workingSet.size();
        workingSet.erase(
            std::remove_if(workingSet.begin(), workingSet.end(),
                           [this](const Atom::Handle& a) { return !pred_(a); }),
            workingSet.end());
        return workingSet.size() != before;
    }

private:
    std::string name_;
    Predicate pred_;
};

/**
 * TruthValueThresholdStep - Filter atoms below a minimum truth strength.
 */
class TruthValueThresholdStep : public InferenceStep {
public:
    explicit TruthValueThresholdStep(float minStrength, float minConfidence = 0.0f)
        : minStrength_(minStrength), minConfidence_(minConfidence) {}

    std::string getName() const override {
        return "TVThreshold[s>=" + std::to_string(minStrength_) + "]";
    }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& /*space*/) override {
        size_t before = workingSet.size();
        workingSet.erase(
            std::remove_if(workingSet.begin(), workingSet.end(),
                [this](const Atom::Handle& a) {
                    auto tv = a->getTruthValue();
                    if (!tv.defined() || tv.numel() < 2) return true; // remove
                    return TruthValue::getStrength(tv) < minStrength_ ||
                           TruthValue::getConfidence(tv) < minConfidence_;
                }),
            workingSet.end());
        return workingSet.size() != before;
    }

private:
    float minStrength_;
    float minConfidence_;
};

/**
 * AttentionBoostStep - Raise attention on all atoms in the working set.
 */
class AttentionBoostStep : public InferenceStep {
public:
    explicit AttentionBoostStep(AttentionBank& bank, float boost = 10.0f)
        : bank_(bank), boost_(boost) {}

    std::string getName() const override { return "AttentionBoost"; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& /*space*/) override {
        for (const auto& a : workingSet) {
            bank_.stimulate(a, boost_);
        }
        return !workingSet.empty();
    }

private:
    AttentionBank& bank_;
    float boost_;
};

/**
 * PatternMatchStep - Expand the working set by pattern-matched atoms.
 */
class PatternMatchStep : public InferenceStep {
public:
    explicit PatternMatchStep(Atom::Handle pattern)
        : pattern_(std::move(pattern)) {}

    std::string getName() const override { return "PatternMatch"; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& space) override {
        auto matches = PatternMatcher::findMatches(space, pattern_);
        bool added = false;
        for (const auto& [atom, bindings] : matches) {
            if (std::find(workingSet.begin(), workingSet.end(), atom) ==
                workingSet.end()) {
                workingSet.push_back(atom);
                added = true;
            }
        }
        return added;
    }

private:
    Atom::Handle pattern_;
};

/**
 * CustomStep - Wrap any callable as a pipeline step.
 */
class CustomStep : public InferenceStep {
public:
    using StepFn = std::function<bool(std::vector<Atom::Handle>&, AtomSpace&)>;

    CustomStep(std::string name, StepFn fn)
        : name_(std::move(name)), fn_(std::move(fn)) {}

    std::string getName() const override { return name_; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& space) override {
        return fn_(workingSet, space);
    }

private:
    std::string name_;
    StepFn fn_;
};

// ======================================================================== //
//  PLN inference steps (Phase 11)                                           //
// ======================================================================== //

/**
 * PLNDeductionStep - Apply the PLN deduction rule to link pairs in the
 * working set.
 *
 * For every pair of arity-2 InheritanceLink / ImplicationLink atoms
 * (A→B) and (B→C), creates a new link (A→C) with a deduced truth value
 * and inserts it into both the AtomSpace and the working set.
 */
class PLNDeductionStep : public InferenceStep {
public:
    explicit PLNDeductionStep(float minConfidence = 0.0f)
        : minConfidence_(minConfidence) {}

    std::string getName() const override { return "PLNDeduction"; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& space) override {
        // Collect arity-2 implication-like links
        std::vector<Atom::Handle> candidates;
        for (const auto& a : workingSet) {
            if (a->isLink() &&
                (a->getType() == Atom::Type::INHERITANCE_LINK ||
                 a->getType() == Atom::Type::IMPLICATION_LINK)) {
                const Link* l = static_cast<const Link*>(a.get());
                if (l->getArity() == 2) candidates.push_back(a);
            }
        }

        DeductionRule rule;
        size_t before = workingSet.size();

        for (size_t i = 0; i < candidates.size(); ++i) {
            for (size_t j = 0; j < candidates.size(); ++j) {
                if (i == j) continue;
                std::vector<Atom::Handle> premises = {candidates[i],
                                                      candidates[j]};
                if (!rule.canApply(premises)) continue;

                auto conclusions = rule.apply(premises, space);
                for (auto& c : conclusions) {
                    float conf = TruthValue::getConfidence(c->getTruthValue());
                    if (conf < minConfidence_) continue;
                    if (std::find(workingSet.begin(), workingSet.end(), c) ==
                        workingSet.end()) {
                        workingSet.push_back(c);
                    }
                }
            }
        }

        return workingSet.size() > before;
    }

private:
    float minConfidence_;
};

/**
 * PLNRevisionStep - Merge truth values of structurally identical atoms.
 *
 * When the working set contains two or more atoms that share the same
 * structural hash (i.e. they are logically the same atom), their truth
 * values are combined using the PLN revision formula and the duplicates
 * are removed, leaving one atom with the revised truth value.
 */
class PLNRevisionStep : public InferenceStep {
public:
    std::string getName() const override { return "PLNRevision"; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& /*space*/) override {
        // Group indices by atom hash
        std::unordered_map<size_t, std::vector<size_t>> byHash;
        for (size_t i = 0; i < workingSet.size(); ++i) {
            byHash[workingSet[i]->getHash()].push_back(i);
        }

        bool revised = false;
        // Process groups with more than one member in reverse-index order
        // so that erasing by index remains safe.
        for (auto& [h, indices] : byHash) {
            if (indices.size() < 2) continue;

            // Combine all truth values via revision
            Tensor combined = workingSet[indices[0]]->getTruthValue();
            for (size_t k = 1; k < indices.size(); ++k) {
                combined = TruthValue::revision(
                    combined, workingSet[indices[k]]->getTruthValue());
            }
            workingSet[indices[0]]->setTruthValue(combined);

            // Erase duplicates from highest index downward
            for (size_t k = indices.size() - 1; k >= 1; --k) {
                workingSet.erase(workingSet.begin() +
                                 static_cast<ptrdiff_t>(indices[k]));
            }
            revised = true;
        }
        return revised;
    }
};

/**
 * PLNAbductionStep - Infer explanatory atoms using the PLN abduction rule.
 *
 * For each "observation" atom B whose truth-value strength exceeds
 * @p minObservationStrength, and for each rule (A→B) in the working set,
 * abduces that A is likely true and adds (or truth-value-revises) A in
 * the working set.
 */
class PLNAbductionStep : public InferenceStep {
public:
    explicit PLNAbductionStep(float minObservationStrength = 0.7f,
                              float minConfidence = 0.0f)
        : minObsStrength_(minObservationStrength)
        , minConfidence_(minConfidence) {}

    std::string getName() const override { return "PLNAbduction"; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& /*space*/) override {
        size_t before = workingSet.size();

        // Observations: atoms with strength >= minObsStrength_
        std::vector<Atom::Handle> observations;
        for (const auto& a : workingSet) {
            if (TruthValue::getStrength(a->getTruthValue()) >= minObsStrength_)
                observations.push_back(a);
        }

        // Rules: arity-2 inheritance/implication links
        std::vector<Atom::Handle> rules;
        for (const auto& a : workingSet) {
            if (!a->isLink()) continue;
            if (a->getType() != Atom::Type::INHERITANCE_LINK &&
                a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            if (static_cast<const Link*>(a.get())->getArity() == 2)
                rules.push_back(a);
        }

        for (const auto& obs : observations) {
            for (const auto& rule : rules) {
                const Link* rl = static_cast<const Link*>(rule.get());
                auto B = rl->getOutgoingAtom(1);
                if (B->getHash() != obs->getHash()) continue;

                auto A = rl->getOutgoingAtom(0);
                Tensor abdTV =
                    TruthValue::abduction(obs->getTruthValue(),
                                         rule->getTruthValue());
                if (TruthValue::getConfidence(abdTV) < minConfidence_) continue;

                // Update existing entry or append a new one
                auto it = std::find(workingSet.begin(), workingSet.end(), A);
                if (it == workingSet.end()) {
                    A->setTruthValue(abdTV);
                    workingSet.push_back(A);
                } else {
                    (*it)->setTruthValue(
                        TruthValue::revision((*it)->getTruthValue(), abdTV));
                }
            }
        }

        return workingSet.size() > before;
    }

private:
    float minObsStrength_;
    float minConfidence_;
};

/**
 * PLNInductionStep - Generalise from instances using the PLN induction rule.
 *
 * For each unique target atom T that appears as the *consequent* of one or
 * more arity-2 links of @p linkType_ in the working set, counts the number
 * of distinct antecedents (instances) and emits one MemberLink(instance, T)
 * per instance carrying an induced truth value based on the instance count.
 */
class PLNInductionStep : public InferenceStep {
public:
    explicit PLNInductionStep(
            Atom::Type linkType = Atom::Type::INHERITANCE_LINK)
        : linkType_(linkType) {}

    std::string getName() const override { return "PLNInduction"; }

    bool execute(std::vector<Atom::Handle>& workingSet,
                 AtomSpace& space) override {
        // Map target-hash → (target atom, list of source atoms)
        std::unordered_map<size_t, Atom::Handle> hashToTarget;
        std::unordered_map<size_t, std::vector<Atom::Handle>> targetToSources;

        for (const auto& a : workingSet) {
            if (!a->isLink() || a->getType() != linkType_) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getArity() != 2) continue;

            auto src = l->getOutgoingAtom(0);
            auto tgt = l->getOutgoingAtom(1);
            hashToTarget[tgt->getHash()] = tgt;
            targetToSources[tgt->getHash()].push_back(src);
        }

        if (targetToSources.empty()) return false;

        size_t before = workingSet.size();

        for (auto& [h, sources] : targetToSources) {
            auto tgt = hashToTarget[h];
            int count = static_cast<int>(sources.size());
            Tensor inducedTV = TruthValue::induction(count, count);

            for (const auto& src : sources) {
                auto memberLink =
                    space.addLink(Atom::Type::MEMBER_LINK, {src, tgt});
                memberLink->setTruthValue(inducedTV);

                if (std::find(workingSet.begin(), workingSet.end(),
                              memberLink) == workingSet.end()) {
                    workingSet.push_back(memberLink);
                }
            }
        }

        return workingSet.size() > before;
    }

private:
    Atom::Type linkType_;
};

// ======================================================================== //
//  Pipeline execution context and result                                     //
// ======================================================================== //

/**
 * StepStats - Per-step execution statistics.
 */
struct StepStats {
    std::string stepName;
    bool produced = false;       ///< Did this step derive/modify atoms?
    size_t workingSetSize = 0;   ///< Working set size after the step
    double elapsedMs = 0.0;      ///< Wall-clock time in milliseconds
};

/**
 * PipelineResult - Outcome of a full pipeline run.
 */
struct PipelineResult {
    std::vector<Atom::Handle> atoms;    ///< Final working set
    std::vector<StepStats> stats;       ///< Per-step statistics
    bool converged = false;             ///< True if fixed-point was reached
    int iterationsRun = 0;              ///< Number of iterations completed

    /** Total pipeline wall time in ms */
    double totalMs() const {
        double t = 0.0;
        for (const auto& s : stats) t += s.elapsedMs;
        return t;
    }
};

// ======================================================================== //
//  InferencePipeline                                                         //
// ======================================================================== //

/**
 * InferencePipeline - Composable, ordered sequence of inference steps.
 *
 * Steps are executed in order on a shared working set of atoms.
 * The pipeline supports:
 *  - Sequential execution
 *  - Fixed-point iteration (run until no step produces new atoms)
 *  - Configurable iteration limit and timeout
 *  - Per-step and per-run statistics
 *
 * Example:
 * @code
 *   InferencePipeline pipeline(space);
 *   pipeline.addStep(std::make_shared<PatternMatchStep>(query_pattern));
 *   pipeline.addStep(std::make_shared<ForwardChainingStep>(3));
 *   pipeline.addStep(std::make_shared<TruthValueThresholdStep>(0.5f));
 *
 *   auto result = pipeline.run(seeds, false, 10);
 *   for (auto& atom : result.atoms) { ... }
 * @endcode
 */
class InferencePipeline {
public:
    explicit InferencePipeline(AtomSpace& space) : space_(space) {}

    // ------------------------------------------------------------------ //
    //  Pipeline construction
    // ------------------------------------------------------------------ //

    /** Append a step to the pipeline */
    InferencePipeline& addStep(std::shared_ptr<InferenceStep> step) {
        steps_.push_back(std::move(step));
        return *this;
    }

    /** Append a forward-chaining step */
    InferencePipeline& forwardChain(int rounds = 1) {
        return addStep(std::make_shared<ForwardChainingStep>(rounds));
    }

    /** Append a backward-chaining step toward a specific goal */
    InferencePipeline& backwardChain(Atom::Handle goal, int maxDepth = 5) {
        return addStep(
            std::make_shared<BackwardChainingStep>(std::move(goal), maxDepth));
    }

    /** Append a pattern-match expansion step */
    InferencePipeline& matchPattern(Atom::Handle pattern) {
        return addStep(std::make_shared<PatternMatchStep>(std::move(pattern)));
    }

    /** Append a truth-value threshold filter */
    InferencePipeline& filterByTV(float minStrength,
                                  float minConfidence = 0.0f) {
        return addStep(
            std::make_shared<TruthValueThresholdStep>(minStrength,
                                                      minConfidence));
    }

    /** Append a custom filter */
    InferencePipeline& filter(const std::string& name,
                              std::function<bool(const Atom::Handle&)> pred) {
        return addStep(std::make_shared<FilterStep>(name, std::move(pred)));
    }

    /** Append an attention-boost step */
    InferencePipeline& boostAttention(AttentionBank& bank, float amount = 10.0f) {
        return addStep(std::make_shared<AttentionBoostStep>(bank, amount));
    }

    /** Append a custom step from a callable */
    InferencePipeline& addCustomStep(
            const std::string& name,
            std::function<bool(std::vector<Atom::Handle>&, AtomSpace&)> fn) {
        return addStep(std::make_shared<CustomStep>(name, std::move(fn)));
    }

    /** Append a PLN deduction step (Phase 11) */
    InferencePipeline& plnDeduction(float minConfidence = 0.0f) {
        return addStep(std::make_shared<PLNDeductionStep>(minConfidence));
    }

    /** Append a PLN revision step (Phase 11) */
    InferencePipeline& plnRevision() {
        return addStep(std::make_shared<PLNRevisionStep>());
    }

    /** Append a PLN abduction step (Phase 11) */
    InferencePipeline& plnAbduction(float minObsStrength = 0.7f,
                                    float minConfidence = 0.0f) {
        return addStep(
            std::make_shared<PLNAbductionStep>(minObsStrength, minConfidence));
    }

    /** Append a PLN induction step (Phase 11) */
    InferencePipeline& plnInduction(
            Atom::Type linkType = Atom::Type::INHERITANCE_LINK) {
        return addStep(std::make_shared<PLNInductionStep>(linkType));
    }

    // ------------------------------------------------------------------ //
    //  Execution
    // ------------------------------------------------------------------ //

    /**
     * Run the pipeline.
     *
     * @param seeds          Initial working set (empty = all atoms in space)
     * @param untilFixedPoint  If true, repeat until no step changes the set
     * @param maxIterations  Maximum number of full pipeline passes
     * @return               PipelineResult with final atoms and statistics
     */
    PipelineResult run(std::vector<Atom::Handle> seeds = {},
                       bool untilFixedPoint = false,
                       int maxIterations = 1) const {

        PipelineResult result;

        // Default seed: all atoms
        if (seeds.empty()) {
            auto atomSet = space_.getAtoms();
            seeds.assign(atomSet.begin(), atomSet.end());
        }
        result.atoms = std::move(seeds);

        // Short-circuit for empty pipelines
        if (steps_.empty()) {
            result.converged = true;
            result.iterationsRun = 0;
            return result;
        }

        int iter = 0;
        bool changed = true;

        while (changed && iter < maxIterations) {
            changed = false;
            ++iter;

            for (const auto& step : steps_) {
                auto t0 = std::chrono::steady_clock::now();
                bool stepChanged = step->execute(result.atoms, space_);
                auto t1 = std::chrono::steady_clock::now();

                double ms = std::chrono::duration<double, std::milli>(
                                t1 - t0).count();

                result.stats.push_back({
                    step->getName(),
                    stepChanged,
                    result.atoms.size(),
                    ms
                });

                changed = changed || stepChanged;
            }

            if (!untilFixedPoint) break;
        }

        result.converged = !changed;
        result.iterationsRun = iter;
        return result;
    }

    /**
     * Number of steps in the pipeline.
     */
    size_t size() const { return steps_.size(); }

    /**
     * Get step names.
     */
    std::vector<std::string> stepNames() const {
        std::vector<std::string> names;
        names.reserve(steps_.size());
        for (const auto& s : steps_) names.push_back(s->getName());
        return names;
    }

private:
    AtomSpace& space_;
    std::vector<std::shared_ptr<InferenceStep>> steps_;
};

// ======================================================================== //
//  Pipeline factory helpers                                                  //
// ======================================================================== //

/**
 * Create a standard forward-reasoning pipeline:
 *   1. Pattern match seeds
 *   2. Forward chain (N rounds)
 *   3. Filter by truth value
 */
inline InferencePipeline makeForwardReasoningPipeline(
        AtomSpace& space,
        Atom::Handle seedPattern,
        float tvThreshold = 0.5f,
        int fcRounds = 3) {

    InferencePipeline p(space);
    p.matchPattern(std::move(seedPattern))
     .forwardChain(fcRounds)
     .filterByTV(tvThreshold);
    return p;
}

/**
 * Create a hypothesis verification pipeline:
 *   1. Backward chain toward goal
 *   2. Filter surviving hypotheses by confidence
 */
inline InferencePipeline makeHypothesisVerificationPipeline(
        AtomSpace& space,
        Atom::Handle goal,
        float minConfidence = 0.6f,
        int maxDepth = 5) {

    InferencePipeline p(space);
    p.backwardChain(std::move(goal), maxDepth)
     .filterByTV(0.0f, minConfidence);
    return p;
}

/**
 * Create a PLN reasoning pipeline (Phase 11):
 *   1. PLN deduction — derive transitive implications
 *   2. PLN revision  — merge duplicate truth values
 *   3. Filter by truth-value threshold
 */
inline InferencePipeline makePLNReasoningPipeline(
        AtomSpace& space,
        float tvThreshold = 0.0f,
        float minConfidence = 0.0f) {

    InferencePipeline p(space);
    p.plnDeduction(minConfidence)
     .plnRevision()
     .filterByTV(tvThreshold, minConfidence);
    return p;
}

} // namespace atomspace
} // namespace at
