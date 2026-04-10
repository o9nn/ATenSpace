#pragma once

#include "Atom.h"
#include "AtomSpace.h"
#include "AttentionBank.h"
#include "TruthValue.h"
#include <vector>
#include <string>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <functional>

namespace at {
namespace atomspace {

// ======================================================================== //
//  HebbianLearnerConfig  (defined outside HebbianLearner to avoid GCC bug) //
// ======================================================================== //

/**
 * Configuration parameters for HebbianLearner.
 */
struct HebbianLearnerConfig {
    /// Learning rate α ∈ (0, 1] – how fast new activations strengthen links
    float learningRate = 0.1f;

    /// Decay rate δ ∈ [0, 1] – fraction of strength lost per decay call
    float decayRate = 0.01f;

    /// Minimum link strength below which the link is pruned (removed)
    float pruneThreshold = 0.005f;

    /// Maximum link strength (clamp upper bound)
    float maxStrength = 1.0f;

    /// Whether to create directed (asymmetric) links
    bool asymmetric = false;

    /// Whether to apply Oja's normalisation rule
    bool ojaRule = false;

    /// Minimum attentional focus STI to be considered "active"
    float minActivationSTI = 0.0f;

    HebbianLearnerConfig() = default;
};

// ======================================================================== //
//  HebbianLearner                                                            //
// ======================================================================== //

/**
 * HebbianLearner - Associative learning for AtomSpace
 *
 * Implements a Hebbian learning rule over the hypergraph:
 *  "Atoms that fire together, wire together."
 *
 * The learner:
 *  1. Observes which atoms are co-active (appear together in the attentional
 *     focus or are presented as co-activating pairs).
 *  2. Creates or strengthens HebbianLinks between co-active atoms.
 *  3. Weakens (decays) links between atoms that are not co-active over time.
 *  4. Provides query methods to retrieve association strengths.
 *  5. Supports Oja's rule (normalised variant) to prevent run-away weights.
 *
 * Truth values on HebbianLinks encode association strength:
 *  - strength  : Hebbian correlation weight  [0, 1]
 *  - confidence: Accumulated evidence count  [0, 1]
 *
 * Design notes
 * ------------
 * - Thread-safe via an internal mutex.
 * - Operates on Atom::Type::HEBBIAN_LINK (symmetric) by default; can be
 *   configured to use ASYMMETRIC_HEBBIAN_LINK.
 * - Intended to be called periodically (e.g. after every cognitive cycle).
 */
class HebbianLearner {
public:
    using Config = HebbianLearnerConfig;

    // ------------------------------------------------------------------ //
    //  Construction
    // ------------------------------------------------------------------ //

    HebbianLearner(AtomSpace& space,
                   AttentionBank& bank,
                   Config cfg = Config())
        : space_(space), bank_(bank), cfg_(std::move(cfg)) {}

    // ------------------------------------------------------------------ //
    //  Core learning interface
    // ------------------------------------------------------------------ //

    /**
     * Record a co-activation event between two atoms.
     *
     * Calling this method is equivalent to "presenting" a (source, target)
     * stimulus pair to the learner. The Hebbian link between them is
     * strengthened according to the configured learning rate.
     *
     * @param source  First co-active atom
     * @param target  Second co-active atom
     */
    void recordCoActivation(const Atom::Handle& source,
                            const Atom::Handle& target) {
        if (!source || !target || source == target) return;
        std::lock_guard<std::mutex> lock(mutex_);

        auto key = pairKey(source, target);
        auto [link, strength] = getOrCreateLink(source, target);

        // Hebbian update: Δw = α · (1 − w)
        float delta = cfg_.learningRate * (cfg_.maxStrength - strength);

        // Oja's rule: normalise by post-synaptic activity proxy
        if (cfg_.ojaRule) {
            float post = bank_.getAttentionValue(target).sti + 1.0f;
            delta /= post;
        }

        float newStrength = std::clamp(strength + delta,
                                       0.0f, cfg_.maxStrength);
        float newConf = std::clamp(
            TruthValue::getConfidence(link->getTruthValue()) +
                cfg_.learningRate * 0.1f,
            0.0f, 1.0f);

        link->setTruthValue(TruthValue::create(newStrength, newConf));
        coActivationCounts_[key]++;
    }

    /**
     * Scan the attentional focus and learn from all co-active atom pairs.
     *
     * Call this once per cognitive cycle to automatically extract Hebbian
     * correlations from attention.
     */
    void learnFromAttentionalFocus() {
        auto focus = bank_.getAttentionalFocus();

        // Filter by activation threshold
        if (cfg_.minActivationSTI > 0.0f) {
            focus.erase(
                std::remove_if(focus.begin(), focus.end(),
                    [this](const Atom::Handle& a) {
                        return bank_.getAttentionValue(a).sti <
                               cfg_.minActivationSTI;
                    }),
                focus.end());
        }

        // Record all pairs
        for (size_t i = 0; i < focus.size(); ++i) {
            for (size_t j = i + 1; j < focus.size(); ++j) {
                recordCoActivation(focus[i], focus[j]);
            }
        }
    }

    /**
     * Apply weight decay to all Hebbian links.
     *
     * Should be called periodically (e.g. once per cognitive cycle) to
     * allow infrequently used associations to fade.
     */
    void decay() {
        std::lock_guard<std::mutex> lock(mutex_);

        auto allAtoms = space_.getAtoms();
        std::vector<Atom::Handle> toRemove;

        for (const auto& atom : allAtoms) {
            if (!isHebbianLink(atom)) continue;

            auto tv = atom->getTruthValue();
            if (!tv.defined() || tv.numel() < 2) continue;

            float s = TruthValue::getStrength(tv);
            float c = TruthValue::getConfidence(tv);

            float decayed = s * (1.0f - cfg_.decayRate);
            if (decayed < cfg_.pruneThreshold) {
                toRemove.push_back(atom);
            } else {
                atom->setTruthValue(TruthValue::create(decayed, c));
            }
        }

        // Remove pruned links (drop lock first to avoid deadlock with removeAtom)
        for (const auto& link : toRemove) {
            space_.removeAtom(link);
        }
    }

    // ------------------------------------------------------------------ //
    //  Query interface
    // ------------------------------------------------------------------ //

    /**
     * Get the Hebbian link between two atoms, or nullptr if none exists.
     */
    Atom::Handle getLink(const Atom::Handle& a,
                         const Atom::Handle& b) const {
        if (!a || !b) return nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        return findLink(a, b);
    }

    /**
     * Get the association strength between two atoms (0 if no link).
     */
    float getStrength(const Atom::Handle& a,
                      const Atom::Handle& b) const {
        auto link = getLink(a, b);
        if (!link) return 0.0f;
        auto tv = link->getTruthValue();
        if (!tv.defined() || tv.numel() < 1) return 0.0f;
        return TruthValue::getStrength(tv);
    }

    /**
     * Get all atoms associated with the given atom, sorted by strength.
     *
     * @param atom       Source atom
     * @param minStrength  Minimum association strength to include
     * @return            (neighbour, strength) pairs, sorted descending
     */
    std::vector<std::pair<Atom::Handle, float>>
    getAssociates(const Atom::Handle& atom,
                  float minStrength = 0.0f) const {

        if (!atom) return {};
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::pair<Atom::Handle, float>> result;

        for (const auto& weakLink : atom->getIncomingSet()) {
            auto link = weakLink.lock();
            if (!link || !isHebbianLink(link)) continue;

            auto tv = link->getTruthValue();
            if (!tv.defined() || tv.numel() < 2) continue;
            float s = TruthValue::getStrength(tv);
            if (s < minStrength) continue;

            // Find the other atom in the link
            if (!link->isLink()) continue;
            const auto* lptr = static_cast<const Link*>(link.get());
            const auto& out = lptr->getOutgoingSet();

            for (const auto& other : out) {
                if (other != atom) {
                    result.push_back({other, s});
                    break;
                }
            }
        }

        // Sort descending by strength
        std::sort(result.begin(), result.end(),
                  [](const auto& x, const auto& y){
                      return x.second > y.second;
                  });
        return result;
    }

    /**
     * Get all Hebbian links in the AtomSpace.
     */
    std::vector<Atom::Handle> getAllHebbianLinks() const {
        std::vector<Atom::Handle> links;
        for (const auto& a : space_.getAtoms()) {
            if (isHebbianLink(a)) links.push_back(a);
        }
        return links;
    }

    /**
     * Total number of co-activation events recorded since construction.
     */
    size_t totalCoActivations() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = 0;
        for (const auto& [k, v] : coActivationCounts_) total += v;
        return total;
    }

    /**
     * Number of co-activations for a specific atom pair.
     */
    size_t coActivationCount(const Atom::Handle& a,
                             const Atom::Handle& b) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = pairKey(a, b);
        auto it = coActivationCounts_.find(key);
        return (it != coActivationCounts_.end()) ? it->second : 0u;
    }

    // ------------------------------------------------------------------ //
    //  Batch learning
    // ------------------------------------------------------------------ //

    /**
     * Perform N learning cycles:
     *   1. learnFromAttentionalFocus()
     *   2. decay()
     *
     * @param cycles  Number of cycles to run
     */
    void runCycles(int cycles = 1) {
        for (int i = 0; i < cycles; ++i) {
            learnFromAttentionalFocus();
            decay();
        }
    }

    /**
     * Reset all Hebbian link strengths to zero and clear counters.
     * (Does not remove the links – just zeroes them out.)
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        coActivationCounts_.clear();

        for (const auto& atom : space_.getAtoms()) {
            if (isHebbianLink(atom)) {
                atom->setTruthValue(TruthValue::create(0.0f, 0.0f));
            }
        }
    }

    // ------------------------------------------------------------------ //
    //  Configuration access
    // ------------------------------------------------------------------ //

    const Config& getConfig() const { return cfg_; }
    void setConfig(Config cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        cfg_ = std::move(cfg);
    }

private:
    AtomSpace& space_;
    AttentionBank& bank_;
    Config cfg_;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, size_t> coActivationCounts_;

    // ------------------------------------------------------------------ //
    //  Internal helpers
    // ------------------------------------------------------------------ //

    static bool isHebbianLink(const Atom::Handle& a) {
        if (!a || !a->isLink()) return false;
        auto t = a->getType();
        return t == Atom::Type::HEBBIAN_LINK ||
               t == Atom::Type::SYMMETRIC_HEBBIAN_LINK ||
               t == Atom::Type::ASYMMETRIC_HEBBIAN_LINK;
    }

    Atom::Handle findLink(const Atom::Handle& a,
                          const Atom::Handle& b) const {
        for (const auto& weakLink : a->getIncomingSet()) {
            auto link = weakLink.lock();
            if (!link || !isHebbianLink(link)) continue;
            const auto* lptr = static_cast<const Link*>(link.get());
            const auto& out = lptr->getOutgoingSet();
            if (out.size() == 2) {
                if ((out[0] == a && out[1] == b) ||
                    (out[0] == b && out[1] == a)) {
                    return link;
                }
            }
        }
        return nullptr;
    }

    /**
     * Get or create a Hebbian link between a and b.
     * Returns (link, current_strength).
     * NOTE: mutex_ must already be held by the caller.
     */
    std::pair<Atom::Handle, float>
    getOrCreateLink(const Atom::Handle& a, const Atom::Handle& b) {
        Atom::Handle link = findLink(a, b);

        if (link) {
            auto tv = link->getTruthValue();
            float s = (tv.defined() && tv.numel() >= 1)
                      ? TruthValue::getStrength(tv)
                      : 0.0f;
            return {link, s};
        }

        // Create a new link
        auto type = cfg_.asymmetric
                    ? Atom::Type::ASYMMETRIC_HEBBIAN_LINK
                    : Atom::Type::HEBBIAN_LINK;

        link = space_.addLink(type, {a, b});
        link->setTruthValue(TruthValue::create(0.0f, 0.0f));
        return {link, 0.0f};
    }

    /**
     * Canonical pair key for co-activation map.
     * Uses pointer values for uniqueness, smaller first.
     */
    static std::string pairKey(const Atom::Handle& a,
                               const Atom::Handle& b) {
        uintptr_t pa = reinterpret_cast<uintptr_t>(a.get());
        uintptr_t pb = reinterpret_cast<uintptr_t>(b.get());
        if (pa > pb) std::swap(pa, pb);
        return std::to_string(pa) + "_" + std::to_string(pb);
    }
};

// ======================================================================== //
//  Convenience factory                                                        //
// ======================================================================== //

/**
 * Create a HebbianLearner with sensible defaults for a cognitive agent.
 *
 * @param space      AtomSpace to learn in
 * @param bank       AttentionBank to observe
 * @param fast       If true, use faster (higher learning rate, more decay)
 */
inline HebbianLearner makeHebbianLearner(
        AtomSpace& space,
        AttentionBank& bank,
        bool fast = false) {

    HebbianLearnerConfig cfg;
    if (fast) {
        cfg.learningRate = 0.3f;
        cfg.decayRate    = 0.05f;
    } else {
        cfg.learningRate = 0.05f;
        cfg.decayRate    = 0.005f;
    }
    cfg.pruneThreshold = 0.01f;
    cfg.ojaRule = true;
    return HebbianLearner(space, bank, cfg);
}

} // namespace atomspace
} // namespace at

