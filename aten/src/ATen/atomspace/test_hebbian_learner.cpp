/**
 * test_hebbian_learner.cpp - Tests for HebbianLearner
 */
#include "ATenSpaceCore.h"
#include "HebbianLearner.h"
#include "AttentionBank.h"
#include <iostream>
#include <cassert>
#include <string>

using namespace at::atomspace;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "  [TEST] " << name << " ... "; \
    try {

#define END_TEST \
        std::cout << "PASS\n"; \
        ++tests_passed; \
    } catch (const std::exception& e) { \
        std::cout << "FAIL: " << e.what() << "\n"; \
        ++tests_failed; \
    } catch (...) { \
        std::cout << "FAIL (unknown exception)\n"; \
        ++tests_failed; \
    }

#define ASSERT(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond)

// ======================================================================== //
//  Tests
// ======================================================================== //

void testCoActivationCreatesLink() {
    TEST("recordCoActivation creates a HebbianLink")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner learner(space, bank);

        auto dog = createConceptNode(space, "hl-dog");
        auto cat = createConceptNode(space, "hl-cat");

        learner.recordCoActivation(dog, cat);

        auto link = learner.getLink(dog, cat);
        ASSERT(link != nullptr);
        ASSERT(link->getType() == Atom::Type::HEBBIAN_LINK ||
               link->getType() == Atom::Type::ASYMMETRIC_HEBBIAN_LINK);
    END_TEST
}

void testStrengthIncreasesWithRepeatedActivation() {
    TEST("repeated co-activations increase Hebbian strength")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner::Config cfg;
        cfg.learningRate = 0.2f;
        cfg.ojaRule = false;
        HebbianLearner learner(space, bank, cfg);

        auto a = createConceptNode(space, "hl-strength-A");
        auto b = createConceptNode(space, "hl-strength-B");

        float prev = learner.getStrength(a, b);  // 0.0
        ASSERT(prev == 0.0f);

        for (int i = 0; i < 10; ++i) {
            learner.recordCoActivation(a, b);
        }

        float after = learner.getStrength(a, b);
        ASSERT(after > prev);
    END_TEST
}

void testDecayReducesStrength() {
    TEST("decay() reduces link strength")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner::Config cfg;
        cfg.learningRate = 0.5f;
        cfg.decayRate    = 0.5f;
        cfg.pruneThreshold = 0.0f;  // Don't prune so we can observe decay
        cfg.ojaRule = false;
        HebbianLearner learner(space, bank, cfg);

        auto a = createConceptNode(space, "hl-decay-A");
        auto b = createConceptNode(space, "hl-decay-B");

        // Build up strength
        for (int i = 0; i < 5; ++i) learner.recordCoActivation(a, b);
        float before = learner.getStrength(a, b);
        ASSERT(before > 0.0f);

        learner.decay();
        float after = learner.getStrength(a, b);
        ASSERT(after < before);
    END_TEST
}

void testPruningRemovesWeakLinks() {
    TEST("decay() prunes links below pruneThreshold")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner::Config cfg;
        cfg.learningRate   = 0.01f;   // Very small
        cfg.decayRate      = 0.99f;   // Aggressive decay
        cfg.pruneThreshold = 0.05f;
        cfg.ojaRule = false;
        HebbianLearner learner(space, bank, cfg);

        auto a = createConceptNode(space, "hl-prune-A");
        auto b = createConceptNode(space, "hl-prune-B");

        learner.recordCoActivation(a, b);
        ASSERT(learner.getLink(a, b) != nullptr);

        // After a few decays with aggressive rate, link should be pruned
        for (int i = 0; i < 10; ++i) learner.decay();

        // Link should be gone or below threshold
        float s = learner.getStrength(a, b);
        ASSERT(s < cfg.pruneThreshold || learner.getLink(a, b) == nullptr);
    END_TEST
}

void testGetAssociates() {
    TEST("getAssociates returns sorted neighbours")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner::Config cfg;
        cfg.learningRate = 0.2f;
        cfg.ojaRule = false;
        HebbianLearner learner(space, bank, cfg);

        auto hub = createConceptNode(space, "hl-hub");
        auto n1  = createConceptNode(space, "hl-n1");
        auto n2  = createConceptNode(space, "hl-n2");
        auto n3  = createConceptNode(space, "hl-n3");

        // Strengthen hub-n1 more than hub-n2 and hub-n3
        for (int i = 0; i < 8; ++i) learner.recordCoActivation(hub, n1);
        for (int i = 0; i < 4; ++i) learner.recordCoActivation(hub, n2);
        for (int i = 0; i < 2; ++i) learner.recordCoActivation(hub, n3);

        auto assoc = learner.getAssociates(hub);
        ASSERT(!assoc.empty());

        // Check sorted descending
        for (size_t i = 1; i < assoc.size(); ++i) {
            ASSERT(assoc[i - 1].second >= assoc[i].second);
        }
    END_TEST
}

void testLearnFromFocus() {
    TEST("learnFromAttentionalFocus creates links for focused atoms")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner learner(space, bank);

        auto a = createConceptNode(space, "focus-A");
        auto b = createConceptNode(space, "focus-B");

        bank.setAttentionValue(a, AttentionBank::AttentionValue(100.0f, 0.0f, 0.0f));
        bank.setAttentionValue(b, AttentionBank::AttentionValue(80.0f, 0.0f, 0.0f));

        learner.learnFromAttentionalFocus();

        // Should have created a link
        auto link = learner.getLink(a, b);
        ASSERT(link != nullptr);
    END_TEST
}

void testRunCycles() {
    TEST("runCycles performs multiple learn + decay cycles")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner::Config cfg;
        cfg.learningRate = 0.3f;
        cfg.decayRate = 0.01f;
        cfg.ojaRule = false;
        HebbianLearner learner(space, bank, cfg);

        auto a = createConceptNode(space, "cycle-A");
        auto b = createConceptNode(space, "cycle-B");
        bank.setAttentionValue(a, AttentionBank::AttentionValue(100.0f, 0.0f, 0.0f));
        bank.setAttentionValue(b, AttentionBank::AttentionValue(80.0f, 0.0f, 0.0f));

        learner.runCycles(5);

        // After 5 cycles with consistent focus, link should exist and be non-trivial
        float s = learner.getStrength(a, b);
        ASSERT(s > 0.0f);
    END_TEST
}

void testCoActivationCount() {
    TEST("coActivationCount tracks events")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner learner(space, bank);

        auto a = createConceptNode(space, "count-A");
        auto b = createConceptNode(space, "count-B");

        ASSERT(learner.coActivationCount(a, b) == 0);
        learner.recordCoActivation(a, b);
        learner.recordCoActivation(a, b);
        learner.recordCoActivation(a, b);
        ASSERT(learner.coActivationCount(a, b) == 3);
        ASSERT(learner.totalCoActivations() == 3);
    END_TEST
}

void testReset() {
    TEST("reset() zeros all link strengths and clears counters")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner::Config cfg;
        cfg.learningRate = 0.5f;
        cfg.ojaRule = false;
        HebbianLearner learner(space, bank, cfg);

        auto a = createConceptNode(space, "reset-A");
        auto b = createConceptNode(space, "reset-B");
        for (int i = 0; i < 5; ++i) learner.recordCoActivation(a, b);
        ASSERT(learner.getStrength(a, b) > 0.0f);

        learner.reset();
        ASSERT(learner.totalCoActivations() == 0);
        ASSERT(learner.getStrength(a, b) == 0.0f);
    END_TEST
}

void testAsymmetricLinks() {
    TEST("asymmetric config creates ASYMMETRIC_HEBBIAN_LINK")
        AtomSpace space;
        AttentionBank bank;
        HebbianLearner::Config cfg;
        cfg.asymmetric = true;
        cfg.ojaRule = false;
        HebbianLearner learner(space, bank, cfg);

        auto a = createConceptNode(space, "asym-A");
        auto b = createConceptNode(space, "asym-B");
        learner.recordCoActivation(a, b);

        auto link = learner.getLink(a, b);
        ASSERT(link != nullptr);
        ASSERT(link->getType() == Atom::Type::ASYMMETRIC_HEBBIAN_LINK);
    END_TEST
}

void testMakeHelperFactory() {
    TEST("makeHebbianLearner factory returns functional learner")
        AtomSpace space;
        AttentionBank bank;
        auto learner = makeHebbianLearner(space, bank, /*fast=*/true);

        auto a = createConceptNode(space, "factory-helper-A");
        auto b = createConceptNode(space, "factory-helper-B");
        learner.recordCoActivation(a, b);
        ASSERT(learner.getStrength(a, b) > 0.0f);
    END_TEST
}

// ======================================================================== //
//  Main
// ======================================================================== //

int main() {
    std::cout << "=== HebbianLearner Tests ===\n\n";

    testCoActivationCreatesLink();
    testStrengthIncreasesWithRepeatedActivation();
    testDecayReducesStrength();
    testPruningRemovesWeakLinks();
    testGetAssociates();
    testLearnFromFocus();
    testRunCycles();
    testCoActivationCount();
    testReset();
    testAsymmetricLinks();
    testMakeHelperFactory();

    std::cout << "\n--- Results ---\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    return (tests_failed == 0) ? 0 : 1;
}
