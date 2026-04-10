/**
 * test_inference_pipeline.cpp - Tests for InferencePipeline and steps
 */
#include "ATenSpaceCore.h"
#include "InferencePipeline.h"
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

void testEmptyPipeline() {
    TEST("empty pipeline returns seeds unchanged")
        AtomSpace space;
        auto a = createConceptNode(space, "empty-pipeline-A");
        auto b = createConceptNode(space, "empty-pipeline-B");

        InferencePipeline p(space);
        ASSERT(p.size() == 0);

        std::vector<Atom::Handle> seeds = {a, b};
        auto result = p.run(seeds);

        ASSERT(result.atoms.size() == 2);
        ASSERT(result.iterationsRun == 0);
    END_TEST
}

void testFilterStep() {
    TEST("FilterStep removes non-matching atoms")
        AtomSpace space;
        auto concept1 = createConceptNode(space, "filter-concept");
        auto pred1    = createPredicateNode(space, "filter-predicate");

        InferencePipeline p(space);
        p.filter("only-concepts",
                 [](const Atom::Handle& a) {
                     return a->getType() == Atom::Type::CONCEPT_NODE;
                 });

        std::vector<Atom::Handle> seeds = {concept1, pred1};
        auto result = p.run(seeds);

        ASSERT(result.atoms.size() == 1);
        ASSERT(result.atoms[0]->getType() == Atom::Type::CONCEPT_NODE);
    END_TEST
}

void testTVThresholdStep() {
    TEST("TruthValueThresholdStep filters by strength")
        AtomSpace space;
        auto high   = createConceptNode(space, "tv-high");
        auto low    = createConceptNode(space, "tv-low");
        auto medium = createConceptNode(space, "tv-medium");

        high->setTruthValue(TruthValue::create(0.9f, 0.8f));
        low->setTruthValue(TruthValue::create(0.1f, 0.8f));
        medium->setTruthValue(TruthValue::create(0.5f, 0.8f));

        InferencePipeline p(space);
        p.filterByTV(0.4f);

        std::vector<Atom::Handle> seeds = {high, low, medium};
        auto result = p.run(seeds);

        ASSERT(result.atoms.size() == 2);  // high and medium pass
    END_TEST
}

void testPatternMatchStep() {
    TEST("PatternMatchStep expands working set")
        AtomSpace space;
        auto mammal = createConceptNode(space, "pm-mammal");
        auto dog    = createConceptNode(space, "pm-dog");
        auto cat    = createConceptNode(space, "pm-cat");
        createInheritanceLink(space, dog, mammal);
        createInheritanceLink(space, cat, mammal);

        // Pattern: InheritanceLink(?X, mammal)
        auto varX   = space.addNode(Atom::Type::VARIABLE_NODE, "?X-pm");
        auto pat    = createInheritanceLink(space, varX, mammal);

        InferencePipeline p(space);
        p.matchPattern(pat);

        // Start with empty seeds (should default to all atoms)
        auto result = p.run();
        // Should include the matched link atoms in the working set
        ASSERT(!result.atoms.empty());
    END_TEST
}

void testCustomStep() {
    TEST("CustomStep callable executes correctly")
        AtomSpace space;
        auto a = createConceptNode(space, "custom-A");

        bool stepCalled = false;
        InferencePipeline p(space);
        p.addCustomStep("mark-called",
            [&stepCalled](std::vector<Atom::Handle>& ws, AtomSpace&) {
                stepCalled = true;
                return false;
            });

        p.run({a});
        ASSERT(stepCalled);
    END_TEST
}

void testPipelineStats() {
    TEST("PipelineResult records per-step stats")
        AtomSpace space;
        auto a = createConceptNode(space, "stats-A");

        InferencePipeline p(space);
        p.filter("pass-all", [](const Atom::Handle&){ return true; });
        p.filter("pass-all-2", [](const Atom::Handle&){ return true; });

        auto result = p.run({a});
        ASSERT(result.stats.size() == 2);
        ASSERT(result.stats[0].stepName == "Filter[pass-all]");
        ASSERT(result.stats[1].stepName == "Filter[pass-all-2]");

        // All steps should have measured time >= 0
        for (const auto& s : result.stats) {
            ASSERT(s.elapsedMs >= 0.0);
        }
    END_TEST
}

void testFixedPointIteration() {
    TEST("fixed-point iteration runs until no change")
        AtomSpace space;
        auto a = createConceptNode(space, "fp-A");
        auto b = createConceptNode(space, "fp-B");

        // A step that always reports "no change"
        int invocations = 0;
        InferencePipeline p(space);
        p.addCustomStep("no-change",
            [&invocations](std::vector<Atom::Handle>&, AtomSpace&) {
                ++invocations;
                return false;  // No change → fixed point immediately
            });

        auto result = p.run({a, b}, /*untilFixedPoint=*/true, /*maxIter=*/100);
        // Should converge after 1 iteration since the step reported no change
        ASSERT(result.converged);
        ASSERT(result.iterationsRun == 1);
        ASSERT(invocations == 1);
    END_TEST
}

void testAttentionBoostStep() {
    TEST("AttentionBoostStep raises STI on working set atoms")
        AtomSpace space;
        AttentionBank bank;
        auto a = createConceptNode(space, "boost-A");
        bank.setAttentionValue(a, AttentionBank::AttentionValue(0.0f, 0.0f, 0.0f));

        InferencePipeline p(space);
        p.boostAttention(bank, 25.0f);

        p.run({a});
        auto av = bank.getAttentionValue(a);
        ASSERT(av.sti > 0.0f);
    END_TEST
}

void testStepNames() {
    TEST("stepNames() returns correct names")
        AtomSpace space;
        InferencePipeline p(space);
        p.filterByTV(0.5f);
        p.filter("my-filter", [](const Atom::Handle&){ return true; });

        auto names = p.stepNames();
        ASSERT(names.size() == 2);
        ASSERT(names[0].find("TVThreshold") != std::string::npos);
        ASSERT(names[1] == "Filter[my-filter]");
    END_TEST
}

void testForwardReasoningFactory() {
    TEST("makeForwardReasoningPipeline factory builds non-empty pipeline")
        AtomSpace space;
        auto varX = space.addNode(Atom::Type::VARIABLE_NODE, "?factory-X");
        auto mammal = createConceptNode(space, "factory-mammal");
        auto pat    = createInheritanceLink(space, varX, mammal);

        auto pipeline = makeForwardReasoningPipeline(space, pat, 0.5f, 1);
        ASSERT(pipeline.size() >= 2);  // at least PatternMatch + TVFilter
    END_TEST
}

// ======================================================================== //
//  Main
// ======================================================================== //

int main() {
    std::cout << "=== InferencePipeline Tests ===\n\n";

    testEmptyPipeline();
    testFilterStep();
    testTVThresholdStep();
    testPatternMatchStep();
    testCustomStep();
    testPipelineStats();
    testFixedPointIteration();
    testAttentionBoostStep();
    testStepNames();
    testForwardReasoningFactory();

    std::cout << "\n--- Results ---\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    return (tests_failed == 0) ? 0 : 1;
}
