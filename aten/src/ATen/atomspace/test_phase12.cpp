/**
 * test_phase12.cpp - Tests for Phase 12 PLN completeness and pattern-matching
 *
 * Covers:
 *  - PLNConjunctionStep  (AND truth values, AND_LINK creation)
 *  - PLNDisjunctionStep  (OR truth values, OR_LINK creation)
 *  - PLNSimilarityStep   (SIMILARITY_LINK creation)
 *  - InferencePipeline fluent Phase 12 methods
 *  - makePLNFullPipeline factory helper
 *  - InferencePipeline::clear()
 *  - PatternMatcher negation-as-failure (NOT_LINK)
 */

#include "ATenSpaceCore.h"
#include "InferencePipeline.h"
#include "PatternMatcher.h"

#include <iostream>
#include <cassert>
#include <string>
#include <cmath>

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

#define ASSERT_NEAR(a, b, eps) \
    if (std::abs((a) - (b)) > (eps)) { \
        throw std::runtime_error( \
            std::string("ASSERT_NEAR failed: |") + std::to_string(a) + \
            " - " + std::to_string(b) + "| > " + std::to_string(eps)); \
    }

// ======================================================================== //
//  PLNConjunctionStep
// ======================================================================== //

void testConjunctionCreatesAndLink() {
    TEST("PLNConjunctionStep: creates AND_LINK for two high-strength atoms")
        AtomSpace space;
        auto A = createConceptNode(space, "A");
        auto B = createConceptNode(space, "B");
        A->setTruthValue(TruthValue::create(0.8f, 0.9f));
        B->setTruthValue(TruthValue::create(0.7f, 0.8f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNConjunctionStep step(0.5f);
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        bool foundAnd = false;
        for (const auto& a : ws) {
            if (a->isLink() && a->getType() == Atom::Type::AND_LINK) {
                foundAnd = true;
            }
        }
        ASSERT(foundAnd);
    END_TEST
}

void testConjunctionTV() {
    TEST("PLNConjunctionStep: AND_LINK TV strength = sA * sB")
        AtomSpace space;
        auto A = createConceptNode(space, "C");
        auto B = createConceptNode(space, "D");
        A->setTruthValue(TruthValue::create(0.8f, 0.9f));
        B->setTruthValue(TruthValue::create(0.5f, 0.8f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNConjunctionStep step(0.3f);
        step.execute(ws, space);

        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::AND_LINK) continue;
            float s = TruthValue::getStrength(a->getTruthValue());
            ASSERT_NEAR(s, 0.8f * 0.5f, 1e-4f);
        }
    END_TEST
}

void testConjunctionSkipsLowStrength() {
    TEST("PLNConjunctionStep: does not create AND_LINK when strength < threshold")
        AtomSpace space;
        auto A = createConceptNode(space, "E");
        auto B = createConceptNode(space, "F");
        // Both have low strength
        A->setTruthValue(TruthValue::create(0.2f, 0.9f));
        B->setTruthValue(TruthValue::create(0.3f, 0.8f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNConjunctionStep step(0.5f);  // threshold = 0.5
        bool changed = step.execute(ws, space);

        ASSERT(!changed);
        for (const auto& a : ws) {
            ASSERT(!a->isLink() || a->getType() != Atom::Type::AND_LINK);
        }
    END_TEST
}

void testConjunctionEvaluatesExistingAndLink() {
    TEST("PLNConjunctionStep: evaluates existing AND_LINK with unset TV")
        AtomSpace space;
        auto A = createConceptNode(space, "G");
        auto B = createConceptNode(space, "H");
        A->setTruthValue(TruthValue::create(0.9f, 0.8f));
        B->setTruthValue(TruthValue::create(0.6f, 0.7f));

        // Manually create an AND_LINK (TV will be default = s=1, c=0)
        auto andLink = space.addLink(Atom::Type::AND_LINK, {A, B});
        ASSERT(TruthValue::getConfidence(andLink->getTruthValue()) == 0.0f);

        std::vector<Atom::Handle> ws = {A, B, andLink};
        PLNConjunctionStep step(0.5f);
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        float s = TruthValue::getStrength(andLink->getTruthValue());
        ASSERT_NEAR(s, 0.9f * 0.6f, 1e-4f);
    END_TEST
}

// ======================================================================== //
//  PLNDisjunctionStep
// ======================================================================== //

void testDisjunctionCreatesOrLink() {
    TEST("PLNDisjunctionStep: creates OR_LINK for two qualifying atoms")
        AtomSpace space;
        auto A = createConceptNode(space, "P");
        auto B = createConceptNode(space, "Q");
        A->setTruthValue(TruthValue::create(0.6f, 0.8f));
        B->setTruthValue(TruthValue::create(0.5f, 0.7f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNDisjunctionStep step(0.3f);
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        bool foundOr = false;
        for (const auto& a : ws) {
            if (a->isLink() && a->getType() == Atom::Type::OR_LINK)
                foundOr = true;
        }
        ASSERT(foundOr);
    END_TEST
}

void testDisjunctionTV() {
    TEST("PLNDisjunctionStep: OR_LINK TV strength = sA + sB - sA*sB")
        AtomSpace space;
        auto A = createConceptNode(space, "R");
        auto B = createConceptNode(space, "S");
        float sA = 0.6f, sB = 0.5f;
        A->setTruthValue(TruthValue::create(sA, 0.8f));
        B->setTruthValue(TruthValue::create(sB, 0.7f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNDisjunctionStep step(0.3f);
        step.execute(ws, space);

        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::OR_LINK) continue;
            float s = TruthValue::getStrength(a->getTruthValue());
            float expected = sA + sB - sA * sB;
            ASSERT_NEAR(s, expected, 1e-4f);
        }
    END_TEST
}

void testDisjunctionEvaluatesExistingOrLink() {
    TEST("PLNDisjunctionStep: evaluates existing OR_LINK with unset TV")
        AtomSpace space;
        auto A = createConceptNode(space, "T");
        auto B = createConceptNode(space, "U");
        float sA = 0.7f, sB = 0.4f;
        A->setTruthValue(TruthValue::create(sA, 0.9f));
        B->setTruthValue(TruthValue::create(sB, 0.6f));

        auto orLink = space.addLink(Atom::Type::OR_LINK, {A, B});
        ASSERT(TruthValue::getConfidence(orLink->getTruthValue()) == 0.0f);

        std::vector<Atom::Handle> ws = {A, B, orLink};
        PLNDisjunctionStep step(0.3f);
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        float s = TruthValue::getStrength(orLink->getTruthValue());
        float expected = sA + sB - sA * sB;
        ASSERT_NEAR(s, expected, 1e-4f);
    END_TEST
}

// ======================================================================== //
//  PLNSimilarityStep
// ======================================================================== //

void testSimilarityCreatesLink() {
    TEST("PLNSimilarityStep: creates SIMILARITY_LINK for same-type atoms with high sim")
        AtomSpace space;
        auto A = createConceptNode(space, "X1");
        auto B = createConceptNode(space, "X2");
        // Both strong → similarity formula gives high value
        A->setTruthValue(TruthValue::create(0.9f, 0.8f));
        B->setTruthValue(TruthValue::create(0.8f, 0.7f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNSimilarityStep step(0.5f);
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        bool foundSim = false;
        for (const auto& a : ws) {
            if (a->isLink() && a->getType() == Atom::Type::SIMILARITY_LINK)
                foundSim = true;
        }
        ASSERT(foundSim);
    END_TEST
}

void testSimilarityTV() {
    TEST("PLNSimilarityStep: SIMILARITY_LINK TV uses PLN formula s = sA*sB/(sA+sB-sA*sB)")
        AtomSpace space;
        auto A = createConceptNode(space, "Y1");
        auto B = createConceptNode(space, "Y2");
        float sA = 0.8f, sB = 0.8f;
        A->setTruthValue(TruthValue::create(sA, 0.9f));
        B->setTruthValue(TruthValue::create(sB, 0.9f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNSimilarityStep step(0.3f);
        step.execute(ws, space);

        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::SIMILARITY_LINK) continue;
            float s = TruthValue::getStrength(a->getTruthValue());
            float expected = (sA * sB) / (sA + sB - sA * sB + kPLNSimEps);
            ASSERT_NEAR(s, expected, 1e-3f);
        }
    END_TEST
}

void testSimilaritySkipsDifferentTypes() {
    TEST("PLNSimilarityStep: does not pair atoms of different types")
        AtomSpace space;
        auto A = createConceptNode(space, "Z1");
        // Create an InheritanceLink as the second atom (different type)
        auto C = createConceptNode(space, "Z2");
        auto link = space.addLink(Atom::Type::INHERITANCE_LINK, {A, C});
        link->setTruthValue(TruthValue::create(0.9f, 0.9f));
        A->setTruthValue(TruthValue::create(0.9f, 0.9f));

        std::vector<Atom::Handle> ws = {A, link};
        PLNSimilarityStep step(0.1f);
        bool changed = step.execute(ws, space);

        // No SIMILARITY_LINK should be created because types differ
        ASSERT(!changed);
    END_TEST
}

void testSimilaritySkipsBelowThreshold() {
    TEST("PLNSimilarityStep: low-similarity pairs are skipped")
        AtomSpace space;
        auto A = createConceptNode(space, "W1");
        auto B = createConceptNode(space, "W2");
        // Very different strengths → low PLN similarity
        A->setTruthValue(TruthValue::create(0.01f, 0.9f));
        B->setTruthValue(TruthValue::create(0.99f, 0.9f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNSimilarityStep step(0.9f);  // high threshold
        bool changed = step.execute(ws, space);

        ASSERT(!changed);
    END_TEST
}

// ======================================================================== //
//  InferencePipeline - Phase 12 fluent API
// ======================================================================== //

void testPipelineConjunctionMethod() {
    TEST("InferencePipeline::plnConjunction appends PLNConjunctionStep")
        AtomSpace space;
        InferencePipeline p(space);
        p.plnConjunction(0.4f);
        ASSERT(p.size() == 1);
        ASSERT(p.stepNames()[0] == "PLNConjunction");
    END_TEST
}

void testPipelineDisjunctionMethod() {
    TEST("InferencePipeline::plnDisjunction appends PLNDisjunctionStep")
        AtomSpace space;
        InferencePipeline p(space);
        p.plnDisjunction(0.3f);
        ASSERT(p.size() == 1);
        ASSERT(p.stepNames()[0] == "PLNDisjunction");
    END_TEST
}

void testPipelineSimilarityMethod() {
    TEST("InferencePipeline::plnSimilarity appends PLNSimilarityStep")
        AtomSpace space;
        InferencePipeline p(space);
        p.plnSimilarity(0.5f);
        ASSERT(p.size() == 1);
        ASSERT(p.stepNames()[0] == "PLNSimilarity");
    END_TEST
}

void testPipelineClear() {
    TEST("InferencePipeline::clear removes all steps")
        AtomSpace space;
        InferencePipeline p(space);
        p.plnDeduction().plnRevision().plnConjunction();
        ASSERT(p.size() == 3);
        p.clear();
        ASSERT(p.size() == 0);
    END_TEST
}

void testMakePLNFullPipeline() {
    TEST("makePLNFullPipeline creates 6-step pipeline")
        AtomSpace space;
        auto p = makePLNFullPipeline(space);
        ASSERT(p.size() == 6);
        auto names = p.stepNames();
        ASSERT(names[0] == "PLNDeduction");
        ASSERT(names[1] == "PLNConjunction");
        ASSERT(names[2] == "PLNDisjunction");
        ASSERT(names[3] == "PLNSimilarity");
        ASSERT(names[4] == "PLNRevision");
        ASSERT(names[5] == "TVThreshold");
    END_TEST
}

void testFullPipelineEndToEnd() {
    TEST("makePLNFullPipeline end-to-end: derives AND/OR/SIM links and deductions")
        AtomSpace space;
        auto A = createConceptNode(space, "Alpha");
        auto B = createConceptNode(space, "Beta");
        auto C = createConceptNode(space, "Gamma");
        A->setTruthValue(TruthValue::create(0.9f, 0.8f));
        B->setTruthValue(TruthValue::create(0.8f, 0.7f));
        C->setTruthValue(TruthValue::create(0.7f, 0.6f));

        auto ab = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::INHERITANCE_LINK, {B, C});
        ab->setTruthValue(TruthValue::create(0.85f, 0.75f));
        bc->setTruthValue(TruthValue::create(0.75f, 0.65f));

        auto p = makePLNFullPipeline(space, 0.0f, 0.0f, 0.5f, 0.3f);
        auto result = p.run({A, B, C, ab, bc});

        ASSERT(result.atoms.size() > 5);
        ASSERT(result.iterationsRun == 1);
    END_TEST
}

// ======================================================================== //
//  PatternMatcher - negation-as-failure
// ======================================================================== //

void testNegationMatchesNonMatching() {
    TEST("PatternMatcher: NOT_LINK(ConceptA) matches ConceptB (different name)")
        AtomSpace space;
        auto A = createConceptNode(space, "NegA");
        auto B = createConceptNode(space, "NegB");

        // Build NOT_LINK pattern wrapping the concrete atom A
        auto notPattern = space.addLink(Atom::Type::NOT_LINK, {A});

        // B doesn't match A → negation succeeds
        VariableBinding bindings;
        bool ok = PatternMatcher::match(notPattern, B, bindings);
        ASSERT(ok);
    END_TEST
}

void testNegationFailsOnMatch() {
    TEST("PatternMatcher: NOT_LINK(ConceptA) does not match ConceptA itself")
        AtomSpace space;
        auto A  = createConceptNode(space, "NegX");
        auto A2 = createConceptNode(space, "NegX");  // same name → same atom

        auto notPattern = space.addLink(Atom::Type::NOT_LINK, {A});

        // A2 structurally equals A → inner match succeeds → negation fails
        VariableBinding bindings;
        bool ok = PatternMatcher::match(notPattern, A2, bindings);
        ASSERT(!ok);
    END_TEST
}

void testNegationWithVariable() {
    TEST("PatternMatcher: NOT_LINK(variable) never matches anything (variable binds all)")
        AtomSpace space;
        auto varX = createVariableNode(space, "?X");
        auto B    = createConceptNode(space, "NegV");

        auto notPattern = space.addLink(Atom::Type::NOT_LINK, {varX});

        // Variable binds B → inner match succeeds → negation fails
        VariableBinding bindings;
        bool ok = PatternMatcher::match(notPattern, B, bindings);
        ASSERT(!ok);
    END_TEST
}

void testNegationFindMatchesFilters() {
    TEST("PatternMatcher::findMatches with NOT_LINK filters out matching atoms")
        AtomSpace space;
        auto cat = createConceptNode(space, "Cat");
        auto dog = createConceptNode(space, "Dog");
        auto fish = createConceptNode(space, "Fish");

        // NOT_LINK(Cat) matches Dog and Fish, not Cat
        auto notCat = space.addLink(Atom::Type::NOT_LINK, {cat});

        auto matches = PatternMatcher::findMatches(space, notCat);
        // The AtomSpace also contains the NOT_LINK itself and cat/dog/fish.
        // We check that "Cat" is not in the results and Dog/Fish are.
        bool foundCat = false;
        bool foundDog = false;
        bool foundFish = false;
        for (const auto& [atom, b] : matches) {
            if (!atom->isNode()) continue;
            const Node* n = static_cast<const Node*>(atom.get());
            if (n->getName() == "Cat")  foundCat  = true;
            if (n->getName() == "Dog")  foundDog  = true;
            if (n->getName() == "Fish") foundFish = true;
        }
        ASSERT(!foundCat);
        ASSERT(foundDog);
        ASSERT(foundFish);
    END_TEST
}

// ======================================================================== //
//  Main
// ======================================================================== //

int main() {
    std::cout << "\n=== Phase 12 Tests ===\n\n";

    std::cout << "-- PLNConjunctionStep --\n";
    testConjunctionCreatesAndLink();
    testConjunctionTV();
    testConjunctionSkipsLowStrength();
    testConjunctionEvaluatesExistingAndLink();

    std::cout << "\n-- PLNDisjunctionStep --\n";
    testDisjunctionCreatesOrLink();
    testDisjunctionTV();
    testDisjunctionEvaluatesExistingOrLink();

    std::cout << "\n-- PLNSimilarityStep --\n";
    testSimilarityCreatesLink();
    testSimilarityTV();
    testSimilaritySkipsDifferentTypes();
    testSimilaritySkipsBelowThreshold();

    std::cout << "\n-- InferencePipeline Phase 12 --\n";
    testPipelineConjunctionMethod();
    testPipelineDisjunctionMethod();
    testPipelineSimilarityMethod();
    testPipelineClear();
    testMakePLNFullPipeline();
    testFullPipelineEndToEnd();

    std::cout << "\n-- PatternMatcher negation-as-failure --\n";
    testNegationMatchesNonMatching();
    testNegationFailsOnMatch();
    testNegationWithVariable();
    testNegationFindMatchesFilters();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n\n";

    return tests_failed > 0 ? 1 : 0;
}
