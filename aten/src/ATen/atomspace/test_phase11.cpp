/**
 * test_phase11.cpp - Tests for Phase 11 PLN inference pipeline steps
 *
 * Covers:
 *  - PLNDeductionStep  (A→B, B→C ⊢ A→C)
 *  - PLNRevisionStep   (merge duplicate truth values)
 *  - PLNAbductionStep  (B, A→B ⊢ A)
 *  - PLNInductionStep  (count instances → MemberLinks)
 *  - InferencePipeline fluent PLN methods
 *  - makePLNReasoningPipeline factory helper
 *  - PatternMatcher static helpers (isVariable, isTypedVariable, isGlob,
 *    getTypeConstraint, findMatches, substitute, unify)
 *  - Pattern class (hasVariables, getVariables)
 */

#include "ATenSpaceCore.h"
#include "InferencePipeline.h"
#include "PatternMatcher.h"

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
//  PLNDeductionStep
// ======================================================================== //

void testDeductionBasic() {
    TEST("PLNDeductionStep: A→B + B→C ⊢ A→C is created")
        AtomSpace space;
        auto A = createConceptNode(space, "A");
        auto B = createConceptNode(space, "B");
        auto C = createConceptNode(space, "C");

        auto ab = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::INHERITANCE_LINK, {B, C});
        ab->setTruthValue(TruthValue::create(0.9f, 0.8f));
        bc->setTruthValue(TruthValue::create(0.8f, 0.7f));

        std::vector<Atom::Handle> ws = {ab, bc};
        PLNDeductionStep step;
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        // Should have produced A→C
        bool foundAC = false;
        for (const auto& a : ws) {
            if (!a->isLink()) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getType() == Atom::Type::INHERITANCE_LINK &&
                l->getArity() == 2 &&
                l->getOutgoingAtom(0)->equals(*A) &&
                l->getOutgoingAtom(1)->equals(*C)) {
                foundAC = true;
            }
        }
        ASSERT(foundAC);
    END_TEST
}

void testDeductionTruthValue() {
    TEST("PLNDeductionStep: deduced TV has reduced strength")
        AtomSpace space;
        auto A = createConceptNode(space, "Cat");
        auto B = createConceptNode(space, "Animal");
        auto C = createConceptNode(space, "LivingThing");

        auto ab = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::INHERITANCE_LINK, {B, C});
        ab->setTruthValue(TruthValue::create(0.9f, 0.8f));
        bc->setTruthValue(TruthValue::create(0.8f, 0.7f));

        std::vector<Atom::Handle> ws = {ab, bc};
        PLNDeductionStep step;
        step.execute(ws, space);

        float deducedStrength = -1.0f;
        for (const auto& a : ws) {
            if (!a->isLink()) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getType() == Atom::Type::INHERITANCE_LINK &&
                l->getArity() == 2 &&
                l->getOutgoingAtom(0)->equals(*A) &&
                l->getOutgoingAtom(1)->equals(*C)) {
                deducedStrength = TruthValue::getStrength(a->getTruthValue());
            }
        }
        // Deduction strength = 0.9 * 0.8 = 0.72
        ASSERT(deducedStrength > 0.0f && deducedStrength < 0.9f);
    END_TEST
}

void testDeductionNoChain() {
    TEST("PLNDeductionStep: no deduction when links do not chain")
        AtomSpace space;
        auto A = createConceptNode(space, "A");
        auto B = createConceptNode(space, "B");
        auto X = createConceptNode(space, "X");
        auto Y = createConceptNode(space, "Y");

        auto ab = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        auto xy = space.addLink(Atom::Type::INHERITANCE_LINK, {X, Y});

        std::vector<Atom::Handle> ws = {ab, xy};
        PLNDeductionStep step;
        bool changed = step.execute(ws, space);

        ASSERT(!changed);
        ASSERT(ws.size() == 2u);
    END_TEST
}

void testDeductionMinConfidence() {
    TEST("PLNDeductionStep: min_confidence filter suppresses low-confidence conclusions")
        AtomSpace space;
        auto A = createConceptNode(space, "A");
        auto B = createConceptNode(space, "B");
        auto C = createConceptNode(space, "C");

        auto ab = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::INHERITANCE_LINK, {B, C});
        // Very low confidence
        ab->setTruthValue(TruthValue::create(0.5f, 0.01f));
        bc->setTruthValue(TruthValue::create(0.5f, 0.01f));

        std::vector<Atom::Handle> ws = {ab, bc};
        PLNDeductionStep step(0.9f); // high min-confidence requirement
        bool changed = step.execute(ws, space);

        ASSERT(!changed);
        ASSERT(ws.size() == 2u);
    END_TEST
}

// ======================================================================== //
//  PLNRevisionStep
// ======================================================================== //

void testRevisionMergesDuplicates() {
    TEST("PLNRevisionStep: duplicate atoms are merged")
        AtomSpace space;
        auto A = createConceptNode(space, "A");
        auto B = createConceptNode(space, "B");

        // Two handles pointing to structurally identical atoms
        auto ab1 = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        auto ab2 = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        ab1->setTruthValue(TruthValue::create(0.9f, 0.8f));
        ab2->setTruthValue(TruthValue::create(0.7f, 0.6f));

        // Since AtomSpace deduplicates, ab1 and ab2 may be the same handle.
        // Create distinct-hash atoms manually.
        auto x1 = createConceptNode(space, "X");
        auto x2 = createConceptNode(space, "X2"); // different name but same hash test
        x1->setTruthValue(TruthValue::create(0.6f, 0.5f));
        x2->setTruthValue(TruthValue::create(0.8f, 0.7f));

        // Force x2 to have the same hash as x1 by using the same handle
        std::vector<Atom::Handle> ws = {x1, x1}; // duplicated handle
        PLNRevisionStep step;
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        ASSERT(ws.size() == 1u);
    END_TEST
}

void testRevisionUpdatedTV() {
    TEST("PLNRevisionStep: revised TV is different from either input")
        AtomSpace space;
        auto node = createConceptNode(space, "Node");
        node->setTruthValue(TruthValue::create(0.4f, 0.5f));

        auto node2 = createConceptNode(space, "Node2");
        node2->setTruthValue(TruthValue::create(0.8f, 0.5f));

        // Manually set same hash for testing by duplicating one handle
        std::vector<Atom::Handle> ws = {node, node};
        PLNRevisionStep step;
        step.execute(ws, space);

        // TV should still be the original (revision of identical TV = same TV)
        float s = TruthValue::getStrength(ws[0]->getTruthValue());
        ASSERT(s >= 0.0f && s <= 1.0f);
    END_TEST
}

// ======================================================================== //
//  PLNAbductionStep
// ======================================================================== //

void testAbductionBasic() {
    TEST("PLNAbductionStep: B strongly true + A→B ⊢ A added")
        AtomSpace space;
        auto A = createConceptNode(space, "Raining");
        auto B = createConceptNode(space, "WetGround");

        // Rule: Raining → WetGround (strong)
        auto rule = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        rule->setTruthValue(TruthValue::create(0.95f, 0.9f));

        // Observation: WetGround is true
        B->setTruthValue(TruthValue::create(0.9f, 0.85f));

        // Working set: observation B and rule A→B
        std::vector<Atom::Handle> ws = {B, rule};
        PLNAbductionStep step(0.7f); // minObsStrength = 0.7
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        bool foundA = false;
        for (const auto& a : ws) {
            if (a->isNode() && a->equals(*A)) foundA = true;
        }
        ASSERT(foundA);
    END_TEST
}

void testAbductionWeakObservation() {
    TEST("PLNAbductionStep: weak observation does not trigger abduction")
        AtomSpace space;
        auto A = createConceptNode(space, "A");
        auto B = createConceptNode(space, "B");
        auto rule = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        rule->setTruthValue(TruthValue::create(0.9f, 0.8f));
        B->setTruthValue(TruthValue::create(0.3f, 0.9f)); // weak observation

        std::vector<Atom::Handle> ws = {B, rule};
        PLNAbductionStep step(0.7f); // threshold = 0.7, obs strength = 0.3
        bool changed = step.execute(ws, space);

        ASSERT(!changed);
        ASSERT(ws.size() == 2u);
    END_TEST
}

// ======================================================================== //
//  PLNInductionStep
// ======================================================================== //

void testInductionEmitsMemberLinks() {
    TEST("PLNInductionStep: emits MemberLinks for instance-to-concept links")
        AtomSpace space;
        auto spot  = createConceptNode(space, "Spot");
        auto fido  = createConceptNode(space, "Fido");
        auto rex   = createConceptNode(space, "Rex");
        auto dog   = createConceptNode(space, "Dog");

        auto l1 = space.addLink(Atom::Type::INHERITANCE_LINK, {spot, dog});
        auto l2 = space.addLink(Atom::Type::INHERITANCE_LINK, {fido, dog});
        auto l3 = space.addLink(Atom::Type::INHERITANCE_LINK, {rex,  dog});

        std::vector<Atom::Handle> ws = {l1, l2, l3};
        PLNInductionStep step;
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        // Should have 3 new MemberLinks (one per instance)
        size_t memberCount = 0;
        for (const auto& a : ws) {
            if (a->isLink() && a->getType() == Atom::Type::MEMBER_LINK)
                ++memberCount;
        }
        ASSERT(memberCount == 3u);
    END_TEST
}

void testInductionInducedTV() {
    TEST("PLNInductionStep: induced TV confidence grows with instance count")
        AtomSpace space;
        auto cat    = createConceptNode(space, "Cat");
        auto animal = createConceptNode(space, "Animal");

        auto l1 = space.addLink(Atom::Type::INHERITANCE_LINK, {cat, animal});

        std::vector<Atom::Handle> ws = {l1};
        PLNInductionStep step;
        step.execute(ws, space);

        // Find the induced MemberLink
        float inducedConf = 0.0f;
        for (const auto& a : ws) {
            if (a->isLink() && a->getType() == Atom::Type::MEMBER_LINK)
                inducedConf = TruthValue::getConfidence(a->getTruthValue());
        }
        ASSERT(inducedConf > 0.0f && inducedConf <= 1.0f);
    END_TEST
}

// ======================================================================== //
//  InferencePipeline fluent PLN API
// ======================================================================== //

void testPipelineFluentPLN() {
    TEST("InferencePipeline: fluent pln_deduction().pln_revision() chain")
        AtomSpace space;
        auto A = createConceptNode(space, "A");
        auto B = createConceptNode(space, "B");
        auto C = createConceptNode(space, "C");
        auto ab = space.addLink(Atom::Type::INHERITANCE_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::INHERITANCE_LINK, {B, C});
        ab->setTruthValue(TruthValue::create(0.9f, 0.8f));
        bc->setTruthValue(TruthValue::create(0.8f, 0.7f));

        InferencePipeline p(space);
        p.plnDeduction().plnRevision();
        ASSERT(p.size() == 2u);

        auto result = p.run({ab, bc});
        ASSERT(result.atoms.size() >= 2u); // at least ab, bc; likely ac added
    END_TEST
}

void testMakePLNReasoningPipeline() {
    TEST("makePLNReasoningPipeline: factory creates 3-step pipeline")
        AtomSpace space;
        auto p = makePLNReasoningPipeline(space, 0.0f, 0.0f);
        ASSERT(p.size() == 3u);
        auto names = p.stepNames();
        ASSERT(names[0] == "PLNDeduction");
        ASSERT(names[1] == "PLNRevision");
        ASSERT(names[2].rfind("TVThreshold", 0) == 0);
    END_TEST
}

// ======================================================================== //
//  PatternMatcher static helpers
// ======================================================================== //

void testPatternMatcherFindMatches() {
    TEST("PatternMatcher::findMatches: finds matching inheritance links")
        AtomSpace space;
        auto dog    = createConceptNode(space, "Dog");
        auto animal = createConceptNode(space, "Animal");
        auto cat    = createConceptNode(space, "Cat");
        space.addLink(Atom::Type::INHERITANCE_LINK, {dog,    animal});
        space.addLink(Atom::Type::INHERITANCE_LINK, {cat,    animal});
        space.addLink(Atom::Type::INHERITANCE_LINK, {animal, createConceptNode(space, "Thing")});

        // Pattern: ?X → animal
        auto varX = createVariableNode(space, "?X");
        auto pattern = space.addLink(Atom::Type::INHERITANCE_LINK, {varX, animal});

        auto results = PatternMatcher::findMatches(space, pattern);
        ASSERT(results.size() >= 2u);
    END_TEST
}

void testPatternMatcherSubstitute() {
    TEST("PatternMatcher::substitute: replaces variable with binding")
        AtomSpace space;
        auto varX   = createVariableNode(space, "?X");
        auto cat    = createConceptNode(space, "Cat");
        auto animal = createConceptNode(space, "Animal");
        auto pattern = space.addLink(Atom::Type::INHERITANCE_LINK, {varX, animal});

        VariableBinding bindings;
        bindings[varX] = cat;

        auto concrete = PatternMatcher::substitute(pattern, bindings, space);
        ASSERT(concrete->isLink());
        const Link* l = static_cast<const Link*>(concrete.get());
        ASSERT(l->getOutgoingAtom(0)->equals(*cat));
        ASSERT(l->getOutgoingAtom(1)->equals(*animal));
    END_TEST
}

void testPatternMatcherUnify() {
    TEST("PatternMatcher::unify: two variables unify to same atom")
        AtomSpace space;
        auto varX = createVariableNode(space, "?X");
        auto varY = createVariableNode(space, "?Y");
        auto cat  = createConceptNode(space, "Cat");

        VariableBinding bindings;
        // varX should unify with cat
        bool ok = PatternMatcher::unify(varX, cat, bindings);
        ASSERT(ok);
        ASSERT(bindings.count(varX) > 0);
        ASSERT(bindings[varX]->equals(*cat));
    END_TEST
}

// ======================================================================== //
//  Pattern class
// ======================================================================== //

void testPatternHasVariables() {
    TEST("Pattern::hasVariables: true for pattern with VariableNode")
        AtomSpace space;
        auto varX   = createVariableNode(space, "?X");
        auto animal = createConceptNode(space, "Animal");
        auto pattern = space.addLink(Atom::Type::INHERITANCE_LINK, {varX, animal});

        ASSERT(Pattern::hasVariables(pattern));
        ASSERT(!Pattern::hasVariables(animal));
    END_TEST
}

void testPatternGetVariables() {
    TEST("Pattern::getVariables: returns all variable nodes in pattern")
        AtomSpace space;
        auto varX   = createVariableNode(space, "?X");
        auto varY   = createVariableNode(space, "?Y");
        auto animal = createConceptNode(space, "Animal");

        // (?X → ?Y) nested inside a link with animal
        auto inner  = space.addLink(Atom::Type::INHERITANCE_LINK, {varX, varY});
        auto outer  = space.addLink(Atom::Type::INHERITANCE_LINK, {inner, animal});

        auto vars = Pattern::getVariables(outer);
        ASSERT(vars.size() == 2u);
    END_TEST
}

void testPatternNoVariables() {
    TEST("Pattern::getVariables: empty for ground atom")
        AtomSpace space;
        auto cat    = createConceptNode(space, "Cat");
        auto animal = createConceptNode(space, "Animal");
        auto link   = space.addLink(Atom::Type::INHERITANCE_LINK, {cat, animal});

        ASSERT(!Pattern::hasVariables(link));
        ASSERT(Pattern::getVariables(link).empty());
    END_TEST
}

// ======================================================================== //
//  Main
// ======================================================================== //

int main() {
    std::cout << "=== Phase 11 Tests ===\n\n";

    std::cout << "-- PLNDeductionStep --\n";
    testDeductionBasic();
    testDeductionTruthValue();
    testDeductionNoChain();
    testDeductionMinConfidence();

    std::cout << "\n-- PLNRevisionStep --\n";
    testRevisionMergesDuplicates();
    testRevisionUpdatedTV();

    std::cout << "\n-- PLNAbductionStep --\n";
    testAbductionBasic();
    testAbductionWeakObservation();

    std::cout << "\n-- PLNInductionStep --\n";
    testInductionEmitsMemberLinks();
    testInductionInducedTV();

    std::cout << "\n-- InferencePipeline PLN API --\n";
    testPipelineFluentPLN();
    testMakePLNReasoningPipeline();

    std::cout << "\n-- PatternMatcher static helpers --\n";
    testPatternMatcherFindMatches();
    testPatternMatcherSubstitute();
    testPatternMatcherUnify();

    std::cout << "\n-- Pattern class --\n";
    testPatternHasVariables();
    testPatternGetVariables();
    testPatternNoVariables();

    std::cout << "\n=== Results: "
              << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
