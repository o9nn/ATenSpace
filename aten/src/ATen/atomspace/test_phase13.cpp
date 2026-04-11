/**
 * test_phase13.cpp - Tests for Phase 13 PLN Implication completeness
 *
 * Covers:
 *  - PLNImplicationStep  (evaluate existing IMPLICATION_LINKs, create new ones)
 *  - PLNImplicationChainStep (multi-hop transitive closure)
 *  - InferencePipeline fluent Phase 13 methods
 *  - makePLNCompletePipeline factory helper
 */

#include "ATenSpaceCore.h"
#include "InferencePipeline.h"

#include <iostream>
#include <cassert>
#include <string>
#include <cmath>

using namespace at::atomspace;

static int tests_passed = 0;
static int tests_failed  = 0;

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
//  PLNImplicationStep                                                        //
// ======================================================================== //

void testImplicationCreatesLink() {
    TEST("PLNImplicationStep: creates IMPLICATION_LINK for two qualifying atoms")
        AtomSpace space;
        auto A = createConceptNode(space, "ImpA1");
        auto B = createConceptNode(space, "ImpB1");
        A->setTruthValue(TruthValue::create(0.8f, 0.9f));
        B->setTruthValue(TruthValue::create(0.7f, 0.8f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNImplicationStep step(0.5f, 0.3f);   // low imp-strength threshold
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        bool foundImp = false;
        for (const auto& a : ws) {
            if (a->isLink() && a->getType() == Atom::Type::IMPLICATION_LINK)
                foundImp = true;
        }
        ASSERT(foundImp);
    END_TEST
}

void testImplicationTV() {
    TEST("PLNImplicationStep: IMPLICATION_LINK TV uses PLN formula s=1-sA+sA*sB")
        AtomSpace space;
        float sA = 0.7f, sB = 0.6f;
        auto A = createConceptNode(space, "ImpA2");
        auto B = createConceptNode(space, "ImpB2");
        A->setTruthValue(TruthValue::create(sA, 0.9f));
        B->setTruthValue(TruthValue::create(sB, 0.8f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNImplicationStep step(0.5f, 0.0f);  // min-imp-strength = 0: accept all
        step.execute(ws, space);

        float expectedStr = 1.0f - sA + sA * sB;
        float expectedConf = std::min(0.9f, 0.8f);

        bool checked = false;
        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            // Check the A→B direction
            if (l->getOutgoingAtom(0)->getHash() != A->getHash()) continue;
            float s = TruthValue::getStrength(a->getTruthValue());
            float c = TruthValue::getConfidence(a->getTruthValue());
            ASSERT_NEAR(s, expectedStr, 1e-4f);
            ASSERT_NEAR(c, expectedConf, 1e-4f);
            checked = true;
        }
        ASSERT(checked);
    END_TEST
}

void testImplicationSkipsLowStrengthAntecedent() {
    TEST("PLNImplicationStep: no link when antecedent strength < minAntecedentStrength")
        AtomSpace space;
        auto A = createConceptNode(space, "ImpA3");
        auto B = createConceptNode(space, "ImpB3");
        A->setTruthValue(TruthValue::create(0.2f, 0.9f));  // low strength
        B->setTruthValue(TruthValue::create(0.8f, 0.9f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNImplicationStep step(0.5f, 0.0f);  // minAntStr = 0.5
        bool changed = step.execute(ws, space);

        ASSERT(!changed);
    END_TEST
}

void testImplicationSkipsBelowImpStrengthThreshold() {
    TEST("PLNImplicationStep: no link when implication strength < minImplicationStrength")
        AtomSpace space;
        // s(A→B) = 1 - sA + sA*sB  →  small sA, small sB: ≈ 1 - sA + sA*sB
        // With sA=0.8, sB=0.1: s = 1 - 0.8 + 0.08 = 0.28 < threshold 0.5
        float sA = 0.8f, sB = 0.1f;
        auto A = createConceptNode(space, "ImpA4");
        auto B = createConceptNode(space, "ImpB4");
        A->setTruthValue(TruthValue::create(sA, 0.9f));
        B->setTruthValue(TruthValue::create(sB, 0.9f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNImplicationStep step(0.5f, 0.5f);  // high imp-strength threshold
        bool changed = step.execute(ws, space);

        // Check that no A→B link was added (B→A may pass: s=1-0.1+0.08=0.98)
        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            // A→B should NOT have been created
            if (l->getOutgoingAtom(0)->getHash() == A->getHash() &&
                l->getOutgoingAtom(1)->getHash() == B->getHash()) {
                throw std::runtime_error("A→B link should not have been created");
            }
        }
    END_TEST
}

void testImplicationEvaluatesExistingLink() {
    TEST("PLNImplicationStep: evaluates existing IMPLICATION_LINK with unset TV")
        AtomSpace space;
        float sA = 0.6f, sB = 0.8f;
        auto A = createConceptNode(space, "ImpA5");
        auto B = createConceptNode(space, "ImpB5");
        A->setTruthValue(TruthValue::create(sA, 0.8f));
        B->setTruthValue(TruthValue::create(sB, 0.9f));

        // Create link and explicitly mark it as "unset" (confidence = 0)
        auto impLink = space.addLink(Atom::Type::IMPLICATION_LINK, {A, B});
        impLink->setTruthValue(TruthValue::create(0.5f, 0.0f));
        ASSERT(TruthValue::getConfidence(impLink->getTruthValue()) == 0.0f);

        std::vector<Atom::Handle> ws = {A, B, impLink};
        PLNImplicationStep step;
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        float s = TruthValue::getStrength(impLink->getTruthValue());
        float expected = 1.0f - sA + sA * sB;
        ASSERT_NEAR(s, expected, 1e-4f);
    END_TEST
}

void testImplicationSkipsAlreadyExistingPair() {
    TEST("PLNImplicationStep: does not create duplicate IMPLICATION_LINK")
        AtomSpace space;
        auto A = createConceptNode(space, "ImpA6");
        auto B = createConceptNode(space, "ImpB6");
        A->setTruthValue(TruthValue::create(0.8f, 0.9f));
        B->setTruthValue(TruthValue::create(0.7f, 0.9f));

        // Pre-create the link with a known TV
        auto existing = space.addLink(Atom::Type::IMPLICATION_LINK, {A, B});
        existing->setTruthValue(TruthValue::create(0.55f, 0.7f));

        std::vector<Atom::Handle> ws = {A, B, existing};
        PLNImplicationStep step(0.5f, 0.0f);
        step.execute(ws, space);

        // Count IMPLICATION_LINKs A→B — should remain exactly 1
        int count = 0;
        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getArity() == 2 &&
                l->getOutgoingAtom(0)->getHash() == A->getHash() &&
                l->getOutgoingAtom(1)->getHash() == B->getHash()) {
                ++count;
            }
        }
        ASSERT(count == 1);
    END_TEST
}

void testImplicationDirectionality() {
    TEST("PLNImplicationStep: A→B and B→A are distinct links")
        AtomSpace space;
        float sA = 0.8f, sB = 0.5f;
        auto A = createConceptNode(space, "DirA");
        auto B = createConceptNode(space, "DirB");
        A->setTruthValue(TruthValue::create(sA, 0.9f));
        B->setTruthValue(TruthValue::create(sB, 0.9f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNImplicationStep step(0.3f, 0.0f);  // accept both directions
        step.execute(ws, space);

        float sAB = 1.0f - sA + sA * sB;  // A→B strength
        float sBA = 1.0f - sB + sB * sA;  // B→A strength

        float foundAB = -1.0f, foundBA = -1.0f;
        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getArity() != 2) continue;
            if (l->getOutgoingAtom(0)->getHash() == A->getHash())
                foundAB = TruthValue::getStrength(a->getTruthValue());
            else if (l->getOutgoingAtom(0)->getHash() == B->getHash())
                foundBA = TruthValue::getStrength(a->getTruthValue());
        }
        ASSERT(foundAB >= 0.0f);
        ASSERT(foundBA >= 0.0f);
        ASSERT_NEAR(foundAB, sAB, 1e-4f);
        ASSERT_NEAR(foundBA, sBA, 1e-4f);
        // A→B ≠ B→A (different strengths given sA ≠ sB)
        ASSERT(std::abs(foundAB - foundBA) > 1e-4f);
    END_TEST
}

// ======================================================================== //
//  PLNImplicationChainStep                                                   //
// ======================================================================== //

void testChainFollowsTwoHops() {
    TEST("PLNImplicationChainStep: derives A→C from A→B, B→C")
        AtomSpace space;
        auto A = createConceptNode(space, "ChA");
        auto B = createConceptNode(space, "ChB");
        auto C = createConceptNode(space, "ChC");

        auto ab = space.addLink(Atom::Type::IMPLICATION_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::IMPLICATION_LINK, {B, C});
        ab->setTruthValue(TruthValue::create(0.8f, 0.7f));
        bc->setTruthValue(TruthValue::create(0.7f, 0.6f));

        std::vector<Atom::Handle> ws = {A, B, C, ab, bc};
        PLNImplicationChainStep step(3, 0.0f);
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        bool foundAC = false;
        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getArity() == 2 &&
                l->getOutgoingAtom(0)->getHash() == A->getHash() &&
                l->getOutgoingAtom(1)->getHash() == C->getHash()) {
                foundAC = true;
            }
        }
        ASSERT(foundAC);
    END_TEST
}

void testChainThreeHops() {
    TEST("PLNImplicationChainStep: derives A→D from A→B, B→C, C→D")
        AtomSpace space;
        auto A = createConceptNode(space, "Tri1A");
        auto B = createConceptNode(space, "Tri1B");
        auto C = createConceptNode(space, "Tri1C");
        auto D = createConceptNode(space, "Tri1D");

        auto ab = space.addLink(Atom::Type::IMPLICATION_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::IMPLICATION_LINK, {B, C});
        auto cd = space.addLink(Atom::Type::IMPLICATION_LINK, {C, D});
        ab->setTruthValue(TruthValue::create(0.9f, 0.8f));
        bc->setTruthValue(TruthValue::create(0.8f, 0.7f));
        cd->setTruthValue(TruthValue::create(0.7f, 0.6f));

        std::vector<Atom::Handle> ws = {A, B, C, D, ab, bc, cd};
        PLNImplicationChainStep step(3, 0.0f);
        bool changed = step.execute(ws, space);

        ASSERT(changed);
        bool foundAD = false;
        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getArity() == 2 &&
                l->getOutgoingAtom(0)->getHash() == A->getHash() &&
                l->getOutgoingAtom(1)->getHash() == D->getHash()) {
                foundAD = true;
            }
        }
        ASSERT(foundAD);
    END_TEST
}

void testChainRespectsMaxDepth() {
    TEST("PLNImplicationChainStep: respects maxDepth=1 (no transitive derivation)")
        AtomSpace space;
        auto A = createConceptNode(space, "Dep1A");
        auto B = createConceptNode(space, "Dep1B");
        auto C = createConceptNode(space, "Dep1C");

        auto ab = space.addLink(Atom::Type::IMPLICATION_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::IMPLICATION_LINK, {B, C});
        ab->setTruthValue(TruthValue::create(0.8f, 0.7f));
        bc->setTruthValue(TruthValue::create(0.7f, 0.6f));

        std::vector<Atom::Handle> ws = {A, B, C, ab, bc};
        PLNImplicationChainStep step(1, 0.0f);  // depth=1 → only 1-hop
        bool changed = step.execute(ws, space);

        // With depth=1 the chain traversal sees each direct edge (depth 0)
        // but never creates a second-hop (depth 1) link.
        ASSERT(!changed);
    END_TEST
}

void testChainDoesNotDuplicateExistingLink() {
    TEST("PLNImplicationChainStep: skips A→C when it already exists")
        AtomSpace space;
        auto A = createConceptNode(space, "DupA");
        auto B = createConceptNode(space, "DupB");
        auto C = createConceptNode(space, "DupC");

        auto ab = space.addLink(Atom::Type::IMPLICATION_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::IMPLICATION_LINK, {B, C});
        auto ac = space.addLink(Atom::Type::IMPLICATION_LINK, {A, C});
        ab->setTruthValue(TruthValue::create(0.8f, 0.7f));
        bc->setTruthValue(TruthValue::create(0.7f, 0.6f));
        ac->setTruthValue(TruthValue::create(0.5f, 0.5f));

        std::vector<Atom::Handle> ws = {A, B, C, ab, bc, ac};
        size_t sizeBefore = ws.size();
        PLNImplicationChainStep step(3, 0.0f);
        step.execute(ws, space);

        // Count A→C links — should remain exactly 1
        int count = 0;
        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getArity() == 2 &&
                l->getOutgoingAtom(0)->getHash() == A->getHash() &&
                l->getOutgoingAtom(1)->getHash() == C->getHash()) {
                ++count;
            }
        }
        ASSERT(count == 1);
    END_TEST
}

void testChainTVUsesDeductionFormula() {
    TEST("PLNImplicationChainStep: chained TV uses PLN deduction formula")
        AtomSpace space;
        auto A = createConceptNode(space, "TVA");
        auto B = createConceptNode(space, "TVB");
        auto C = createConceptNode(space, "TVC");

        float sAB = 0.8f, cAB = 0.9f;
        float sBC = 0.7f, cBC = 0.6f;

        auto ab = space.addLink(Atom::Type::IMPLICATION_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::IMPLICATION_LINK, {B, C});
        ab->setTruthValue(TruthValue::create(sAB, cAB));
        bc->setTruthValue(TruthValue::create(sBC, cBC));

        // Expected via PLN deduction: strength = sAB * sBC
        float expectedStr  = sAB * sBC;

        std::vector<Atom::Handle> ws = {A, B, C, ab, bc};
        PLNImplicationChainStep step(3, 0.0f);
        step.execute(ws, space);

        for (const auto& a : ws) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getArity() != 2) continue;
            if (l->getOutgoingAtom(0)->getHash() != A->getHash()) continue;
            if (l->getOutgoingAtom(1)->getHash() != C->getHash()) continue;

            float s = TruthValue::getStrength(a->getTruthValue());
            ASSERT_NEAR(s, expectedStr, 1e-4f);
        }
    END_TEST
}

void testChainIgnoresLowConfidenceLinks() {
    TEST("PLNImplicationChainStep: links with confidence=0 are not followed")
        AtomSpace space;
        auto A = createConceptNode(space, "LC_A");
        auto B = createConceptNode(space, "LC_B");
        auto C = createConceptNode(space, "LC_C");

        // Explicitly set A→B with confidence=0 (unset / not yet derived)
        auto ab = space.addLink(Atom::Type::IMPLICATION_LINK, {A, B});
        ab->setTruthValue(TruthValue::create(0.8f, 0.0f));   // confidence = 0
        auto bc = space.addLink(Atom::Type::IMPLICATION_LINK, {B, C});
        bc->setTruthValue(TruthValue::create(0.8f, 0.7f));

        std::vector<Atom::Handle> ws = {A, B, C, ab, bc};
        PLNImplicationChainStep step(3, 0.0f);
        bool changed = step.execute(ws, space);

        // A→C must NOT be derived because A→B has zero confidence
        ASSERT(!changed);
    END_TEST
}

void testChainNoPairWithNoLinks() {
    TEST("PLNImplicationChainStep: no-op when working set has no implication links")
        AtomSpace space;
        auto A = createConceptNode(space, "NL_A");
        auto B = createConceptNode(space, "NL_B");
        A->setTruthValue(TruthValue::create(0.8f, 0.9f));
        B->setTruthValue(TruthValue::create(0.7f, 0.8f));

        std::vector<Atom::Handle> ws = {A, B};
        PLNImplicationChainStep step(3, 0.0f);
        bool changed = step.execute(ws, space);
        ASSERT(!changed);
    END_TEST
}

// ======================================================================== //
//  InferencePipeline - Phase 13 fluent API                                  //
// ======================================================================== //

void testPipelinePlnImplicationMethod() {
    TEST("InferencePipeline::plnImplication appends PLNImplicationStep")
        AtomSpace space;
        InferencePipeline p(space);
        p.plnImplication(0.5f, 0.4f);
        ASSERT(p.size() == 1);
        ASSERT(p.stepNames()[0] == "PLNImplication");
    END_TEST
}

void testPipelinePlnImplicationChainMethod() {
    TEST("InferencePipeline::plnImplicationChain appends PLNImplicationChainStep")
        AtomSpace space;
        InferencePipeline p(space);
        p.plnImplicationChain(3, 0.0f);
        ASSERT(p.size() == 1);
        ASSERT(p.stepNames()[0] == "PLNImplicationChain");
    END_TEST
}

void testMakePLNCompletePipelineStepCount() {
    TEST("makePLNCompletePipeline creates 8-step pipeline")
        AtomSpace space;
        auto p = makePLNCompletePipeline(space);
        ASSERT(p.size() == 8);
    END_TEST
}

void testMakePLNCompletePipelineStepNames() {
    TEST("makePLNCompletePipeline has expected step names in order")
        AtomSpace space;
        auto p = makePLNCompletePipeline(space);
        auto names = p.stepNames();
        ASSERT(names[0] == "PLNDeduction");
        ASSERT(names[1] == "PLNConjunction");
        ASSERT(names[2] == "PLNDisjunction");
        ASSERT(names[3] == "PLNSimilarity");
        ASSERT(names[4] == "PLNImplication");
        ASSERT(names[5] == "PLNImplicationChain");
        ASSERT(names[6] == "PLNRevision");
        // names[7] starts with "TVThreshold"
        ASSERT(names[7].substr(0, 11) == "TVThreshold");
    END_TEST
}

void testMakePLNFullPipelineUnchanged() {
    TEST("makePLNFullPipeline still creates 6-step pipeline (backward compat)")
        AtomSpace space;
        auto p = makePLNFullPipeline(space);
        ASSERT(p.size() == 6);
        auto names = p.stepNames();
        ASSERT(names[0] == "PLNDeduction");
        ASSERT(names[5].substr(0, 11) == "TVThreshold");
    END_TEST
}

void testFluentChainingPhase13() {
    TEST("Fluent chaining: plnImplication().plnImplicationChain() builds 2-step pipeline")
        AtomSpace space;
        InferencePipeline p(space);
        p.plnImplication().plnImplicationChain();
        ASSERT(p.size() == 2);
        ASSERT(p.stepNames()[0] == "PLNImplication");
        ASSERT(p.stepNames()[1] == "PLNImplicationChain");
    END_TEST
}

// ======================================================================== //
//  End-to-end                                                                //
// ======================================================================== //

void testCompletePipelineEndToEnd() {
    TEST("makePLNCompletePipeline end-to-end: derives implication chains")
        AtomSpace space;
        auto A = createConceptNode(space, "E2E_A");
        auto B = createConceptNode(space, "E2E_B");
        auto C = createConceptNode(space, "E2E_C");
        A->setTruthValue(TruthValue::create(0.9f, 0.85f));
        B->setTruthValue(TruthValue::create(0.8f, 0.75f));
        C->setTruthValue(TruthValue::create(0.7f, 0.65f));

        auto ab = space.addLink(Atom::Type::IMPLICATION_LINK, {A, B});
        auto bc = space.addLink(Atom::Type::IMPLICATION_LINK, {B, C});
        ab->setTruthValue(TruthValue::create(0.85f, 0.80f));
        bc->setTruthValue(TruthValue::create(0.75f, 0.70f));

        auto p = makePLNCompletePipeline(space, 0.0f, 0.0f, 0.4f, 0.4f, 0.4f, 0.0f, 3);
        auto result = p.run({A, B, C, ab, bc});

        // Should have more atoms than we started with (new links derived)
        ASSERT(result.atoms.size() > 5);
        ASSERT(result.iterationsRun == 1);

        // Find A→C derived by implication chain
        bool foundAC = false;
        for (const auto& a : result.atoms) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getArity() == 2 &&
                l->getOutgoingAtom(0)->getHash() == A->getHash() &&
                l->getOutgoingAtom(1)->getHash() == C->getHash()) {
                foundAC = true;
            }
        }
        ASSERT(foundAC);
    END_TEST
}

void testImplicationStepInPipelineRun() {
    TEST("PLNImplicationStep via InferencePipeline::run derives directed links")
        AtomSpace space;
        auto X = createConceptNode(space, "RunX");
        auto Y = createConceptNode(space, "RunY");
        X->setTruthValue(TruthValue::create(0.75f, 0.85f));
        Y->setTruthValue(TruthValue::create(0.65f, 0.75f));

        InferencePipeline p(space);
        p.plnImplication(0.5f, 0.0f);
        auto result = p.run({X, Y});

        bool hasImpLink = false;
        for (const auto& a : result.atoms) {
            if (a->isLink() && a->getType() == Atom::Type::IMPLICATION_LINK)
                hasImpLink = true;
        }
        ASSERT(hasImpLink);
    END_TEST
}

void testImplicationChainInPipelineRun() {
    TEST("PLNImplicationChainStep via InferencePipeline::run derives transitive link")
        AtomSpace space;
        auto P = createConceptNode(space, "RunP");
        auto Q = createConceptNode(space, "RunQ");
        auto R = createConceptNode(space, "RunR");

        auto pq = space.addLink(Atom::Type::IMPLICATION_LINK, {P, Q});
        auto qr = space.addLink(Atom::Type::IMPLICATION_LINK, {Q, R});
        pq->setTruthValue(TruthValue::create(0.8f, 0.7f));
        qr->setTruthValue(TruthValue::create(0.7f, 0.6f));

        InferencePipeline pipe(space);
        pipe.plnImplicationChain(3, 0.0f);
        auto result = pipe.run({P, Q, R, pq, qr});

        bool foundPR = false;
        for (const auto& a : result.atoms) {
            if (!a->isLink() || a->getType() != Atom::Type::IMPLICATION_LINK) continue;
            const Link* l = static_cast<const Link*>(a.get());
            if (l->getArity() == 2 &&
                l->getOutgoingAtom(0)->getHash() == P->getHash() &&
                l->getOutgoingAtom(1)->getHash() == R->getHash()) {
                foundPR = true;
            }
        }
        ASSERT(foundPR);
    END_TEST
}

// ======================================================================== //
//  Main
// ======================================================================== //

int main() {
    std::cout << "\n=== Phase 13 Tests ===\n\n";

    std::cout << "-- PLNImplicationStep --\n";
    testImplicationCreatesLink();
    testImplicationTV();
    testImplicationSkipsLowStrengthAntecedent();
    testImplicationSkipsBelowImpStrengthThreshold();
    testImplicationEvaluatesExistingLink();
    testImplicationSkipsAlreadyExistingPair();
    testImplicationDirectionality();

    std::cout << "\n-- PLNImplicationChainStep --\n";
    testChainFollowsTwoHops();
    testChainThreeHops();
    testChainRespectsMaxDepth();
    testChainDoesNotDuplicateExistingLink();
    testChainTVUsesDeductionFormula();
    testChainIgnoresLowConfidenceLinks();
    testChainNoPairWithNoLinks();

    std::cout << "\n-- InferencePipeline Phase 13 --\n";
    testPipelinePlnImplicationMethod();
    testPipelinePlnImplicationChainMethod();
    testMakePLNCompletePipelineStepCount();
    testMakePLNCompletePipelineStepNames();
    testMakePLNFullPipelineUnchanged();
    testFluentChainingPhase13();

    std::cout << "\n-- End-to-end --\n";
    testCompletePipelineEndToEnd();
    testImplicationStepInPipelineRun();
    testImplicationChainInPipelineRun();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed  << " failed ===\n\n";

    return tests_failed > 0 ? 1 : 0;
}
