#include <ATen/atomspace/ATenSpace.h>
#include <ATen/atomspace/TensorLogicEngine.h>
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>

using namespace at::atomspace;

// Helper function for approximate equality
bool approxEqual(float a, float b, float epsilon = 0.01f) {
    return std::abs(a - b) < epsilon;
}

// Tighter tolerance for the vectorised vs scalar equivalence checks
// (Iteration 2, FR-2.5: batch results must match the scalar reference).
bool tvNear(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) <= eps;
}

void testPatternMatching() {
    std::cout << "Testing Pattern Matching... ";
    
    AtomSpace space;
    
    // Create knowledge base
    auto cat = createConceptNode(space, "cat");
    auto dog = createConceptNode(space, "dog");
    auto mammal = createConceptNode(space, "mammal");
    
    auto catMammal = createInheritanceLink(space, cat, mammal);
    auto dogMammal = createInheritanceLink(space, dog, mammal);
    
    // Test 1: Pattern with variable
    auto varX = createVariableNode(space, "$X");
    auto pattern = createInheritanceLink(space, varX, mammal);
    
    auto matches = PatternMatcher::findMatches(space, pattern);
    assert(matches.size() == 2);
    
    // Test 2: Variable binding
    VariableBinding bindings;
    bool matched = PatternMatcher::match(pattern, catMammal, bindings);
    assert(matched);
    assert(bindings.size() == 1);
    assert(bindings[varX]->equals(*cat));
    
    // Test 3: Pattern substitution
    auto result = PatternMatcher::substitute(pattern, bindings, space);
    assert(result->equals(*catMammal));
    
    // Test 4: Unification
    auto varY = createVariableNode(space, "$Y");
    auto pattern2 = createInheritanceLink(space, varY, mammal);
    VariableBinding unifyBindings;
    bool unified = PatternMatcher::unify(pattern, pattern2, unifyBindings);
    assert(unified);
    
    std::cout << "PASSED" << std::endl;
}

void testTruthValueFormulas() {
    std::cout << "Testing Truth Value Formulas... ";
    
    // Test 1: Deduction
    auto tv1 = TruthValue::create(0.9f, 0.8f);
    auto tv2 = TruthValue::create(0.8f, 0.9f);
    auto tvDeduced = TruthValue::deduction(tv1, tv2);
    
    float strength = TruthValue::getStrength(tvDeduced);
    float confidence = TruthValue::getConfidence(tvDeduced);
    
    assert(strength > 0.7f && strength < 0.75f); // 0.9 * 0.8 = 0.72
    assert(confidence > 0.5f && confidence < 0.8f);
    
    // Test 2: Induction
    auto tvInduced = TruthValue::induction(8, 10);
    assert(approxEqual(TruthValue::getStrength(tvInduced), 0.8f));
    assert(TruthValue::getConfidence(tvInduced) > 0.8f);
    
    // Test 3: Conjunction
    auto tvA = TruthValue::create(0.8f, 0.9f);
    auto tvB = TruthValue::create(0.7f, 0.85f);
    auto tvAnd = TruthValue::conjunction(tvA, tvB);
    
    assert(approxEqual(TruthValue::getStrength(tvAnd), 0.56f)); // 0.8 * 0.7
    
    // Test 4: Disjunction
    auto tvOr = TruthValue::disjunction(tvA, tvB);
    float orStrength = TruthValue::getStrength(tvOr);
    assert(orStrength > 0.9f && orStrength < 1.0f); // 0.8 + 0.7 - 0.56 = 0.94
    
    // Test 5: Negation
    auto tvNot = TruthValue::negation(tvA);
    assert(approxEqual(TruthValue::getStrength(tvNot), 0.2f)); // 1 - 0.8
    assert(approxEqual(TruthValue::getConfidence(tvNot), 0.9f)); // Same confidence
    
    // Test 6: Revision
    auto tv1a = TruthValue::create(0.7f, 0.8f);
    auto tv1b = TruthValue::create(0.8f, 0.7f);
    auto tvRevised = TruthValue::revision(tv1a, tv1b);
    float revisedStrength = TruthValue::getStrength(tvRevised);
    assert(revisedStrength > 0.7f && revisedStrength < 0.8f); // Weighted average
    
    // Test 7: Abduction
    auto tvObserved = TruthValue::create(0.9f, 0.8f);
    auto tvAB = TruthValue::create(0.85f, 0.9f);
    auto tvAbduced = TruthValue::abduction(tvObserved, tvAB);
    assert(TruthValue::getStrength(tvAbduced) > 0.5f);
    assert(TruthValue::getConfidence(tvAbduced) < TruthValue::getConfidence(tvAB));
    
    std::cout << "PASSED" << std::endl;
}

void testDeductionRule() {
    std::cout << "Testing Deduction Rule... ";
    
    AtomSpace space;
    
    // Create A→B and B→C
    auto A = createConceptNode(space, "A");
    auto B = createConceptNode(space, "B");
    auto C = createConceptNode(space, "C");
    
    auto AB = createInheritanceLink(space, A, B);
    AB->setTruthValue(TruthValue::create(0.9f, 0.8f));
    
    auto BC = createInheritanceLink(space, B, C);
    BC->setTruthValue(TruthValue::create(0.85f, 0.9f));
    
    // Apply deduction rule
    DeductionRule rule;
    std::vector<Atom::Handle> premises = {AB, BC};
    
    assert(rule.canApply(premises));
    
    auto conclusions = rule.apply(premises, space);
    assert(conclusions.size() == 1);
    
    // Check that A→C was created
    auto AC = conclusions[0];
    const Link* acLink = static_cast<const Link*>(AC.get());
    assert(acLink->getOutgoingAtom(0)->equals(*A));
    assert(acLink->getOutgoingAtom(1)->equals(*C));
    
    // Check truth value
    float strength = TruthValue::getStrength(AC->getTruthValue());
    assert(strength > 0.7f && strength < 0.8f); // 0.9 * 0.85 ≈ 0.765
    
    std::cout << "PASSED" << std::endl;
}

void testForwardChaining() {
    std::cout << "Testing Forward Chaining... ";
    
    AtomSpace space;
    
    // Create a chain: A→B→C→D
    auto A = createConceptNode(space, "A");
    auto B = createConceptNode(space, "B");
    auto C = createConceptNode(space, "C");
    auto D = createConceptNode(space, "D");
    
    auto AB = createInheritanceLink(space, A, B);
    AB->setTruthValue(TruthValue::create(0.9f, 0.9f));
    
    auto BC = createInheritanceLink(space, B, C);
    BC->setTruthValue(TruthValue::create(0.9f, 0.9f));
    
    auto CD = createInheritanceLink(space, C, D);
    CD->setTruthValue(TruthValue::create(0.9f, 0.9f));
    
    size_t initialSize = space.getSize();
    
    // Run forward chaining
    ForwardChainer chainer(space);
    chainer.setMaxIterations(10);
    chainer.setConfidenceThreshold(0.1f);
    
    int newAtoms = chainer.run();
    assert(newAtoms > 0);
    assert(space.getSize() > initialSize);
    
    // Check that A→C was inferred
    auto AC = space.getLink(Atom::Type::INHERITANCE_LINK, {A, C});
    assert(AC != nullptr);
    
    // Check that A→D was inferred
    auto AD = space.getLink(Atom::Type::INHERITANCE_LINK, {A, D});
    assert(AD != nullptr);
    
    std::cout << "PASSED" << std::endl;
}

void testBackwardChaining() {
    std::cout << "Testing Backward Chaining... ";
    
    AtomSpace space;
    
    // Create knowledge base: Socrates→Human→Mortal
    auto socrates = createConceptNode(space, "Socrates");
    auto human = createConceptNode(space, "Human");
    auto mortal = createConceptNode(space, "Mortal");
    
    auto socratesHuman = createInheritanceLink(space, socrates, human);
    socratesHuman->setTruthValue(TruthValue::create(1.0f, 0.95f));
    
    auto humanMortal = createInheritanceLink(space, human, mortal);
    humanMortal->setTruthValue(TruthValue::create(0.99f, 0.99f));
    
    // Goal: Prove Socrates→Mortal
    auto goal = createInheritanceLink(space, socrates, mortal);
    
    BackwardChainer chainer(space);
    chainer.addRule(std::make_shared<DeductionRule>());
    chainer.setMaxDepth(5);
    
    auto proofs = chainer.prove(goal);
    assert(proofs.size() > 0);
    
    // Check proof structure
    auto proof = proofs[0];
    assert(proof->goal->equals(*goal));
    
    // Check that truth value is high
    float confidence = TruthValue::getConfidence(proof->truthValue);
    assert(confidence > 0.8f);
    
    // Test canProve
    assert(chainer.canProve(goal));
    
    // Test query
    auto tv = chainer.query(goal);
    assert(TruthValue::getConfidence(tv) > 0.8f);
    
    std::cout << "PASSED" << std::endl;
}

void testComplexPatterns() {
    std::cout << "Testing Complex Patterns... ";
    
    AtomSpace space;
    
    // Create nested structure
    auto predicate = createPredicateNode(space, "likes");
    auto john = createConceptNode(space, "John");
    auto mary = createConceptNode(space, "Mary");
    
    auto eval = createEvaluationLink(space, predicate, {john, mary});
    
    // Pattern with variable in nested structure
    auto varX = createVariableNode(space, "$X");
    auto patternEval = createEvaluationLink(space, predicate, {john, varX});
    
    VariableBinding bindings;
    bool matched = PatternMatcher::match(patternEval, eval, bindings);
    assert(matched);
    assert(bindings[varX]->equals(*mary));
    
    // Test pattern extraction
    assert(Pattern::hasVariables(patternEval));
    auto vars = Pattern::getVariables(patternEval);
    assert(vars.size() == 1);
    assert(vars[0]->equals(*varX));
    
    std::cout << "PASSED" << std::endl;
}

void testLogicalOperations() {
    std::cout << "Testing Logical Operations... ";
    
    AtomSpace space;
    
    auto A = createConceptNode(space, "A");
    auto B = createConceptNode(space, "B");
    auto C = createConceptNode(space, "C");
    
    // Test AND link
    auto andLink = createAndLink(space, {A, B, C});
    assert(andLink->getType() == Atom::Type::AND_LINK);
    const Link* andLinkPtr = static_cast<const Link*>(andLink.get());
    assert(andLinkPtr->getArity() == 3);
    
    // Test OR link
    auto orLink = createOrLink(space, {A, B});
    assert(orLink->getType() == Atom::Type::OR_LINK);
    
    // Test NOT link
    auto notLink = createNotLink(space, A);
    assert(notLink->getType() == Atom::Type::NOT_LINK);
    
    // Test truth value propagation
    A->setTruthValue(TruthValue::create(0.8f, 0.9f));
    B->setTruthValue(TruthValue::create(0.7f, 0.85f));
    
    auto tvAnd = TruthValue::conjunction(A->getTruthValue(), B->getTruthValue());
    andLink->setTruthValue(tvAnd);
    
    float andStrength = TruthValue::getStrength(andLink->getTruthValue());
    assert(approxEqual(andStrength, 0.56f));
    
    std::cout << "PASSED" << std::endl;
}

void testImplicationLink() {
    std::cout << "Testing Implication Link... ";
    
    AtomSpace space;
    
    auto A = createConceptNode(space, "A");
    auto B = createConceptNode(space, "B");
    
    // Create implication link
    auto impl = createImplicationLink(space, A, B);
    assert(impl->getType() == Atom::Type::IMPLICATION_LINK);
    
    const Link* implLink = static_cast<const Link*>(impl.get());
    assert(implLink->getArity() == 2);
    assert(implLink->getOutgoingAtom(0)->equals(*A));
    assert(implLink->getOutgoingAtom(1)->equals(*B));
    
    // Set truth values and compute implication
    A->setTruthValue(TruthValue::create(0.8f, 0.9f));
    B->setTruthValue(TruthValue::create(0.7f, 0.85f));
    
    auto tvImpl = TruthValue::implication(A->getTruthValue(), B->getTruthValue());
    impl->setTruthValue(tvImpl);
    
    float strength = TruthValue::getStrength(impl->getTruthValue());
    assert(strength > 0.7f); // Should be relatively high
    
    std::cout << "PASSED" << std::endl;
}

void testAttentionGuidedInference() {
    std::cout << "Testing Attention-Guided Inference... ";
    
    AtomSpace space;
    AttentionBank attentionBank;
    
    // Create atoms with different attention values
    auto A = createConceptNode(space, "A");
    auto B = createConceptNode(space, "B");
    auto C = createConceptNode(space, "C");
    
    auto AB = createInheritanceLink(space, A, B);
    AB->setTruthValue(TruthValue::create(0.9f, 0.9f));
    attentionBank.setSTI(AB, 100.0f);
    
    auto BC = createInheritanceLink(space, B, C);
    BC->setTruthValue(TruthValue::create(0.9f, 0.9f));
    attentionBank.setSTI(BC, 50.0f);
    
    // Forward chain with attention guidance
    ForwardChainer chainer(space);
    chainer.setConfidenceThreshold(0.1f);
    
    int newAtoms = chainer.run(&attentionBank);
    assert(newAtoms > 0);
    
    // Check that inference was performed
    auto AC = space.getLink(Atom::Type::INHERITANCE_LINK, {A, C});
    assert(AC != nullptr);
    
    std::cout << "PASSED" << std::endl;
}

void testIndefiniteTruthValues() {
    std::cout << "Testing Indefinite Truth Values... ";
    
    // Test with various counts
    auto tv1 = TruthValue::indefinite(7, 3);  // 7 positive, 3 negative
    assert(approxEqual(TruthValue::getStrength(tv1), 0.7f));
    assert(TruthValue::getConfidence(tv1) > 0.0f);
    
    auto tv2 = TruthValue::indefinite(100, 50);  // More observations
    assert(approxEqual(TruthValue::getStrength(tv2), 0.667f, 0.01f));
    assert(TruthValue::getConfidence(tv2) > TruthValue::getConfidence(tv1));
    
    // Test default cases
    auto tvDefault = TruthValue::defaultTV();
    assert(approxEqual(TruthValue::getStrength(tvDefault), 0.5f));
    assert(approxEqual(TruthValue::getConfidence(tvDefault), 0.0f));
    
    auto tvTrue = TruthValue::trueTV();
    assert(approxEqual(TruthValue::getStrength(tvTrue), 1.0f));
    assert(TruthValue::getConfidence(tvTrue) > 0.8f);
    
    auto tvFalse = TruthValue::falseTV();
    assert(approxEqual(TruthValue::getStrength(tvFalse), 0.0f));
    assert(TruthValue::getConfidence(tvFalse) > 0.8f);
    
    std::cout << "PASSED" << std::endl;
}

// ======================================================================= //
//  Iteration 2: vectorised PLN tensor contractions                          //
// ======================================================================= //

void testBatchDeductionEquivalence() {
    std::cout << "Testing Batch Deduction (vectorised vs scalar)... ";

    AtomSpace space;
    TensorLogicEngine engine;

    // Fixed corpus of (A_i→B_i, B_i→C_i) premise pairs.
    const float kStrengths1[] = {0.9f, 0.5f, 0.1f, 1.0f, 0.73f};
    const float kConfs1[]     = {0.8f, 0.2f, 0.9f, 0.5f, 0.66f};
    const float kStrengths2[] = {0.8f, 0.4f, 0.3f, 0.9f, 0.21f};
    const float kConfs2[]     = {0.9f, 0.7f, 0.1f, 0.4f, 0.88f};
    const size_t n = 5;

    std::vector<Atom::Handle> p1, p2;
    for (size_t i = 0; i < n; ++i) {
        auto A = createConceptNode(space, "bd-A" + std::to_string(i));
        auto B = createConceptNode(space, "bd-B" + std::to_string(i));
        auto C = createConceptNode(space, "bd-C" + std::to_string(i));
        auto AB = createInheritanceLink(space, A, B);
        auto BC = createInheritanceLink(space, B, C);
        AB->setTruthValue(TruthValue::create(kStrengths1[i], kConfs1[i]));
        BC->setTruthValue(TruthValue::create(kStrengths2[i], kConfs2[i]));
        p1.push_back(AB);
        p2.push_back(BC);
    }

    at::Tensor batched = engine.batchDeduction(p1, p2);
    assert(batched.size(0) == static_cast<int64_t>(n));
    assert(batched.size(1) == 2);

    for (size_t i = 0; i < n; ++i) {
        at::Tensor ref = TruthValue::deduction(p1[i]->getTruthValue(),
                                           p2[i]->getTruthValue());
        assert(tvNear(batched[i][0].item<float>(),
                      TruthValue::getStrength(ref)));
        assert(tvNear(batched[i][1].item<float>(),
                      TruthValue::getConfidence(ref)));
    }

    // Raw-tensor contraction overload agrees with the atom-based overload.
    at::Tensor raw = engine.batchDeductionTV(engine.batchDeduction(p1, p2),
                                         engine.batchDeduction(p1, p2));
    at::Tensor twice = engine.batchDeduction(p1, p2);
    assert(raw.sizes() == twice.sizes());

    std::cout << "PASSED" << std::endl;
}

void testBatchInductionAbductionEquivalence() {
    std::cout << "Testing Batch Induction/Abduction (vectorised vs scalar)... ";

    AtomSpace space;
    TensorLogicEngine engine;

    const size_t n = 4;
    const float kS1[] = {0.9f, 0.6f, 0.3f, 0.85f};
    const float kC1[] = {0.8f, 0.5f, 0.95f, 0.42f};
    const float kS2[] = {0.7f, 0.55f, 0.25f, 0.9f};
    const float kC2[] = {0.9f, 0.65f, 0.8f, 0.37f};

    std::vector<Atom::Handle> p1, p2;
    for (size_t i = 0; i < n; ++i) {
        auto A = createConceptNode(space, "ia-A" + std::to_string(i));
        auto B = createConceptNode(space, "ia-B" + std::to_string(i));
        auto C = createConceptNode(space, "ia-C" + std::to_string(i));
        auto AB = createInheritanceLink(space, A, B);
        auto AC = createInheritanceLink(space, A, C);
        AB->setTruthValue(TruthValue::create(kS1[i], kC1[i]));
        AC->setTruthValue(TruthValue::create(kS2[i], kC2[i]));
        p1.push_back(AB);
        p2.push_back(AC);
    }

    at::Tensor ind = engine.batchInduction(p1, p2);
    at::Tensor abd = engine.batchAbduction(p1, p2);
    assert(ind.size(0) == static_cast<int64_t>(n));
    assert(abd.size(0) == static_cast<int64_t>(n));

    for (size_t i = 0; i < n; ++i) {
        float s1 = kS1[i], c1 = kC1[i], s2 = kS2[i], c2 = kC2[i];

        // Induction: s = s1*s2, c = deduction-like confidence * 0.8 discount
        float expIndS = s1 * s2;
        float expIndC = (c1 * c2 * (s1 + s2)) / (1.0f + s1 * s2)
                        * TruthValue::INDUCTION_DISCOUNT;
        assert(tvNear(ind[i][0].item<float>(), expIndS));
        assert(tvNear(ind[i][1].item<float>(), expIndC));

        // Abduction must match the scalar TruthValue::abduction reference.
        at::Tensor refAbd = TruthValue::abduction(p1[i]->getTruthValue(),
                                              p2[i]->getTruthValue());
        assert(tvNear(abd[i][0].item<float>(),
                      TruthValue::getStrength(refAbd)));
        assert(tvNear(abd[i][1].item<float>(),
                      TruthValue::getConfidence(refAbd)));
    }

    std::cout << "PASSED" << std::endl;
}

void testBatchRevisionEquivalence() {
    std::cout << "Testing Batch Revision (vectorised vs scalar)... ";

    AtomSpace space;
    TensorLogicEngine engine;

    const size_t n = 3;
    std::vector<Atom::Handle> e1, e2;
    const float kS1[] = {0.7f, 0.2f, 0.95f};
    const float kC1[] = {0.8f, 0.4f, 0.6f};
    const float kS2[] = {0.8f, 0.3f, 0.9f};
    const float kC2[] = {0.7f, 0.5f, 0.55f};

    for (size_t i = 0; i < n; ++i) {
        auto a = createConceptNode(space, "rev-a" + std::to_string(i));
        auto b = createConceptNode(space, "rev-b" + std::to_string(i));
        a->setTruthValue(TruthValue::create(kS1[i], kC1[i]));
        b->setTruthValue(TruthValue::create(kS2[i], kC2[i]));
        e1.push_back(a);
        e2.push_back(b);
    }

    at::Tensor revised = engine.batchRevision(e1, e2);
    for (size_t i = 0; i < n; ++i) {
        at::Tensor ref = TruthValue::revision(e1[i]->getTruthValue(),
                                          e2[i]->getTruthValue());
        assert(tvNear(revised[i][0].item<float>(),
                      TruthValue::getStrength(ref)));
        assert(tvNear(revised[i][1].item<float>(),
                      TruthValue::getConfidence(ref)));
    }

    // Edge case: empty batch.
    at::Tensor emptyRev = engine.batchRevision({}, {});
    assert(emptyRev.size(0) == 0);

    std::cout << "PASSED" << std::endl;
}

void testChainerBatchOffload() {
    std::cout << "Testing Chainer Batch Offload equivalence... ";

    AtomSpace space;

    // A→B→C chain with non-trivial truth values.
    auto A = createConceptNode(space, "off-A");
    auto B = createConceptNode(space, "off-B");
    auto C = createConceptNode(space, "off-C");

    auto AB = createInheritanceLink(space, A, B);
    AB->setTruthValue(TruthValue::create(0.9f, 0.8f));
    auto BC = createInheritanceLink(space, B, C);
    BC->setTruthValue(TruthValue::create(0.85f, 0.9f));

    // Batched forward chaining (default: batch offload enabled).
    ForwardChainer batchChainer(space);
    batchChainer.setMaxIterations(2);
    batchChainer.setConfidenceThreshold(0.1f);
    int newAtoms = batchChainer.run();
    assert(newAtoms > 0);

    auto AC = space.getLink(Atom::Type::INHERITANCE_LINK, {A, C});
    assert(AC != nullptr);

    // Vectorised result must match the scalar deduction reference.
    at::Tensor ref = TruthValue::deduction(AB->getTruthValue(),
                                       BC->getTruthValue());
    assert(tvNear(TruthValue::getStrength(AC->getTruthValue()),
                  TruthValue::getStrength(ref)));
    assert(tvNear(TruthValue::getConfidence(AC->getTruthValue()),
                  TruthValue::getConfidence(ref)));

    // Scalar fallback path (offload disabled) must produce the same TV.
    AtomSpace space2;
    auto A2 = createConceptNode(space2, "off-A");
    auto B2 = createConceptNode(space2, "off-B");
    auto C2 = createConceptNode(space2, "off-C");
    auto AB2 = createInheritanceLink(space2, A2, B2);
    AB2->setTruthValue(TruthValue::create(0.9f, 0.8f));
    auto BC2 = createInheritanceLink(space2, B2, C2);
    BC2->setTruthValue(TruthValue::create(0.85f, 0.9f));

    ForwardChainer scalarChainer(space2);
    scalarChainer.setBatchOffload(false);
    scalarChainer.setMaxIterations(2);
    scalarChainer.setConfidenceThreshold(0.1f);
    scalarChainer.run();

    auto AC2 = space2.getLink(Atom::Type::INHERITANCE_LINK, {A2, C2});
    assert(AC2 != nullptr);
    assert(tvNear(TruthValue::getStrength(AC2->getTruthValue()),
                  TruthValue::getStrength(ref)));
    assert(tvNear(TruthValue::getConfidence(AC2->getTruthValue()),
                  TruthValue::getConfidence(ref)));

    std::cout << "PASSED" << std::endl;
}

void testBatchDispatchHeuristic() {
    std::cout << "Testing TensorLogicEngine dispatch heuristic... ";

    TensorLogicEngine engine;

    // CPU mode always resolves to CPU.
    engine.setInferenceMode(TensorLogicEngine::InferenceMode::CPU);
    assert(engine.deviceFor(100000).is_cpu());

    // AUTO without CUDA always resolves to CPU (CPU fallback, T2.5).
    engine.setInferenceMode(TensorLogicEngine::InferenceMode::AUTO);
    if (!torch::cuda::is_available()) {
        assert(engine.deviceFor(0).is_cpu());
        assert(engine.deviceFor(4096).is_cpu());
    }

    // Opting out of GPU forces CPU even in GPU mode.
    engine.setUseGPU(false);
    engine.setInferenceMode(TensorLogicEngine::InferenceMode::GPU);
    assert(engine.deviceFor(4096).is_cpu());

    // GPU threshold heuristic is configurable.
    engine.setGPUThreshold(64);
    assert(engine.getGPUThreshold() == 64);

    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "Running PLN (Probabilistic Logic Networks) Tests" << std::endl;
    std::cout << "==================================================" << std::endl;

    try {
        testPatternMatching();
        testTruthValueFormulas();
        testDeductionRule();
        testForwardChaining();
        testBackwardChaining();
        testComplexPatterns();
        testLogicalOperations();
        testImplicationLink();
        testAttentionGuidedInference();
        testIndefiniteTruthValues();
        // Iteration 2: vectorised PLN
        testBatchDeductionEquivalence();
        testBatchInductionAbductionEquivalence();
        testBatchRevisionEquivalence();
        testChainerBatchOffload();
        testBatchDispatchHeuristic();
        
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nTest failed with unknown exception" << std::endl;
        return 1;
    }
}
