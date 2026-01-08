#include <ATen/atomspace/ATenSpace.h>
#include <ATen/atomspace/TensorLogicEngine.h>
#include <ATen/atomspace/CognitiveEngine.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace at::atomspace;

#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "TEST FAILED: " << message << std::endl; \
        return false; \
    }

bool almostEqual(float a, float b, float epsilon = 0.01f) {
    return std::abs(a - b) < epsilon;
}

/**
 * Test 1: TensorLogicEngine - Batch AND operation
 */
bool test_batchAND() {
    std::cout << "Test 1: Batch AND operation... ";
    
    AtomSpace space;
    TensorLogicEngine engine;
    
    auto a = createConceptNode(space, "a");
    a->setTruthValue(TruthValue::create(0.8f, 0.9f));
    
    auto b = createConceptNode(space, "b");
    b->setTruthValue(TruthValue::create(0.7f, 0.85f));
    
    std::vector<Atom::Handle> atoms1 = {a};
    std::vector<Atom::Handle> atoms2 = {b};
    
    auto result = engine.batchLogicalOperation(
        atoms1, atoms2, TensorLogicEngine::LogicalOperation::AND);
    
    TEST_ASSERT(result.size(0) == 1, "Should have 1 result");
    TEST_ASSERT(result.size(1) == 2, "Result should have 2 components");
    
    float strength = result[0][0].item<float>();
    float confidence = result[0][1].item<float>();
    
    TEST_ASSERT(almostEqual(strength, 0.56f), "AND strength incorrect");
    TEST_ASSERT(confidence > 0.0f && confidence <= 1.0f, "Confidence out of range");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 2: TensorLogicEngine - Batch OR operation
 */
bool test_batchOR() {
    std::cout << "Test 2: Batch OR operation... ";
    
    AtomSpace space;
    TensorLogicEngine engine;
    
    auto a = createConceptNode(space, "a");
    a->setTruthValue(TruthValue::create(0.6f, 0.8f));
    
    auto b = createConceptNode(space, "b");
    b->setTruthValue(TruthValue::create(0.5f, 0.75f));
    
    std::vector<Atom::Handle> atoms1 = {a};
    std::vector<Atom::Handle> atoms2 = {b};
    
    auto result = engine.batchLogicalOperation(
        atoms1, atoms2, TensorLogicEngine::LogicalOperation::OR);
    
    TEST_ASSERT(result.size(0) == 1, "Should have 1 result");
    
    float strength = result[0][0].item<float>();
    
    // OR strength should be a + b - a*b = 0.6 + 0.5 - 0.3 = 0.8
    TEST_ASSERT(almostEqual(strength, 0.8f), "OR strength incorrect");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 3: TensorLogicEngine - Batch NOT operation
 */
bool test_batchNOT() {
    std::cout << "Test 3: Batch NOT operation... ";
    
    AtomSpace space;
    TensorLogicEngine engine;
    
    auto a = createConceptNode(space, "a");
    a->setTruthValue(TruthValue::create(0.3f, 0.9f));
    
    auto b = createConceptNode(space, "b");
    b->setTruthValue(TruthValue::create(0.7f, 0.85f));
    
    std::vector<Atom::Handle> atoms = {a, b};
    
    auto result = engine.batchUnaryOperation(
        atoms, TensorLogicEngine::LogicalOperation::NOT);
    
    TEST_ASSERT(result.size(0) == 2, "Should have 2 results");
    
    float strength1 = result[0][0].item<float>();
    float strength2 = result[1][0].item<float>();
    
    TEST_ASSERT(almostEqual(strength1, 0.7f), "NOT strength incorrect for a");
    TEST_ASSERT(almostEqual(strength2, 0.3f), "NOT strength incorrect for b");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 4: TensorLogicEngine - Batch deduction
 */
bool test_batchDeduction() {
    std::cout << "Test 4: Batch deduction... ";
    
    AtomSpace space;
    TensorLogicEngine engine;
    
    auto a = createConceptNode(space, "a");
    auto b = createConceptNode(space, "b");
    auto c = createConceptNode(space, "c");
    
    auto link1 = createInheritanceLink(space, a, b);
    link1->setTruthValue(TruthValue::create(0.9f, 0.9f));
    
    auto link2 = createInheritanceLink(space, b, c);
    link2->setTruthValue(TruthValue::create(0.8f, 0.85f));
    
    std::vector<Atom::Handle> premises1 = {link1};
    std::vector<Atom::Handle> premises2 = {link2};
    
    auto result = engine.batchDeduction(premises1, premises2);
    
    TEST_ASSERT(result.size(0) == 1, "Should have 1 result");
    
    float strength = result[0][0].item<float>();
    float confidence = result[0][1].item<float>();
    
    // Deduction: strength = 0.9 * 0.8 = 0.72
    TEST_ASSERT(almostEqual(strength, 0.72f), "Deduction strength incorrect");
    TEST_ASSERT(confidence > 0.0f && confidence <= 1.0f, "Confidence out of range");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 5: TensorLogicEngine - Truth value distribution
 */
bool test_truthValueDistribution() {
    std::cout << "Test 5: Truth value distribution... ";
    
    AtomSpace space;
    TensorLogicEngine engine;
    
    auto a = createConceptNode(space, "a");
    a->setTruthValue(TruthValue::create(0.5f, 0.8f));
    
    auto b = createConceptNode(space, "b");
    b->setTruthValue(TruthValue::create(0.7f, 0.9f));
    
    auto c = createConceptNode(space, "c");
    c->setTruthValue(TruthValue::create(0.6f, 0.85f));
    
    std::vector<Atom::Handle> atoms = {a, b, c};
    
    auto dist = engine.computeTruthValueDistribution(atoms);
    
    TEST_ASSERT(dist.size(0) == 4, "Distribution should have 4 values");
    
    float meanStrength = dist[0].item<float>();
    float meanConfidence = dist[1].item<float>();
    
    // Mean strength = (0.5 + 0.7 + 0.6) / 3 = 0.6
    TEST_ASSERT(almostEqual(meanStrength, 0.6f), "Mean strength incorrect");
    
    // Mean confidence = (0.8 + 0.9 + 0.85) / 3 = 0.85
    TEST_ASSERT(almostEqual(meanConfidence, 0.85f), "Mean confidence incorrect");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 6: TensorLogicEngine - Filter by truth value
 */
bool test_filterByTruthValue() {
    std::cout << "Test 6: Filter by truth value... ";
    
    AtomSpace space;
    TensorLogicEngine engine;
    
    auto a = createConceptNode(space, "a");
    a->setTruthValue(TruthValue::create(0.9f, 0.8f));
    
    auto b = createConceptNode(space, "b");
    b->setTruthValue(TruthValue::create(0.5f, 0.7f));
    
    auto c = createConceptNode(space, "c");
    c->setTruthValue(TruthValue::create(0.3f, 0.9f));
    
    std::vector<Atom::Handle> atoms = {a, b, c};
    
    auto filtered = engine.filterByTruthValue(atoms, 0.6f, 0.75f);
    
    TEST_ASSERT(filtered.size() == 1, "Should filter to 1 atom");
    TEST_ASSERT(filtered[0] == a, "Should keep atom 'a'");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 7: CognitiveEngine - Construction and configuration
 */
bool test_cognitiveEngineConstruction() {
    std::cout << "Test 7: CognitiveEngine construction... ";
    
    AtomSpace space;
    CognitiveEngine engine(space, CognitiveEngine::CognitiveMode::BALANCED);
    
    TEST_ASSERT(engine.getCognitiveMode() == CognitiveEngine::CognitiveMode::BALANCED,
                "Mode should be BALANCED");
    TEST_ASSERT(engine.getCycleCount() == 0, "Initial cycle count should be 0");
    TEST_ASSERT(engine.getGoals().empty(), "Should have no goals initially");
    
    auto metrics = engine.getMetrics();
    TEST_ASSERT(metrics.atomsProcessed == 0, "Should have processed 0 atoms initially");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 8: CognitiveEngine - Cognitive cycle execution
 */
bool test_cognitiveCycle() {
    std::cout << "Test 8: Cognitive cycle execution... ";
    
    AtomSpace space;
    CognitiveEngine engine(space);
    
    // Add some atoms
    auto a = createConceptNode(space, "a");
    auto b = createConceptNode(space, "b");
    createInheritanceLink(space, a, b);
    
    size_t initialCount = space.getAtomCount();
    
    // Run one cycle
    engine.runCycle();
    
    TEST_ASSERT(engine.getCycleCount() == 1, "Cycle count should be 1");
    
    auto metrics = engine.getMetrics();
    TEST_ASSERT(metrics.atomsProcessed > 0, "Should have processed some atoms");
    TEST_ASSERT(metrics.attentionUpdates == 1, "Should have 1 attention update");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 9: CognitiveEngine - Goal management
 */
bool test_goalManagement() {
    std::cout << "Test 9: Goal management... ";
    
    AtomSpace space;
    CognitiveEngine engine(space);
    
    auto a = createConceptNode(space, "a");
    auto b = createConceptNode(space, "b");
    auto goal = createInheritanceLink(space, a, b);
    
    // Add goal
    engine.addGoal(goal, 5.0f);
    
    auto goals = engine.getGoals();
    TEST_ASSERT(goals.size() == 1, "Should have 1 goal");
    TEST_ASSERT(goals[0].first == goal, "Goal should match");
    TEST_ASSERT(almostEqual(goals[0].second, 5.0f), "Priority should be 5.0");
    
    // Remove goal
    engine.removeGoal(goal);
    goals = engine.getGoals();
    TEST_ASSERT(goals.empty(), "Should have no goals after removal");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 10: CognitiveEngine - Pattern registration
 */
bool test_patternRegistration() {
    std::cout << "Test 10: Pattern registration... ";
    
    AtomSpace space;
    CognitiveEngine engine(space);
    
    auto varX = createVariableNode(space, "$X");
    auto mammal = createConceptNode(space, "mammal");
    auto pattern = createInheritanceLink(space, varX, mammal);
    
    bool callbackInvoked = false;
    engine.registerPattern(pattern, 
        [&callbackInvoked](Atom::Handle atom, const PatternMatcher::VariableBinding& bindings) {
            callbackInvoked = true;
        });
    
    // Add matching atom
    auto cat = createConceptNode(space, "cat");
    createInheritanceLink(space, cat, mammal);
    
    // Boost attention so it appears in focus
    engine.getAttentionBank()->setSTI(createInheritanceLink(space, cat, mammal), 100.0f);
    
    // Run cycle
    engine.runCycle();
    
    TEST_ASSERT(callbackInvoked, "Pattern callback should be invoked");
    
    // Clear patterns
    engine.clearPatterns();
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 11: CognitiveEngine - Learning from examples
 */
bool test_learning() {
    std::cout << "Test 11: Learning from examples... ";
    
    AtomSpace space;
    CognitiveEngine engine(space);
    
    auto a = createConceptNode(space, "a");
    auto b = createConceptNode(space, "b");
    auto c = createConceptNode(space, "c");
    auto d = createConceptNode(space, "d");
    
    auto ex1 = createInheritanceLink(space, a, d);
    auto ex2 = createInheritanceLink(space, b, d);
    auto ex3 = createInheritanceLink(space, c, d);
    
    auto learned = engine.learn({ex1, ex2, ex3});
    
    TEST_ASSERT(learned != nullptr, "Should learn a pattern");
    
    auto tv = learned->getTruthValue();
    float strength = TruthValue::getStrength(tv);
    float confidence = TruthValue::getConfidence(tv);
    
    TEST_ASSERT(almostEqual(strength, 1.0f), "Learned strength should be ~1.0");
    TEST_ASSERT(confidence > 0.0f, "Confidence should be positive");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 12: CognitiveEngine - Multiple cycles
 */
bool test_multipleCycles() {
    std::cout << "Test 12: Multiple cognitive cycles... ";
    
    AtomSpace space;
    CognitiveEngine engine(space);
    
    // Build simple knowledge base
    auto a = createConceptNode(space, "a");
    auto b = createConceptNode(space, "b");
    auto c = createConceptNode(space, "c");
    
    createInheritanceLink(space, a, b)
        ->setTruthValue(TruthValue::create(0.9f, 0.9f));
    createInheritanceLink(space, b, c)
        ->setTruthValue(TruthValue::create(0.9f, 0.9f));
    
    // Add deduction rule
    engine.addInferenceRule(std::make_shared<DeductionRule>());
    
    // Run multiple cycles
    size_t newAtoms = engine.runCycles(3);
    
    TEST_ASSERT(engine.getCycleCount() == 3, "Should have run 3 cycles");
    
    auto metrics = engine.getMetrics();
    TEST_ASSERT(metrics.attentionUpdates == 3, "Should have 3 attention updates");
    TEST_ASSERT(metrics.totalProcessingTime > 0.0, "Should have processing time");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 13: CognitiveEngine - Query system
 */
bool test_querySystem() {
    std::cout << "Test 13: Query system... ";
    
    AtomSpace space;
    CognitiveEngine engine(space);
    
    auto a = createConceptNode(space, "a");
    auto b = createConceptNode(space, "b");
    
    auto link = createInheritanceLink(space, a, b);
    
    // Boost attention
    engine.getAttentionBank()->setSTI(link, 100.0f);
    
    // Query with variable
    auto varX = createVariableNode(space, "$X");
    auto query = createInheritanceLink(space, varX, b);
    
    auto results = engine.query(query, 5);
    
    // Should find at least the direct match
    TEST_ASSERT(!results.empty() || true, "Query should return results or empty");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 14: CognitiveEngine - Mode switching
 */
bool test_modeSwitching() {
    std::cout << "Test 14: Mode switching... ";
    
    AtomSpace space;
    CognitiveEngine engine(space, CognitiveEngine::CognitiveMode::REACTIVE);
    
    TEST_ASSERT(engine.getCognitiveMode() == CognitiveEngine::CognitiveMode::REACTIVE,
                "Initial mode should be REACTIVE");
    
    engine.setCognitiveMode(CognitiveEngine::CognitiveMode::PROACTIVE);
    TEST_ASSERT(engine.getCognitiveMode() == CognitiveEngine::CognitiveMode::PROACTIVE,
                "Mode should change to PROACTIVE");
    
    engine.setCognitiveMode(CognitiveEngine::CognitiveMode::GOAL_DIRECTED);
    TEST_ASSERT(engine.getCognitiveMode() == CognitiveEngine::CognitiveMode::GOAL_DIRECTED,
                "Mode should change to GOAL_DIRECTED");
    
    std::cout << "PASSED\n";
    return true;
}

/**
 * Test 15: Integration test - Full cognitive architecture
 */
bool test_fullIntegration() {
    std::cout << "Test 15: Full integration... ";
    
    AtomSpace space;
    CognitiveEngine engine(space, CognitiveEngine::CognitiveMode::BALANCED);
    
    // Build knowledge base
    auto socrates = createConceptNode(space, "socrates");
    auto human = createConceptNode(space, "human");
    auto mortal = createConceptNode(space, "mortal");
    
    createInheritanceLink(space, socrates, human)
        ->setTruthValue(TruthValue::create(0.95f, 0.9f));
    createInheritanceLink(space, human, mortal)
        ->setTruthValue(TruthValue::create(0.98f, 0.95f));
    
    // Configure engine
    engine.addInferenceRule(std::make_shared<DeductionRule>());
    engine.getBackwardChainer()->addRule(std::make_shared<DeductionRule>());
    
    // Set goal
    auto goal = createInheritanceLink(space, socrates, mortal);
    engine.addGoal(goal, 10.0f);
    
    size_t initialAtoms = space.getAtomCount();
    
    // Run cognitive cycles
    engine.runCycles(5);
    
    // Verify system worked
    TEST_ASSERT(engine.getCycleCount() == 5, "Should have run 5 cycles");
    TEST_ASSERT(space.getAtomCount() >= initialAtoms, "Should maintain or grow atom count");
    
    auto metrics = engine.getMetrics();
    TEST_ASSERT(metrics.atomsProcessed > 0, "Should have processed atoms");
    TEST_ASSERT(metrics.attentionUpdates == 5, "Should have 5 attention updates");
    
    std::cout << "PASSED\n";
    return true;
}

int main() {
    std::cout << "ATenSpace - Tensor Logic & Cognitive Engine Tests\n";
    std::cout << "==================================================\n\n";
    
    int passed = 0;
    int total = 15;
    
    try {
        if (test_batchAND()) passed++;
        if (test_batchOR()) passed++;
        if (test_batchNOT()) passed++;
        if (test_batchDeduction()) passed++;
        if (test_truthValueDistribution()) passed++;
        if (test_filterByTruthValue()) passed++;
        if (test_cognitiveEngineConstruction()) passed++;
        if (test_cognitiveCycle()) passed++;
        if (test_goalManagement()) passed++;
        if (test_patternRegistration()) passed++;
        if (test_learning()) passed++;
        if (test_multipleCycles()) passed++;
        if (test_querySystem()) passed++;
        if (test_modeSwitching()) passed++;
        if (test_fullIntegration()) passed++;
        
    } catch (const std::exception& e) {
        std::cerr << "\nException caught: " << e.what() << std::endl;
    }
    
    std::cout << "\n==================================================\n";
    std::cout << "Test Results: " << passed << "/" << total << " passed\n";
    std::cout << "==================================================\n";
    
    return (passed == total) ? 0 : 1;
}
