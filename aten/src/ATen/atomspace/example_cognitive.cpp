#include <ATen/atomspace/ATenSpace.h>
#include <ATen/atomspace/TensorLogicEngine.h>
#include <ATen/atomspace/CognitiveEngine.h>
#include <iostream>
#include <iomanip>

using namespace at::atomspace;

void printSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void printTruthValue(const Tensor& tv) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[s:" << TruthValue::getStrength(tv) 
              << ", c:" << TruthValue::getConfidence(tv) << "]";
}

/**
 * Example 1: TensorLogicEngine - Batch Logical Operations
 */
void example1_tensorLogicEngine() {
    printSeparator("Example 1: TensorLogicEngine - Batch Operations");
    
    AtomSpace space;
    TensorLogicEngine tensorLogic;
    
    // Create atoms with truth values
    auto a = createConceptNode(space, "socrates");
    a->setTruthValue(TruthValue::create(0.9f, 0.95f));
    
    auto b = createConceptNode(space, "plato");
    b->setTruthValue(TruthValue::create(0.85f, 0.9f));
    
    auto c = createConceptNode(space, "aristotle");
    c->setTruthValue(TruthValue::create(0.95f, 0.92f));
    
    auto d = createConceptNode(space, "human");
    d->setTruthValue(TruthValue::create(0.8f, 0.88f));
    
    std::vector<Atom::Handle> atoms1 = {a, b, c};
    std::vector<Atom::Handle> atoms2 = {b, c, d};
    
    // Batch AND operation
    std::cout << "\nBatch AND operation:\n";
    auto andResults = tensorLogic.batchLogicalOperation(
        atoms1, atoms2, TensorLogicEngine::LogicalOperation::AND);
    
    for (int i = 0; i < andResults.size(0); ++i) {
        std::cout << "  " << atoms1[i]->toString() << " AND " << atoms2[i]->toString() << " = ";
        printTruthValue(andResults[i]);
        std::cout << "\n";
    }
    
    // Batch OR operation
    std::cout << "\nBatch OR operation:\n";
    auto orResults = tensorLogic.batchLogicalOperation(
        atoms1, atoms2, TensorLogicEngine::LogicalOperation::OR);
    
    for (int i = 0; i < orResults.size(0); ++i) {
        std::cout << "  " << atoms1[i]->toString() << " OR " << atoms2[i]->toString() << " = ";
        printTruthValue(orResults[i]);
        std::cout << "\n";
    }
    
    // Batch NOT operation
    std::cout << "\nBatch NOT operation:\n";
    auto notResults = tensorLogic.batchUnaryOperation(
        atoms1, TensorLogicEngine::LogicalOperation::NOT);
    
    for (int i = 0; i < notResults.size(0); ++i) {
        std::cout << "  NOT " << atoms1[i]->toString() << " = ";
        printTruthValue(notResults[i]);
        std::cout << "\n";
    }
    
    // Truth value distribution
    std::cout << "\nTruth value distribution across all atoms:\n";
    std::vector<Atom::Handle> allAtoms = {a, b, c, d};
    auto dist = tensorLogic.computeTruthValueDistribution(allAtoms);
    std::cout << "  Mean strength: " << dist[0].item<float>() << "\n";
    std::cout << "  Mean confidence: " << dist[1].item<float>() << "\n";
    std::cout << "  Var strength: " << dist[2].item<float>() << "\n";
    std::cout << "  Var confidence: " << dist[3].item<float>() << "\n";
}

/**
 * Example 2: TensorLogicEngine - Batch Deduction
 */
void example2_batchDeduction() {
    printSeparator("Example 2: Batch Deduction Inference");
    
    AtomSpace space;
    TensorLogicEngine tensorLogic;
    
    // Create inheritance hierarchy
    auto socrates = createConceptNode(space, "socrates");
    auto plato = createConceptNode(space, "plato");
    auto aristotle = createConceptNode(space, "aristotle");
    auto human = createConceptNode(space, "human");
    auto mortal = createConceptNode(space, "mortal");
    auto animal = createConceptNode(space, "animal");
    
    // Create implications with truth values
    auto link1 = createInheritanceLink(space, socrates, human);
    link1->setTruthValue(TruthValue::create(0.95f, 0.9f));
    
    auto link2 = createInheritanceLink(space, plato, human);
    link2->setTruthValue(TruthValue::create(0.93f, 0.88f));
    
    auto link3 = createInheritanceLink(space, aristotle, human);
    link3->setTruthValue(TruthValue::create(0.94f, 0.91f));
    
    auto link4 = createInheritanceLink(space, human, mortal);
    link4->setTruthValue(TruthValue::create(0.99f, 0.95f));
    
    auto link5 = createInheritanceLink(space, human, animal);
    link5->setTruthValue(TruthValue::create(0.98f, 0.96f));
    
    auto link6 = createInheritanceLink(space, mortal, animal);
    link6->setTruthValue(TruthValue::create(0.85f, 0.8f));
    
    // Batch deduction
    std::vector<Atom::Handle> premises1 = {link1, link2, link3};
    std::vector<Atom::Handle> premises2 = {link4, link4, link4};
    
    std::cout << "\nBatch deduction: (X→human, human→mortal) ⊢ X→mortal\n";
    auto deduced = tensorLogic.batchDeduction(premises1, premises2);
    
    for (int i = 0; i < deduced.size(0); ++i) {
        std::cout << "  ";
        auto link = static_cast<const Link*>(premises1[i].get());
        std::cout << link->getOutgoingAtom(0)->toString() << "→mortal = ";
        printTruthValue(deduced[i]);
        std::cout << "\n";
    }
}

/**
 * Example 3: CognitiveEngine - Basic Cognitive Cycle
 */
void example3_cognitiveCycle() {
    printSeparator("Example 3: CognitiveEngine - Cognitive Cycle");
    
    AtomSpace space;
    CognitiveEngine engine(space, CognitiveEngine::CognitiveMode::BALANCED);
    
    // Build knowledge base
    auto socrates = createConceptNode(space, "socrates");
    auto human = createConceptNode(space, "human");
    auto mortal = createConceptNode(space, "mortal");
    auto animal = createConceptNode(space, "animal");
    
    auto link1 = createInheritanceLink(space, socrates, human);
    link1->setTruthValue(TruthValue::create(0.95f, 0.9f));
    
    auto link2 = createInheritanceLink(space, human, mortal);
    link2->setTruthValue(TruthValue::create(0.98f, 0.95f));
    
    auto link3 = createInheritanceLink(space, human, animal);
    link3->setTruthValue(TruthValue::create(0.99f, 0.96f));
    
    // Add deduction rule
    engine.addInferenceRule(std::make_shared<DeductionRule>());
    
    std::cout << "\nInitial atom count: " << space.getAtomCount() << "\n";
    
    // Run cognitive cycles
    std::cout << "\nRunning 3 cognitive cycles...\n";
    size_t newAtoms = engine.runCycles(3);
    
    std::cout << "\nFinal atom count: " << space.getAtomCount() << "\n";
    std::cout << "New atoms created: " << newAtoms << "\n";
    
    // Print metrics
    auto metrics = engine.getMetrics();
    std::cout << "\nCognitive Metrics:\n";
    std::cout << "  Atoms processed: " << metrics.atomsProcessed << "\n";
    std::cout << "  Inferences performed: " << metrics.inferencesPerformed << "\n";
    std::cout << "  Attention updates: " << metrics.attentionUpdates << "\n";
    std::cout << "  Total processing time: " << metrics.totalProcessingTime << "s\n";
    std::cout << "  New knowledge: " << metrics.newKnowledgeGenerated << " atoms\n";
}

/**
 * Example 4: CognitiveEngine - Goal-Directed Reasoning
 */
void example4_goalDirected() {
    printSeparator("Example 4: Goal-Directed Reasoning");
    
    AtomSpace space;
    CognitiveEngine engine(space, CognitiveEngine::CognitiveMode::GOAL_DIRECTED);
    
    // Build knowledge base
    auto socrates = createConceptNode(space, "socrates");
    auto human = createConceptNode(space, "human");
    auto mortal = createConceptNode(space, "mortal");
    
    createInheritanceLink(space, socrates, human)
        ->setTruthValue(TruthValue::create(0.95f, 0.9f));
    createInheritanceLink(space, human, mortal)
        ->setTruthValue(TruthValue::create(0.98f, 0.95f));
    
    // Add rules
    engine.addInferenceRule(std::make_shared<DeductionRule>());
    engine.getBackwardChainer()->addRule(std::make_shared<DeductionRule>());
    
    // Set goal: prove that socrates is mortal
    auto goal = createInheritanceLink(space, socrates, mortal);
    engine.addGoal(goal, 10.0f);
    
    std::cout << "\nGoal: Prove that " << socrates->toString() 
              << " → " << mortal->toString() << "\n";
    std::cout << "\nRunning goal-directed reasoning...\n";
    
    engine.runCycles(5);
    
    // Check if goal was achieved
    auto goals = engine.getGoals();
    if (goals.empty() || goals[0].first != goal) {
        std::cout << "\n✓ Goal achieved! Socrates is proven to be mortal.\n";
    } else {
        std::cout << "\n✗ Goal not yet achieved, continuing...\n";
    }
    
    auto metrics = engine.getMetrics();
    std::cout << "\nInferences performed: " << metrics.inferencesPerformed << "\n";
}

/**
 * Example 5: CognitiveEngine - Pattern Recognition
 */
void example5_patternRecognition() {
    printSeparator("Example 5: Pattern Recognition with Cognitive Engine");
    
    AtomSpace space;
    CognitiveEngine engine(space);
    
    // Create knowledge base with pattern
    auto cat = createConceptNode(space, "cat");
    auto dog = createConceptNode(space, "dog");
    auto fish = createConceptNode(space, "fish");
    auto mammal = createConceptNode(space, "mammal");
    auto animal = createConceptNode(space, "animal");
    
    createInheritanceLink(space, cat, mammal);
    createInheritanceLink(space, dog, mammal);
    createInheritanceLink(space, mammal, animal);
    
    // Register pattern: X → mammal
    auto varX = createVariableNode(space, "$X");
    auto pattern = createInheritanceLink(space, varX, mammal);
    
    int matchCount = 0;
    engine.registerPattern(pattern, 
        [&matchCount](Atom::Handle atom, const PatternMatcher::VariableBinding& bindings) {
            std::cout << "  Pattern matched: " << atom->toString() << "\n";
            for (const auto& [var, value] : bindings) {
                std::cout << "    " << var->toString() << " = " << value->toString() << "\n";
            }
            matchCount++;
        });
    
    std::cout << "\nRegistered pattern: $X → mammal\n";
    std::cout << "\nRunning cognitive cycle with pattern matching...\n";
    
    engine.runCycle();
    
    std::cout << "\nTotal patterns matched: " << matchCount << "\n";
}

/**
 * Example 6: CognitiveEngine - Learning from Examples
 */
void example6_learning() {
    printSeparator("Example 6: Learning from Examples");
    
    AtomSpace space;
    CognitiveEngine engine(space);
    
    // Create positive examples: birds that fly
    auto sparrow = createConceptNode(space, "sparrow");
    auto eagle = createConceptNode(space, "eagle");
    auto robin = createConceptNode(space, "robin");
    auto bird = createConceptNode(space, "bird");
    auto flies = createConceptNode(space, "flies");
    
    auto ex1 = createInheritanceLink(space, sparrow, bird);
    auto ex2 = createInheritanceLink(space, eagle, bird);
    auto ex3 = createInheritanceLink(space, robin, bird);
    
    // Create negative examples: birds that don't fly
    auto penguin = createConceptNode(space, "penguin");
    auto ostrich = createConceptNode(space, "ostrich");
    
    auto neg1 = createInheritanceLink(space, penguin, bird);
    auto neg2 = createInheritanceLink(space, ostrich, bird);
    
    std::cout << "\nPositive examples (birds that fly): 3\n";
    std::cout << "Negative examples (birds that don't fly): 2\n";
    
    // Learn pattern
    auto learnedPattern = engine.learn(
        {ex1, ex2, ex3},
        {neg1, neg2}
    );
    
    if (learnedPattern) {
        std::cout << "\nLearned pattern: " << learnedPattern->toString() << "\n";
        std::cout << "Truth value: ";
        printTruthValue(learnedPattern->getTruthValue());
        std::cout << "\n";
        
        // Check attention
        auto sti = engine.getAttentionBank()->getSTI(learnedPattern);
        std::cout << "Attention (STI): " << sti << "\n";
    }
}

/**
 * Example 7: Full Integration - Knowledge Discovery System
 */
void example7_fullIntegration() {
    printSeparator("Example 7: Full Integration - Knowledge Discovery");
    
    AtomSpace space;
    CognitiveEngine engine(space, CognitiveEngine::CognitiveMode::PROACTIVE);
    
    std::cout << "\nBuilding comprehensive knowledge base...\n";
    
    // Build rich knowledge base
    auto socrates = createConceptNode(space, "socrates");
    auto plato = createConceptNode(space, "plato");
    auto aristotle = createConceptNode(space, "aristotle");
    auto human = createConceptNode(space, "human");
    auto philosopher = createConceptNode(space, "philosopher");
    auto mortal = createConceptNode(space, "mortal");
    auto animal = createConceptNode(space, "animal");
    auto livingThing = createConceptNode(space, "living_thing");
    
    // Create knowledge graph
    createInheritanceLink(space, socrates, philosopher)
        ->setTruthValue(TruthValue::create(0.99f, 0.95f));
    createInheritanceLink(space, plato, philosopher)
        ->setTruthValue(TruthValue::create(0.99f, 0.95f));
    createInheritanceLink(space, aristotle, philosopher)
        ->setTruthValue(TruthValue::create(0.99f, 0.95f));
    createInheritanceLink(space, philosopher, human)
        ->setTruthValue(TruthValue::create(0.95f, 0.9f));
    createInheritanceLink(space, human, mortal)
        ->setTruthValue(TruthValue::create(0.99f, 0.98f));
    createInheritanceLink(space, human, animal)
        ->setTruthValue(TruthValue::create(0.98f, 0.97f));
    createInheritanceLink(space, animal, livingThing)
        ->setTruthValue(TruthValue::create(0.99f, 0.99f));
    createInheritanceLink(space, mortal, livingThing)
        ->setTruthValue(TruthValue::create(0.95f, 0.9f));
    
    // Add inference rules
    engine.addInferenceRule(std::make_shared<DeductionRule>());
    engine.getBackwardChainer()->addRule(std::make_shared<DeductionRule>());
    
    std::cout << "Initial atoms: " << space.getAtomCount() << "\n";
    
    // Add discovery goals
    auto goal1 = createInheritanceLink(space, socrates, livingThing);
    engine.addGoal(goal1, 5.0f);
    
    std::cout << "\nGoal: Discover that socrates → living_thing\n";
    std::cout << "\nRunning cognitive engine in proactive mode...\n";
    
    // Run multiple cycles
    for (int i = 1; i <= 5; ++i) {
        std::cout << "\n--- Cycle " << i << " ---\n";
        size_t newAtoms = engine.runCycle();
        std::cout << "New knowledge: " << newAtoms << " atoms\n";
        
        auto focus = engine.getAttentionBank()->getAttentionalFocus();
        std::cout << "Attentional focus: " << focus.size() << " atoms\n";
    }
    
    std::cout << "\nFinal atoms: " << space.getAtomCount() << "\n";
    
    // Print final metrics
    auto metrics = engine.getMetrics();
    std::cout << "\n=== Final Cognitive Metrics ===\n";
    std::cout << "Cycles run: " << engine.getCycleCount() << "\n";
    std::cout << "Total atoms processed: " << metrics.atomsProcessed << "\n";
    std::cout << "Inferences performed: " << metrics.inferencesPerformed << "\n";
    std::cout << "Patterns matched: " << metrics.patternsMatched << "\n";
    std::cout << "Attention updates: " << metrics.attentionUpdates << "\n";
    std::cout << "New knowledge generated: " << metrics.newKnowledgeGenerated << " atoms\n";
    std::cout << "Total processing time: " << metrics.totalProcessingTime << "s\n";
    std::cout << "Avg time per cycle: " 
              << (metrics.totalProcessingTime / engine.getCycleCount()) << "s\n";
    
    // Query the system
    std::cout << "\n=== Query: Is socrates a living_thing? ===\n";
    auto queryResults = engine.query(goal1, 10);
    
    if (!queryResults.empty()) {
        std::cout << "✓ Yes! Found " << queryResults.size() << " proof(s)\n";
        for (const auto& [atom, bindings] : queryResults) {
            std::cout << "  Result: " << atom->toString() << "\n";
        }
    } else {
        std::cout << "✗ Could not prove it yet\n";
    }
}

int main() {
    std::cout << "ATenSpace - Tensor Logic Engine & Cognitive Engine Examples\n";
    std::cout << "=============================================================\n";
    
    try {
        example1_tensorLogicEngine();
        example2_batchDeduction();
        example3_cognitiveCycle();
        example4_goalDirected();
        example5_patternRecognition();
        example6_learning();
        example7_fullIntegration();
        
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "All examples completed successfully!\n";
        std::cout << std::string(60, '=') << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
