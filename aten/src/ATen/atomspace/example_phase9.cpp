/**
 * example_phase9.cpp - Phase 9 Feature Showcase
 *
 * Demonstrates:
 *   1. QueryEngine / QueryBuilder  - multi-pattern conjunctive queries
 *   2. BinarySerializer            - fast binary persistence
 *   3. InferencePipeline           - composable reasoning steps
 *   4. HebbianLearner              - associative learning
 */
#include "ATenSpaceCore.h"
#include "QueryEngine.h"
#include "BinarySerializer.h"
#include "InferencePipeline.h"
#include "HebbianLearner.h"
#include "AttentionBank.h"
#include <iostream>
#include <iomanip>

using namespace at::atomspace;

// ======================================================================== //
//  1. QueryEngine - Multi-Pattern Conjunctive Query
// ======================================================================== //

void demoQueryEngine() {
    std::cout << "\n============================================\n";
    std::cout << "  Demo 1: QueryEngine / QueryBuilder\n";
    std::cout << "============================================\n\n";

    AtomSpace space;

    // Build a small animal knowledge graph
    auto animal  = createConceptNode(space, "Animal");
    auto mammal  = createConceptNode(space, "Mammal");
    auto bird    = createConceptNode(space, "Bird");
    auto dog     = createConceptNode(space, "Dog");
    auto cat     = createConceptNode(space, "Cat");
    auto eagle   = createConceptNode(space, "Eagle");
    auto canFly  = createConceptNode(space, "CanFly");
    auto hasLegs = createConceptNode(space, "HasLegs");

    // Truth values
    dog->setTruthValue(TruthValue::create(0.95f, 0.99f));
    cat->setTruthValue(TruthValue::create(0.90f, 0.99f));
    eagle->setTruthValue(TruthValue::create(0.85f, 0.95f));

    // Taxonomy
    createInheritanceLink(space, mammal, animal);
    createInheritanceLink(space, bird,   animal);
    createInheritanceLink(space, dog,    mammal);
    createInheritanceLink(space, cat,    mammal);
    createInheritanceLink(space, eagle,  bird);

    // Properties
    createInheritanceLink(space, eagle, canFly);
    createInheritanceLink(space, dog,   hasLegs);
    createInheritanceLink(space, cat,   hasLegs);
    createInheritanceLink(space, eagle, hasLegs);

    std::cout << "Knowledge graph: " << space.size() << " atoms\n\n";

    // --- Query 1: All mammals ---
    auto varX   = space.addNode(Atom::Type::VARIABLE_NODE, "?X");
    auto mammalPat = createInheritanceLink(space, varX, mammal);

    QueryEngine qe(space);
    auto mammals = qe.findMatches(mammalPat);
    std::cout << "Query: ?X inherits Mammal\n";
    for (const auto& row : mammals) {
        auto it = row.find(varX);
        if (it != row.end()) {
            const auto* n = static_cast<const Node*>(it->second.get());
            std::cout << "  -> " << n->getName() << "\n";
        }
    }

    // --- Query 2: Animals that have legs ---
    auto varY       = space.addNode(Atom::Type::VARIABLE_NODE, "?Y");
    auto animalPat  = createInheritanceLink(space, varY, animal);
    auto legsPat    = createInheritanceLink(space, varY, hasLegs);

    auto results = QueryBuilder(space)
                       .match(animalPat)
                       .match(legsPat)
                       .execute();

    std::cout << "\nQuery: ?Y inherits Animal AND ?Y inherits HasLegs\n";
    for (const auto& row : results) {
        auto it = row.find(varY);
        if (it != row.end()) {
            const auto* n = static_cast<const Node*>(it->second.get());
            std::cout << "  -> " << n->getName() << "\n";
        }
    }

    // --- Query 3: Semantic similarity search ---
    auto embedding = torch::randn({64});
    dog->setTruthValue(TruthValue::create(0.9f, 0.9f));

    // Add embeddings to concept nodes
    space.addNode(Atom::Type::CONCEPT_NODE, "Dog",   torch::randn({64}));
    space.addNode(Atom::Type::CONCEPT_NODE, "Cat",   torch::randn({64}));
    space.addNode(Atom::Type::CONCEPT_NODE, "Eagle", torch::randn({64}));

    auto similar = qe.findSimilar(embedding, 3, -1.0f);
    std::cout << "\nTop-3 semantically similar to query vector:\n";
    for (const auto& [atom, sim] : similar) {
        const auto* n = static_cast<const Node*>(atom.get());
        std::cout << "  " << n->getName()
                  << " (sim=" << std::fixed << std::setprecision(3) << sim << ")\n";
    }

    std::cout << "\n[QueryEngine demo complete]\n";
}

// ======================================================================== //
//  2. BinarySerializer
// ======================================================================== //

void demoBinarySerializer() {
    std::cout << "\n============================================\n";
    std::cout << "  Demo 2: BinarySerializer\n";
    std::cout << "============================================\n\n";

    AtomSpace original;

    // Add rich content with embeddings and truth values
    for (int i = 0; i < 20; ++i) {
        auto n = createConceptNode(original, "concept_" + std::to_string(i));
        n->setTruthValue(TruthValue::create(
            0.5f + 0.02f * i, 0.8f));
    }
    // Node with embedding
    auto main = original.addNode(
        Atom::Type::CONCEPT_NODE, "main-concept",
        torch::randn({128}));

    // Some links
    auto a = createConceptNode(original, "node-A");
    auto b = createConceptNode(original, "node-B");
    auto c = createConceptNode(original, "node-C");
    createInheritanceLink(original, a, b);
    createInheritanceLink(original, b, c);
    createListLink(original, {a, b, c});

    std::cout << "Original AtomSpace: " << original.size() << " atoms\n";

    // Serialize to memory
    auto t0 = std::chrono::steady_clock::now();
    auto bytes = BinarySerializer::serialize(original);
    auto t1 = std::chrono::steady_clock::now();

    double serMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Serialized " << bytes.size() << " bytes in "
              << std::fixed << std::setprecision(2) << serMs << " ms\n";

    // Deserialize
    auto t2 = std::chrono::steady_clock::now();
    AtomSpace restored;
    BinarySerializer::deserialize(restored, bytes);
    auto t3 = std::chrono::steady_clock::now();

    double deserMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "Deserialized " << restored.size() << " atoms in "
              << deserMs << " ms\n";

    // Verify
    auto rmain = restored.getNode(Atom::Type::CONCEPT_NODE, "main-concept");
    bool hasEmb = false;
    if (rmain) {
        const auto* nptr = static_cast<const Node*>(rmain.get());
        hasEmb = nptr->hasEmbedding();
    }
    std::cout << "main-concept embedding restored: "
              << (hasEmb ? "YES" : "NO") << "\n";

    // Save/load file round-trip
    std::string path = "/tmp/atenspace_demo_phase9.bin";
    bool saved = BinarySerializer::save(original, path);
    AtomSpace fromFile;
    bool loaded = BinarySerializer::load(fromFile, path);
    std::cout << "File round-trip (" << path << "): "
              << (saved && loaded && fromFile.size() == original.size()
                  ? "OK" : "FAIL") << "\n";

    std::cout << "\n[BinarySerializer demo complete]\n";
}

// ======================================================================== //
//  3. InferencePipeline
// ======================================================================== //

void demoInferencePipeline() {
    std::cout << "\n============================================\n";
    std::cout << "  Demo 3: InferencePipeline\n";
    std::cout << "============================================\n\n";

    AtomSpace space;

    // Build a simple ontology
    auto thing   = createConceptNode(space, "Thing");
    auto living  = createConceptNode(space, "LivingThing");
    auto plant   = createConceptNode(space, "Plant");
    auto animal  = createConceptNode(space, "Animal");
    auto dog     = createConceptNode(space, "Dog");
    auto fido    = createConceptNode(space, "Fido");

    // High confidence facts
    auto setTV = [](Atom::Handle a, float s, float c) {
        a->setTruthValue(TruthValue::create(s, c));
    };
    setTV(thing,  1.0f, 1.0f);
    setTV(living, 0.9f, 0.9f);
    setTV(animal, 0.9f, 0.9f);
    setTV(dog,    0.85f, 0.9f);
    setTV(fido,   0.7f, 0.8f);

    createInheritanceLink(space, living, thing);
    createInheritanceLink(space, plant,  living);
    createInheritanceLink(space, animal, living);
    createInheritanceLink(space, dog,    animal);
    createInheritanceLink(space, fido,   dog);

    std::cout << "Initial AtomSpace: " << space.size() << " atoms\n";

    // Build a pipeline: match animals, forward chain, filter by TV
    auto varX       = space.addNode(Atom::Type::VARIABLE_NODE, "?Pipeline-X");
    auto animalPat  = createInheritanceLink(space, varX, animal);

    InferencePipeline pipeline(space);
    pipeline
        .matchPattern(animalPat)          // Find animals
        .forwardChain(2)                  // Derive new facts
        .filterByTV(0.6f, 0.5f)          // Keep only confident facts
        .filter("nodes-only",
                [](const Atom::Handle& a) { return a->isNode(); });

    std::cout << "\nPipeline steps: ";
    for (const auto& name : pipeline.stepNames()) {
        std::cout << "[" << name << "] ";
    }
    std::cout << "\n";

    auto result = pipeline.run();

    std::cout << "Pipeline results: " << result.atoms.size()
              << " atoms in working set\n";
    std::cout << "Iterations: " << result.iterationsRun
              << ", Converged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(2)
              << result.totalMs() << " ms\n\n";

    std::cout << "Step stats:\n";
    for (const auto& s : result.stats) {
        std::cout << "  " << std::left << std::setw(30) << s.stepName
                  << " produced=" << (s.produced ? "yes" : "no ")
                  << "  set=" << s.workingSetSize
                  << "  " << std::fixed << std::setprecision(2)
                  << s.elapsedMs << "ms\n";
    }

    std::cout << "\n[InferencePipeline demo complete]\n";
}

// ======================================================================== //
//  4. HebbianLearner
// ======================================================================== //

void demoHebbianLearner() {
    std::cout << "\n============================================\n";
    std::cout << "  Demo 4: HebbianLearner\n";
    std::cout << "============================================\n\n";

    AtomSpace space;
    AttentionBank bank;

    // Create concept nodes
    auto coffee  = createConceptNode(space, "Coffee");
    auto morning = createConceptNode(space, "Morning");
    auto work    = createConceptNode(space, "Work");
    auto tired   = createConceptNode(space, "Tired");
    auto energy  = createConceptNode(space, "Energy");

    // Configure learner
    HebbianLearner::Config cfg;
    cfg.learningRate = 0.15f;
    cfg.decayRate    = 0.02f;
    cfg.ojaRule      = false;
    HebbianLearner learner(space, bank, cfg);

    // Simulate co-activations:
    // Coffee + Morning appear together often
    std::cout << "Simulating 20 coffee-morning co-activations...\n";
    for (int i = 0; i < 20; ++i) {
        learner.recordCoActivation(coffee, morning);
    }

    // Coffee + Energy appear together sometimes
    std::cout << "Simulating 8 coffee-energy co-activations...\n";
    for (int i = 0; i < 8; ++i) {
        learner.recordCoActivation(coffee, energy);
    }

    // Work + Tired appear together rarely
    std::cout << "Simulating 3 work-tired co-activations...\n";
    for (int i = 0; i < 3; ++i) {
        learner.recordCoActivation(work, tired);
    }

    std::cout << "\nTotal co-activations: "
              << learner.totalCoActivations() << "\n";

    // Query associations for coffee
    auto associates = learner.getAssociates(coffee);
    std::cout << "\nAssociates of 'Coffee' (sorted by strength):\n";
    for (const auto& [atom, strength] : associates) {
        const auto* n = static_cast<const Node*>(atom.get());
        std::cout << "  " << std::left << std::setw(12) << n->getName()
                  << " strength=" << std::fixed << std::setprecision(4)
                  << strength << "\n";
    }

    // Simulate attentional focus learning
    bank.setAttentionValue(coffee,  AttentionBank::AttentionValue(100.0f, 0.0f, 0.0f));
    bank.setAttentionValue(morning, AttentionBank::AttentionValue(90.0f,  0.0f, 0.0f));
    bank.setAttentionValue(work,    AttentionBank::AttentionValue(60.0f,  0.0f, 0.0f));

    std::cout << "\nRunning 5 Hebbian learning cycles from attentional focus...\n";
    learner.runCycles(5);

    std::cout << "Hebbian links in space: "
              << learner.getAllHebbianLinks().size() << "\n";

    std::cout << "\nFinal coffee-morning strength: "
              << std::fixed << std::setprecision(4)
              << learner.getStrength(coffee, morning) << "\n";
    std::cout << "Final coffee-energy  strength: "
              << learner.getStrength(coffee, energy) << "\n";
    std::cout << "Final work-tired     strength: "
              << learner.getStrength(work, tired) << "\n";

    std::cout << "\n[HebbianLearner demo complete]\n";
}

// ======================================================================== //
//  Main
// ======================================================================== //

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  ATenSpace Phase 9 Feature Showcase      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    demoQueryEngine();
    demoBinarySerializer();
    demoInferencePipeline();
    demoHebbianLearner();

    std::cout << "\n✓ All Phase 9 demos completed successfully.\n";
    return 0;
}
