/**
 * benchmark.cpp - Systematic performance benchmarks for ATenSpace (Phase 10)
 *
 * Measures wall-clock time for core operations:
 *   1. Atom creation (nodes and links)
 *   2. Pattern matching (single-clause and conjunctive)
 *   3. Embedding similarity search
 *   4. Forward chaining
 *   5. Binary serialization / deserialization
 *   6. Hebbian learning cycles
 *   7. QueryEngine operations
 *   8. AtomSpace clear / reload round-trip
 *
 * Usage:
 *   ./atomspace_benchmark [--iterations N]
 */

#include "ATenSpaceCore.h"
#include "QueryEngine.h"
#include "BinarySerializer.h"
#include "InferencePipeline.h"
#include "HebbianLearner.h"
#include "AttentionBank.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <numeric>
#include <algorithm>

using namespace at::atomspace;
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ============================================================
// Benchmark harness
// ============================================================

struct BenchResult {
    std::string name;
    int         iterations;
    double      totalMs;
    double      minMs;
    double      maxMs;

    double meanMs()   const { return totalMs / iterations; }
    double ops_per_s()const { return 1000.0 / meanMs(); }
};

BenchResult runBenchmark(const std::string& name,
                         int iterations,
                         std::function<void()> setup,
                         std::function<void()> fn) {
    setup();
    // Warmup
    for (int i = 0; i < std::min(5, iterations / 10 + 1); ++i) fn();

    std::vector<double> times;
    times.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times.push_back(Ms(t1 - t0).count());
    }

    BenchResult r;
    r.name       = name;
    r.iterations = iterations;
    r.totalMs    = std::accumulate(times.begin(), times.end(), 0.0);
    r.minMs      = *std::min_element(times.begin(), times.end());
    r.maxMs      = *std::max_element(times.begin(), times.end());
    return r;
}

void printHeader() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          ATenSpace Phase 10 — Performance Benchmark Suite           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";
    std::cout << std::left
              << std::setw(42) << "Benchmark"
              << std::right
              << std::setw(8)  << "Iter"
              << std::setw(12) << "Mean(ms)"
              << std::setw(12) << "Min(ms)"
              << std::setw(12) << "Max(ms)"
              << std::setw(14) << "Ops/sec"
              << "\n";
    std::cout << std::string(100, '-') << "\n";
}

void printResult(const BenchResult& r) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left
              << std::setw(42) << r.name
              << std::right
              << std::setw(8)  << r.iterations
              << std::setw(12) << r.meanMs()
              << std::setw(12) << r.minMs
              << std::setw(12) << r.maxMs
              << std::setw(14) << static_cast<long long>(r.ops_per_s())
              << "\n";
}

// ============================================================
// Individual benchmarks
// ============================================================

BenchResult benchAtomCreation(int N) {
    AtomSpace space;
    int counter = 0;
    return runBenchmark(
        "Atom creation (ConceptNode)",
        N,
        [&] { space.clear(); counter = 0; },
        [&] {
            createConceptNode(space, "node-" + std::to_string(counter++));
        });
}

BenchResult benchLinkCreation(int N) {
    AtomSpace space;
    auto a = createConceptNode(space, "A");
    auto b = createConceptNode(space, "B");
    int counter = 0;
    return runBenchmark(
        "Link creation (InheritanceLink)",
        N,
        [&] { /* reuse a, b */ counter = 0; },
        [&] {
            // Create unique links by using slightly varied targets
            space.addLink(Atom::Type::INHERITANCE_LINK, {a, b});
            ++counter;
        });
}

BenchResult benchAtomLookup(int N) {
    AtomSpace space;
    for (int i = 0; i < 1000; ++i) {
        createConceptNode(space, "lookup-" + std::to_string(i));
    }
    return runBenchmark(
        "Atom lookup (getNode)",
        N,
        [] {},
        [&] {
            space.getNode(Atom::Type::CONCEPT_NODE, "lookup-500");
        });
}

BenchResult benchPatternMatch(int N) {
    AtomSpace space;
    // Build 100-node taxonomy
    auto thing = createConceptNode(space, "thing");
    for (int i = 0; i < 100; ++i) {
        auto entity = createConceptNode(space, "entity-" + std::to_string(i));
        createInheritanceLink(space, entity, thing);
    }
    auto varX = createVariableNode(space, "?X");
    auto pattern = space.addLink(Atom::Type::INHERITANCE_LINK, {varX, thing});

    return runBenchmark(
        "Pattern match (single clause, 100 links)",
        N,
        [] {},
        [&] {
            QueryEngine qe(space);
            auto results = qe.findMatches(pattern);
            (void)results;
        });
}

BenchResult benchConjunctiveQuery(int N) {
    AtomSpace space;
    auto mammal  = createConceptNode(space, "bm-mammal");
    auto hasLegs = createConceptNode(space, "bm-has-legs");
    for (int i = 0; i < 50; ++i) {
        auto e = createConceptNode(space, "bm-e-" + std::to_string(i));
        createInheritanceLink(space, e, mammal);
        if (i % 2 == 0) {
            createInheritanceLink(space, e, hasLegs);
        }
    }

    auto varX = createVariableNode(space, "?BX");
    auto p1 = space.addLink(Atom::Type::INHERITANCE_LINK, {varX, mammal});
    auto p2 = space.addLink(Atom::Type::INHERITANCE_LINK, {varX, hasLegs});

    return runBenchmark(
        "Conjunctive query (2 clauses, 50+25 links)",
        N,
        [] {},
        [&] {
            auto results = QueryBuilder(space)
                .match(p1)
                .match(p2)
                .execute();
            (void)results;
        });
}

BenchResult benchEmbeddingSimilarity(int N) {
    AtomSpace space;
    // Create 200 nodes with 128-dim embeddings via addNode(type, name, embedding)
    for (int i = 0; i < 200; ++i) {
        space.addNode(Atom::Type::CONCEPT_NODE,
                      "emb-" + std::to_string(i),
                      torch::randn({128}));
    }
    auto query = torch::randn({128});

    return runBenchmark(
        "Embedding similarity search (k=10, n=200)",
        N,
        [] {},
        [&] {
            QueryEngine qe(space);
            auto results = qe.findSimilar(query, 10);
            (void)results;
        });
}

BenchResult benchForwardChaining(int N) {
    AtomSpace space;
    auto animal = createConceptNode(space, "fc-animal");
    auto mammal = createConceptNode(space, "fc-mammal");
    auto dog    = createConceptNode(space, "fc-dog");

    auto l1 = createInheritanceLink(space, mammal, animal);
    auto l2 = createInheritanceLink(space, dog, mammal);
    l1->setTruthValue(torch::tensor({0.9f, 0.9f}));
    l2->setTruthValue(torch::tensor({0.95f, 0.9f}));

    ForwardChainer fc(space);

    return runBenchmark(
        "Forward chaining (3-node chain, 3 rounds)",
        N,
        [] {},
        [&] {
            fc.setMaxIterations(3);
            fc.run();
        });
}

BenchResult benchBinarySerialization(int N) {
    AtomSpace space;
    for (int i = 0; i < 100; ++i) {
        auto a = createConceptNode(space, "ser-a-" + std::to_string(i));
        auto b = createConceptNode(space, "ser-b-" + std::to_string(i));
        createInheritanceLink(space, a, b);
    }

    return runBenchmark(
        "Binary serialization (100 nodes + 100 links)",
        N,
        [] {},
        [&] {
            auto buf = BinarySerializer::serialize(space);
            (void)buf;
        });
}

BenchResult benchBinaryDeserialization(int N) {
    AtomSpace src;
    for (int i = 0; i < 100; ++i) {
        auto a = createConceptNode(src, "ds-a-" + std::to_string(i));
        auto b = createConceptNode(src, "ds-b-" + std::to_string(i));
        createInheritanceLink(src, a, b);
    }
    auto buf = BinarySerializer::serialize(src);

    AtomSpace dst;
    return runBenchmark(
        "Binary deserialization (100 nodes + 100 links)",
        N,
        [&] { dst.clear(); },
        [&] {
            BinarySerializer::deserialize(dst, buf);
        });
}

BenchResult benchHebbianLearning(int N) {
    AtomSpace space;
    AttentionBank bank;
    HebbianLearner learner(space, bank);

    auto a = createConceptNode(space, "hl-a");
    auto b = createConceptNode(space, "hl-b");

    return runBenchmark(
        "Hebbian learning (1 co-activation)",
        N,
        [] {},
        [&] {
            learner.recordCoActivation(a, b);
        });
}

BenchResult benchHebbianDecay(int N) {
    AtomSpace space;
    AttentionBank bank;
    HebbianLearner learner(space, bank);

    // Pre-populate with 50 pairs
    for (int i = 0; i < 50; ++i) {
        auto a = createConceptNode(space, "hd-a-" + std::to_string(i));
        auto b = createConceptNode(space, "hd-b-" + std::to_string(i));
        for (int j = 0; j < 5; ++j) learner.recordCoActivation(a, b);
    }

    return runBenchmark(
        "Hebbian decay (50 links)",
        N,
        [] {},
        [&] {
            learner.decay();
        });
}

BenchResult benchAtomSpaceClear(int N) {
    AtomSpace space;
    return runBenchmark(
        "AtomSpace clear (1000 atoms)",
        N,
        [&] {
            space.clear();
            for (int i = 0; i < 1000; ++i) {
                createConceptNode(space, "clr-" + std::to_string(i));
            }
        },
        [&] { space.clear(); });
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    int iterations = 1000;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        }
    }

    printHeader();

    std::vector<BenchResult> results = {
        benchAtomCreation(iterations),
        benchLinkCreation(iterations),
        benchAtomLookup(iterations),
        benchPatternMatch(std::max(10, iterations / 100)),
        benchConjunctiveQuery(std::max(10, iterations / 100)),
        benchEmbeddingSimilarity(std::max(10, iterations / 50)),
        benchForwardChaining(std::max(10, iterations / 100)),
        benchBinarySerialization(std::max(10, iterations / 100)),
        benchBinaryDeserialization(std::max(10, iterations / 100)),
        benchHebbianLearning(iterations),
        benchHebbianDecay(std::max(10, iterations / 50)),
        benchAtomSpaceClear(std::max(10, iterations / 100)),
    };

    for (const auto& r : results) {
        printResult(r);
    }

    std::cout << std::string(100, '-') << "\n";
    std::cout << "\nAll benchmarks complete.\n";
    return 0;
}
