/**
 * test_binary_serializer.cpp - Tests for BinarySerializer
 */
#include "ATenSpaceCore.h"
#include "BinarySerializer.h"
#include "TruthValue.h"
#include <iostream>
#include <cassert>
#include <cstdio>
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
//  Helpers
// ======================================================================== //

static std::string tempFile(const std::string& suffix) {
    // Use a fixed temp path with a process-specific suffix to avoid collisions
    return "/tmp/atenspace_test_" + suffix + ".bin";
}

// ======================================================================== //
//  Tests
// ======================================================================== //

void testSerializeNodes() {
    TEST("serialize / deserialize nodes round-trips correctly")
        AtomSpace original;
        auto dog   = createConceptNode(original, "dog");
        auto cat   = createConceptNode(original, "cat");
        auto pred  = createPredicateNode(original, "is-mammal");

        dog->setTruthValue(TruthValue::create(0.9f, 0.8f));
        dog->setAttention(42.0f);

        auto bytes = BinarySerializer::serialize(original);
        ASSERT(!bytes.empty());

        AtomSpace restored;
        BinarySerializer::deserialize(restored, bytes);

        ASSERT(restored.size() == 3);

        auto rdog = restored.getNode(Atom::Type::CONCEPT_NODE, "dog");
        ASSERT(rdog != nullptr);
        auto tv = rdog->getTruthValue();
        ASSERT(tv.defined());
        ASSERT(std::abs(TruthValue::getStrength(tv) - 0.9f) < 0.001f);
        ASSERT(std::abs(TruthValue::getConfidence(tv) - 0.8f) < 0.001f);
        ASSERT(std::abs(rdog->getAttention() - 42.0f) < 0.001f);
    END_TEST
}

void testSerializeLinks() {
    TEST("serialize / deserialize links with outgoing sets")
        AtomSpace original;
        auto dog    = createConceptNode(original, "dog-link");
        auto mammal = createConceptNode(original, "mammal-link");
        auto link   = createInheritanceLink(original, dog, mammal);
        link->setTruthValue(TruthValue::create(0.75f, 0.6f));

        auto bytes = BinarySerializer::serialize(original);
        AtomSpace restored;
        BinarySerializer::deserialize(restored, bytes);

        ASSERT(restored.size() == 3); // dog + mammal + link

        auto rdog    = restored.getNode(Atom::Type::CONCEPT_NODE, "dog-link");
        auto rmammal = restored.getNode(Atom::Type::CONCEPT_NODE, "mammal-link");
        ASSERT(rdog != nullptr);
        ASSERT(rmammal != nullptr);

        // Find the inheritance link
        auto links = restored.getAtomsByType(Atom::Type::INHERITANCE_LINK);
        ASSERT(links.size() == 1);
        auto tv = links[0]->getTruthValue();
        ASSERT(tv.defined());
        ASSERT(std::abs(TruthValue::getStrength(tv) - 0.75f) < 0.001f);
    END_TEST
}

void testSerializeEmbedding() {
    TEST("tensor embeddings are preserved after round-trip")
        AtomSpace original;
        auto emb = torch::randn({32});
        auto node = original.addNode(Atom::Type::CONCEPT_NODE, "embedded-node", emb);

        auto bytes = BinarySerializer::serialize(original);
        AtomSpace restored;
        BinarySerializer::deserialize(restored, bytes);

        auto rnode = restored.getNode(Atom::Type::CONCEPT_NODE, "embedded-node");
        ASSERT(rnode != nullptr);

        const auto* nptr = static_cast<const Node*>(rnode.get());
        ASSERT(nptr->hasEmbedding());

        auto remb = nptr->getEmbedding();
        ASSERT(remb.size(0) == 32);

        // Check all values match within float precision
        auto origAcc  = emb.accessor<float, 1>();
        auto restAcc  = remb.accessor<float, 1>();
        for (int i = 0; i < 32; ++i) {
            ASSERT(std::abs(origAcc[i] - restAcc[i]) < 1e-5f);
        }
    END_TEST
}

void testSaveLoadFile() {
    TEST("save() and load() work via file path")
        AtomSpace original;
        createConceptNode(original, "file-test-A");
        createConceptNode(original, "file-test-B");
        createInheritanceLink(original,
            original.getNode(Atom::Type::CONCEPT_NODE, "file-test-A"),
            original.getNode(Atom::Type::CONCEPT_NODE, "file-test-B"));

        std::string path = tempFile("saveload");
        bool ok = BinarySerializer::save(original, path);
        ASSERT(ok);

        AtomSpace restored;
        bool loadOk = BinarySerializer::load(restored, path);
        ASSERT(loadOk);
        ASSERT(restored.size() == 3);

        std::remove(path.c_str());
    END_TEST
}

void testBadMagicRejects() {
    TEST("deserializing corrupted bytes throws")
        std::vector<uint8_t> garbage = {0xDE, 0xAD, 0xBE, 0xEF,
                                        0x01, 0x00, 0x00, 0x00};
        AtomSpace space;
        bool threw = false;
        try {
            BinarySerializer::deserialize(space, garbage);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        ASSERT(threw);
    END_TEST
}

void testEmptyAtomSpace() {
    TEST("empty AtomSpace serializes and restores cleanly")
        AtomSpace original;
        auto bytes = BinarySerializer::serialize(original);
        ASSERT(!bytes.empty()); // At least header bytes

        AtomSpace restored;
        BinarySerializer::deserialize(restored, bytes);
        ASSERT(restored.size() == 0);
    END_TEST
}

void testLargeGraph() {
    TEST("large graph with 200 nodes + 100 links round-trips")
        AtomSpace original;
        std::vector<Atom::Handle> nodes;
        for (int i = 0; i < 200; ++i) {
            nodes.push_back(createConceptNode(original, "n" + std::to_string(i)));
        }
        for (int i = 0; i < 100; ++i) {
            createInheritanceLink(original, nodes[i * 2], nodes[i * 2 + 1]);
        }
        ASSERT(original.size() == 300);

        auto bytes = BinarySerializer::serialize(original);
        AtomSpace restored;
        BinarySerializer::deserialize(restored, bytes);
        ASSERT(restored.size() == 300);
    END_TEST
}

void testNestedLinks() {
    TEST("link-to-link nesting serializes correctly")
        AtomSpace original;
        auto A = createConceptNode(original, "nested-A");
        auto B = createConceptNode(original, "nested-B");
        auto C = createConceptNode(original, "nested-C");

        auto AB = createInheritanceLink(original, A, B);
        auto BC = createInheritanceLink(original, B, C);

        // A list link containing two other links
        auto list = createListLink(original, {AB, BC});

        auto bytes = BinarySerializer::serialize(original);
        AtomSpace restored;
        BinarySerializer::deserialize(restored, bytes);

        // 3 nodes + 2 inheritance links + 1 list link = 6 total
        ASSERT(restored.size() == 6);

        auto lists = restored.getAtomsByType(Atom::Type::LIST_LINK);
        ASSERT(lists.size() == 1);

        const auto* lptr = static_cast<const Link*>(lists[0].get());
        ASSERT(lptr->getArity() == 2);
    END_TEST
}

// ======================================================================== //
//  Main
// ======================================================================== //

int main() {
    std::cout << "=== BinarySerializer Tests ===\n\n";

    testEmptyAtomSpace();
    testSerializeNodes();
    testSerializeLinks();
    testSerializeEmbedding();
    testSaveLoadFile();
    testBadMagicRejects();
    testLargeGraph();
    testNestedLinks();

    std::cout << "\n--- Results ---\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    return (tests_failed == 0) ? 0 : 1;
}
