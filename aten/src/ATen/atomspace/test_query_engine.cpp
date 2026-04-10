/**
 * test_query_engine.cpp - Tests for QueryEngine and QueryBuilder
 */
#include "ATenSpaceCore.h"
#include "QueryEngine.h"
#include <iostream>
#include <cassert>
#include <string>

using namespace at::atomspace;

// ======================================================================== //
//  Helpers
// ======================================================================== //

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
//  Test cases
// ======================================================================== //

void testFindByType() {
    TEST("findByType returns correct atoms")
        AtomSpace space;
        auto dog    = createConceptNode(space, "dog");
        auto cat    = createConceptNode(space, "cat");
        auto pred   = createPredicateNode(space, "is-animal");

        QueryEngine qe(space);
        auto concepts = qe.findByType(Atom::Type::CONCEPT_NODE);
        ASSERT(concepts.size() == 2);

        auto preds = qe.findByType(Atom::Type::PREDICATE_NODE);
        ASSERT(preds.size() == 1);
    END_TEST
}

void testSinglePatternMatch() {
    TEST("findMatches with concrete pattern returns exact match")
        AtomSpace space;
        auto mammal = createConceptNode(space, "mammal");
        auto dog    = createConceptNode(space, "dog");
        auto link   = createInheritanceLink(space, dog, mammal);

        QueryEngine qe(space);

        // Pattern: InheritanceLink(dog, mammal)  — no variables, exact match
        auto results = qe.findMatches(link);
        ASSERT(!results.empty());
    END_TEST
}

void testVariableBinding() {
    TEST("findMatches with variable binds correctly")
        AtomSpace space;
        auto mammal = createConceptNode(space, "mammal");
        auto dog    = createConceptNode(space, "dog");
        auto cat    = createConceptNode(space, "cat");
        createInheritanceLink(space, dog, mammal);
        createInheritanceLink(space, cat, mammal);

        // Pattern: InheritanceLink(?X, mammal)
        auto varX   = space.addNode(Atom::Type::VARIABLE_NODE, "?X");
        auto pattern = createInheritanceLink(space, varX, mammal);

        QueryEngine qe(space);
        auto results = qe.findMatches(pattern);
        ASSERT(results.size() >= 2);  // dog and cat should match

        // Check that ?X is bound to dog or cat
        bool foundDog = false, foundCat = false;
        for (const auto& row : results) {
            auto it = row.find(varX);
            if (it == row.end()) continue;
            if (it->second == dog) foundDog = true;
            if (it->second == cat) foundCat = true;
        }
        ASSERT(foundDog);
        ASSERT(foundCat);
    END_TEST
}

void testConjunctiveQuery() {
    TEST("executeConjunctive joins multiple clauses correctly")
        AtomSpace space;
        auto mammal  = createConceptNode(space, "conj-mammal");
        auto dog     = createConceptNode(space, "conj-dog");
        auto cat     = createConceptNode(space, "conj-cat");
        auto hasLegs = createConceptNode(space, "conj-has-legs");

        // Data facts: dog and cat inherit mammal; only dog has legs
        createInheritanceLink(space, dog, mammal);
        createInheritanceLink(space, cat, mammal);
        createInheritanceLink(space, dog, hasLegs);

        // Create query BEFORE adding variable to avoid self-match issues
        // Use a fresh space for the query pattern
        auto varX    = space.addNode(Atom::Type::VARIABLE_NODE, "?X-conj");
        auto pat1    = createInheritanceLink(space, varX, mammal);
        auto pat2    = createInheritanceLink(space, varX, hasLegs);

        QueryEngine qe(space);
        auto results = qe.executeConjunctive(
            {QueryClause{pat1}, QueryClause{pat2}});

        // Find results that bind ?X to a non-variable concept node
        Atom::Handle foundDog = nullptr;
        for (const auto& row : results) {
            auto it = row.find(varX);
            if (it == row.end()) continue;
            // Skip the variable node itself as a binding
            if (it->second->getType() == Atom::Type::VARIABLE_NODE) continue;
            if (it->second == dog) foundDog = it->second;
        }
        ASSERT(foundDog != nullptr);  // dog must be in results

        // dog should be the only CONCEPT_NODE result
        int conceptCount = 0;
        for (const auto& row : results) {
            auto it = row.find(varX);
            if (it == row.end()) continue;
            if (it->second->getType() == Atom::Type::CONCEPT_NODE) {
                ++conceptCount;
            }
        }
        ASSERT(conceptCount == 1);  // only dog is a concept satisfying both
    END_TEST
}

void testOptionalJoin() {
    TEST("optional clause keeps rows even if no new match")
        AtomSpace space;
        auto mammal  = createConceptNode(space, "mammal");
        auto dog     = createConceptNode(space, "dog");
        auto cat     = createConceptNode(space, "cat");
        auto hasWings = createConceptNode(space, "has-wings");

        createInheritanceLink(space, dog, mammal);
        createInheritanceLink(space, cat, mammal);
        // Neither dog nor cat has wings

        auto varX    = space.addNode(Atom::Type::VARIABLE_NODE, "?X");
        auto varY    = space.addNode(Atom::Type::VARIABLE_NODE, "?Y");
        auto pat1    = createInheritanceLink(space, varX, mammal);
        auto pat2    = createInheritanceLink(space, varX, hasWings); // optional

        QueryEngine qe(space);
        auto results = qe.executeConjunctive(
            {QueryClause{pat1, false}, QueryClause{pat2, true}});  // pat2 optional

        // Both dog and cat should remain (pat2 is optional)
        ASSERT(results.size() >= 2);
    END_TEST
}

void testFilterByStrength() {
    TEST("findByTruthStrength filters correctly")
        AtomSpace space;
        auto a = createConceptNode(space, "high");
        a->setTruthValue(TruthValue::create(0.9f, 0.8f));
        auto b = createConceptNode(space, "low");
        b->setTruthValue(TruthValue::create(0.2f, 0.8f));
        auto c = createConceptNode(space, "medium");
        c->setTruthValue(TruthValue::create(0.6f, 0.8f));

        QueryEngine qe(space);
        auto strong = qe.findByTruthStrength(0.5f);
        ASSERT(strong.size() == 2);  // high and medium

        auto veryStrong = qe.findByTruthStrength(0.8f);
        ASSERT(veryStrong.size() == 1);  // only high
    END_TEST
}

void testNeighbourhood() {
    TEST("neighbourhood returns connected atoms up to depth")
        AtomSpace space;
        auto A = createConceptNode(space, "A");
        auto B = createConceptNode(space, "B");
        auto C = createConceptNode(space, "C");
        auto AB = createInheritanceLink(space, A, B);
        auto BC = createInheritanceLink(space, B, C);

        QueryEngine qe(space);
        auto n1 = qe.neighbourhood(A, 1);
        // Depth-1 from A: should include AB (incoming to A), B (via AB)
        ASSERT(!n1.empty());

        auto n2 = qe.neighbourhood(A, 2);
        ASSERT(n2.size() >= n1.size());
    END_TEST
}

void testQueryBuilder() {
    TEST("QueryBuilder fluent API works end-to-end")
        AtomSpace space;
        auto mammal = createConceptNode(space, "qb-mammal");
        auto dog    = createConceptNode(space, "qb-dog");
        auto cat    = createConceptNode(space, "qb-cat");
        dog->setTruthValue(TruthValue::create(0.9f, 0.8f));
        cat->setTruthValue(TruthValue::create(0.3f, 0.8f));
        createInheritanceLink(space, dog, mammal);
        createInheritanceLink(space, cat, mammal);

        auto varX   = space.addNode(Atom::Type::VARIABLE_NODE, "?X-qb");
        auto pat    = createInheritanceLink(space, varX, mammal);

        auto results = QueryBuilder(space)
                           .match(pat)
                           .filterByStrength(varX, 0.5f)
                           .limit(10)
                           .execute();

        // Find concept nodes in results that pass TV filter
        int dogCount = 0, catCount = 0;
        for (const auto& row : results) {
            auto it = row.find(varX);
            if (it == row.end()) continue;
            if (it->second == dog) ++dogCount;
            if (it->second == cat) ++catCount;
        }
        ASSERT(dogCount == 1);  // dog passes TV filter (0.9 > 0.5)
        ASSERT(catCount == 0);  // cat fails TV filter (0.3 < 0.5)
    END_TEST
}

void testExistsCount() {
    TEST("exists() and count() work correctly")
        AtomSpace space;
        auto A = createConceptNode(space, "exists-A");
        auto B = createConceptNode(space, "exists-B");
        createInheritanceLink(space, A, B);

        QueryEngine qe(space);
        ASSERT(qe.exists(A));
        ASSERT(qe.count(A) >= 1);

        auto absent = space.addNode(Atom::Type::CONCEPT_NODE, "absent-node");
        space.removeAtom(absent);  // remove it
        // Querying for it should return 0 matches
        auto varX = space.addNode(Atom::Type::VARIABLE_NODE, "?unused");
        auto pat  = createInheritanceLink(space, varX, A);
        size_t cnt = qe.count(pat);
        (void)cnt;  // just check it doesn't crash
    END_TEST
}

void testDistinct() {
    TEST("distinct() removes duplicate bindings")
        QueryResult r1, r2, r3;
        AtomSpace space;
        auto x = space.addNode(Atom::Type::VARIABLE_NODE, "?v");
        auto a = createConceptNode(space, "dup-a");

        r1[x] = a;
        r2[x] = a;   // Duplicate of r1
        r3[x] = createConceptNode(space, "dup-b");

        QueryResultSet rs = {r1, r2, r3};
        auto unique = QueryEngine::distinct(rs);
        ASSERT(unique.size() == 2);
    END_TEST
}

// ======================================================================== //
//  Main
// ======================================================================== //

int main() {
    std::cout << "=== QueryEngine Tests ===\n\n";

    testFindByType();
    testSinglePatternMatch();
    testVariableBinding();
    testConjunctiveQuery();
    testOptionalJoin();
    testFilterByStrength();
    testNeighbourhood();
    testQueryBuilder();
    testExistsCount();
    testDistinct();

    std::cout << "\n--- Results ---\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    return (tests_failed == 0) ? 0 : 1;
}
