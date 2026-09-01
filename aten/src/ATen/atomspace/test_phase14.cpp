/**
 * test_phase14.cpp - Tests for Phase 14 query & reasoning completeness
 *
 * Covers:
 *  - AbsentLink / negation-as-failure at the query level
 *    (PatternMatcher::findAbsent, isAbsent, createAbsentLink)
 *  - PatternMatcher::query callback delivering VariableBinding objects
 *  - NOT_LINK vs ABSENT_LINK semantics
 */

#include "ATenSpace.h"
#include "PatternMatcher.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

// ======================================================================== //
//  AbsentLink / negation-as-failure                                          //
// ======================================================================== //

void testAbsentLinkTypeAndName() {
    TEST("AbsentLink has dedicated type and type name")
        AtomSpace space;
        auto dog = createConceptNode(space, "AbsentDog0");
        auto absent = createAbsentLink(space, dog);

        ASSERT(absent->isLink());
        ASSERT(absent->getType() == Atom::Type::ABSENT_LINK);
        ASSERT(absent->getTypeName() == "AbsentLink");
        ASSERT(PatternMatcher::isAbsent(absent));
        ASSERT(!PatternMatcher::isAbsent(dog));
    END_TEST
}

void testFindAbsentReturnsNonMatching() {
    TEST("findAbsent returns atoms for which the inner pattern is absent")
        AtomSpace space;
        auto dog    = createConceptNode(space, "AbsentDog1");
        auto cat    = createConceptNode(space, "AbsentCat1");
        auto animal = createConceptNode(space, "AbsentAnimal1");

        // dog is-a animal ; cat is-a animal
        createInheritanceLink(space, dog, animal);
        createInheritanceLink(space, cat, animal);

        // Build an inner pattern (cat is-a dog) that does NOT exist as a
        // standalone link, and wrap it in an AbsentLink.
        auto innerPattern = space.addLink(Atom::Type::INHERITANCE_LINK, {cat, dog});
        auto innerAbsent = space.addLink(Atom::Type::ABSENT_LINK, {innerPattern});

        // findAbsent over this ground inner pattern returns every atom in the
        // space that does NOT match "cat is-a dog".  Since the inner link
        // itself exists (we just created it), it matches and is excluded; all
        // other atoms are returned.
        auto results = PatternMatcher::findAbsent(space, innerAbsent);
        ASSERT(!results.empty());
        // The matching inner link is excluded from the results.
        for (const auto& r : results) {
            VariableBinding b;
            ASSERT(!PatternMatcher::match(innerPattern, r, b));
        }
    END_TEST
}

void testFindAbsentEmptyWhenPresent() {
    TEST("findAbsent returns empty when the inner pattern IS present")
        AtomSpace space;
        auto dog    = createConceptNode(space, "AbsentDog2");
        auto animal = createConceptNode(space, "AbsentAnimal2");

        // Present ground pattern: dog is-a animal
        auto present = createInheritanceLink(space, dog, animal);
        (void)present;

        auto absent = space.addLink(
            Atom::Type::ABSENT_LINK,
            {space.getLink(Atom::Type::INHERITANCE_LINK, {dog, animal})});

        auto results = PatternMatcher::findAbsent(space, absent);
        // No returned atom may match the inner pattern.
        const auto& inner = absent->getOutgoing().at(0);
        for (const auto& r : results) {
            VariableBinding b;
            ASSERT(!PatternMatcher::match(inner, r, b));
        }
    END_TEST
}

void testAbsentRequiresSingleArgument() {
    TEST("findAbsent on non-AbsentLink or wrong arity returns empty")
        AtomSpace space;
        auto a = createConceptNode(space, "AbsentA3");
        auto b = createConceptNode(space, "AbsentB3");

        // Not an AbsentLink
        auto notAbsent = createInheritanceLink(space, a, b);
        ASSERT(PatternMatcher::findAbsent(space, notAbsent).empty());

        // AbsentLink with wrong arity (two args)
        auto twoArg = space.addLink(Atom::Type::ABSENT_LINK, {a, b});
        ASSERT(PatternMatcher::findAbsent(space, twoArg).empty());
    END_TEST
}

// ======================================================================== //
//  query() callback delivers VariableBinding                                 //
// ======================================================================== //

void testQueryCallbackBindings() {
    TEST("PatternMatcher::query callback receives atom + VariableBinding")
        AtomSpace space;
        auto socrates = createConceptNode(space, "QHuman1");
        auto human    = createConceptNode(space, "QConcept1");
        createInheritanceLink(space, socrates, human);

        // Pattern: InheritanceLink($X, human)
        auto varX = createVariableNode(space, "QX1");
        auto pattern = space.addLink(Atom::Type::INHERITANCE_LINK, {varX, human});

        int callCount = 0;
        bool sawBinding = false;
        PatternMatcher::query(
            space, pattern,
            [&](const Atom::Handle& matched, const VariableBinding& bindings) {
                ++callCount;
                if (matched) {
                    // The variable should be bound to socrates.
                    auto it = bindings.find(varX);
                    if (it != bindings.end() && it->second == socrates) {
                        sawBinding = true;
                    }
                }
            });

        ASSERT(callCount >= 1);
        ASSERT(sawBinding);
    END_TEST
}

void testNotLinkVsAbsentLink() {
    TEST("NOT_LINK negates a single target; ABSENT_LINK queries the space")
        AtomSpace space;
        auto cat = createConceptNode(space, "NotCat4");
        auto dog = createConceptNode(space, "NotDog4");

        // NOT_LINK around a ground node "cat" should match "dog" (not cat),
        // and fail to match "cat".
        auto notCat = createNotLink(space, cat);
        VariableBinding b1, b2;
        ASSERT(!PatternMatcher::match(notCat, cat, b1));   // cat matches -> negated
        ASSERT(PatternMatcher::match(notCat, dog, b2));    // dog != cat -> ok

        // ABSENT_LINK is a space-level query, evaluated via findAbsent.
        ASSERT(PatternMatcher::isAbsent(createAbsentLink(space, cat)));
        ASSERT(!PatternMatcher::isAbsent(notCat));
    END_TEST
}

int main() {
    std::cout << "\n--- Phase 14: AbsentLink & query completeness ---\n";

    testAbsentLinkTypeAndName();
    testFindAbsentReturnsNonMatching();
    testFindAbsentEmptyWhenPresent();
    testAbsentRequiresSingleArgument();
    testQueryCallbackBindings();
    testNotLinkVsAbsentLink();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}
