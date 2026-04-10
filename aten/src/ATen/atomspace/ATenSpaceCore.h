#pragma once
/**
 * ATenSpaceCore.h - Minimal working includes + convenience API
 *
 * Use this header instead of the full ATenSpace.h to avoid pulling in
 * headers that depend on optional external features.
 */
#include "Atom.h"
#include "AtomSpace.h"
#include "TruthValue.h"
#include "AttentionBank.h"
#include "PatternMatcher.h"
#include "TimeServer.h"
#include "Serializer.h"
#include "ForwardChainer.h"
#include "BackwardChainer.h"
#include "ECAN.h"

namespace at {
namespace atomspace {

// =====================================================================
// Convenience node/link constructors (mirrors ATenSpace.h public API)
// =====================================================================

inline Atom::Handle createConceptNode(AtomSpace& space,
                                      const std::string& name) {
    return space.addNode(Atom::Type::CONCEPT_NODE, name);
}

inline Atom::Handle createConceptNode(AtomSpace& space,
                                      const std::string& name,
                                      const Tensor& embedding) {
    return space.addNode(Atom::Type::CONCEPT_NODE, name, embedding);
}

inline Atom::Handle createPredicateNode(AtomSpace& space,
                                        const std::string& name) {
    return space.addNode(Atom::Type::PREDICATE_NODE, name);
}

inline Atom::Handle createInheritanceLink(AtomSpace& space,
                                          const Atom::Handle& from,
                                          const Atom::Handle& to) {
    return space.addLink(Atom::Type::INHERITANCE_LINK, {from, to});
}

inline Atom::Handle createEvaluationLink(AtomSpace& space,
                                         const Atom::Handle& predicate,
                                         const Atom::Handle& argList) {
    return space.addLink(Atom::Type::EVALUATION_LINK, {predicate, argList});
}

inline Atom::Handle createListLink(AtomSpace& space,
                                   const std::vector<Atom::Handle>& atoms) {
    return space.addLink(Atom::Type::LIST_LINK, atoms);
}

inline Atom::Handle createImplicationLink(AtomSpace& space,
                                          const Atom::Handle& from,
                                          const Atom::Handle& to) {
    return space.addLink(Atom::Type::IMPLICATION_LINK, {from, to});
}

inline Atom::Handle createSimilarityLink(AtomSpace& space,
                                         const Atom::Handle& a,
                                         const Atom::Handle& b) {
    return space.addLink(Atom::Type::SIMILARITY_LINK, {a, b});
}

} // namespace atomspace
} // namespace at
