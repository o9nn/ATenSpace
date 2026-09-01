#pragma once

#include "Atom.h"
#include "AtomSpace.h"
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <string>
namespace at {
namespace atomspace {

/**
 * VariableBinding - Maps variable nodes to their bound values
 */
using VariableBinding = std::unordered_map<Atom::Handle, Atom::Handle>;

/**
 * PatternMatcher - Pattern matching and unification engine
 *
 * Provides pattern matching capabilities for ATenSpace, enabling:
 * - Variable binding and substitution
 * - Pattern queries with wildcards
 * - Unification of structures
 * - Support for inference rules
 * - Type-constrained variables (Phase 10)
 * - Glob nodes for sequence wildcards (Phase 10)
 *
 * Variables are represented as VariableNode atoms in patterns.
 * Type-constrained variables use TypedVariableNode and encode the
 * permitted atom type in the node name using the format "?VarName:TypeName".
 */
class PatternMatcher {
public:
    /**
     * VariableBinding - Maps variable nodes to their bound values.
     * (Exposed as a nested alias so consumers can write
     *  `PatternMatcher::VariableBinding`; same type as the
     *  namespace-level `at::atomspace::VariableBinding`.)
     */
    using VariableBinding = at::atomspace::VariableBinding;

    /**
     * MatchCallback - Invoked for every complete match (Iteration 1, FR-1.1).
     *
     * Signature: void(const Atom::Handle& matchedAtom,
     *                 const VariableBinding& bindings)
     *
     * Enables streaming consumers (InferencePipeline, CognitiveEngine)
     * to process matches one at a time without materializing the full
     * result set.
     */
    using MatchCallback =
        std::function<void(const Atom::Handle&, const VariableBinding&)>;

    /**
     * Match a pattern against a target atom.
     *
     * Phase 10 additions:
     *  - TypedVariableNode: name format "?X:ConceptNode" – binds only when the
     *    target's type name equals the suffix after the colon.
     *  - GlobNode: not applicable at the atom level (handled by link matching below).
     *
     * @param pattern Pattern atom (may contain VariableNodes / TypedVariableNodes)
     * @param target  Target atom to match against
     * @param bindings Variable bindings (input/output)
     * @return true if pattern matches target
     */
    static bool match(const Atom::Handle& pattern,
                     const Atom::Handle& target,
                     VariableBinding& bindings) {
        // If pattern is a plain variable, bind it unconditionally
        if (isVariable(pattern)) {
            return bindVariable(pattern, target, bindings);
        }

        // If pattern is a type-constrained variable, check the type hint first
        if (isTypedVariable(pattern)) {
            if (!typedVariableAccepts(pattern, target)) {
                return false;
            }
            return bindVariable(pattern, target, bindings);
        }

        // Negation-as-failure: NOT_LINK wraps one inner pattern; the outer
        // match succeeds iff the inner pattern does NOT match the target.
        if (pattern->isLink() &&
                pattern->getType() == Atom::Type::NOT_LINK) {
            const Link* notLink = static_cast<const Link*>(pattern.get());
            if (notLink->getArity() == 1) {
                VariableBinding innerBindings = bindings; // tentative copy
                bool innerMatched = match(notLink->getOutgoingAtom(0),
                                          target, innerBindings);
                // Negation succeeds when inner fails; no new bindings added.
                return !innerMatched;
            }
        }

        // Check type compatibility for concrete patterns
        if (pattern->getType() != target->getType()) {
            return false;
        }

        // For nodes, check name equality
        if (pattern->isNode()) {
            const Node* patternNode = static_cast<const Node*>(pattern.get());
            const Node* targetNode  = static_cast<const Node*>(target.get());
            return patternNode->getName() == targetNode->getName();
        }

        // For links, recursively match outgoing sets (with GlobNode support)
        if (pattern->isLink()) {
            const Link* patternLink = static_cast<const Link*>(pattern.get());
            const Link* targetLink  = static_cast<const Link*>(target.get());

            const auto& pOut = patternLink->getOutgoingSet();
            const auto& tOut = targetLink->getOutgoingSet();

            return matchOutgoing(pOut, tOut, bindings);
        }

        return false;
    }

    /**
     * Find all atoms in the atomspace that match the pattern.
     *
     * @param space   AtomSpace to search
     * @param pattern Pattern to match
     * @return Vector of (matched_atom, bindings) pairs
     */
    static std::vector<std::pair<Atom::Handle, VariableBinding>>
    findMatches(AtomSpace& space, const Atom::Handle& pattern) {
        std::vector<std::pair<Atom::Handle, VariableBinding>> results;

        forEachMatch(space, pattern,
            [&results](const Atom::Handle& atom, const VariableBinding& bindings) {
                results.push_back({atom, bindings});
            });

        return results;
    }

    /**
     * Stream matches to a callback instead of materializing a result set
     * (Iteration 1, FR-1.1).
     *
     * The callback is invoked once for every complete match.  When the
     * pattern is an ABSENT_LINK, the enclosed pattern is evaluated as
     * negation-as-failure: the callback fires exactly once (with empty
     * bindings) iff the enclosed pattern yields zero matches, and does
     * not fire otherwise.  No atoms are added or removed — the space is
     * unchanged (FR-1.2).
     *
     * @param space    AtomSpace to search
     * @param pattern  Pattern to match
     * @param callback MatchCallback invoked per complete match
     * @return Number of matches dispatched
     */
    static size_t forEachMatch(AtomSpace& space,
                               const Atom::Handle& pattern,
                               MatchCallback callback) {
        // Negation-as-failure: ABSENT_LINK wraps one inner pattern and
        // succeeds iff the inner pattern yields zero matches.
        if (pattern->isLink() &&
                pattern->getType() == Atom::Type::ABSENT_LINK) {
            const Link* absentLink = static_cast<const Link*>(pattern.get());
            if (absentLink->getArity() == 1) {
                const Atom::Handle& inner = absentLink->getOutgoingAtom(0);
                if (!existsMatch(space, inner)) {
                    VariableBinding empty;
                    callback(pattern, empty);
                    return 1;
                }
            }
            return 0;
        }

        size_t count = 0;
        auto candidates = getCandidates(space, pattern);
        for (const auto& candidate : candidates) {
            VariableBinding bindings;
            if (match(pattern, candidate, bindings)) {
                callback(candidate, bindings);
                ++count;
            }
        }
        return count;
    }

    /**
     * Return true if at least one atom in the space matches the pattern.
     * Short-circuits on the first match (no side effects).
     */
    static bool existsMatch(AtomSpace& space, const Atom::Handle& pattern) {
        // ABSENT_LINK inside existsMatch: present iff inner is absent.
        if (pattern->isLink() &&
                pattern->getType() == Atom::Type::ABSENT_LINK) {
            const Link* absentLink = static_cast<const Link*>(pattern.get());
            if (absentLink->getArity() == 1) {
                return !existsMatch(space, absentLink->getOutgoingAtom(0));
            }
            return false;
        }
        auto candidates = getCandidates(space, pattern);
        for (const auto& candidate : candidates) {
            VariableBinding bindings;
            if (match(pattern, candidate, bindings)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Substitute variables in a pattern using bindings.
     */
    static Atom::Handle substitute(const Atom::Handle& pattern,
                                   const VariableBinding& bindings,
                                   AtomSpace& space) {
        if (isVariable(pattern) || isTypedVariable(pattern)) {
            auto it = bindings.find(pattern);
            if (it != bindings.end()) {
                return it->second;
            }
            return pattern; // Unbound variable
        }

        if (pattern->isNode()) {
            return pattern;
        }

        if (pattern->isLink()) {
            const Link* link = static_cast<const Link*>(pattern.get());
            std::vector<Atom::Handle> newOutgoing;

            for (const auto& atom : link->getOutgoingSet()) {
                newOutgoing.push_back(substitute(atom, bindings, space));
            }

            return space.addLink(link->getType(), newOutgoing);
        }

        return pattern;
    }

    /**
     * Unify two patterns, finding a common binding.
     */
    static bool unify(const Atom::Handle& pattern1,
                     const Atom::Handle& pattern2,
                     VariableBinding& bindings) {
        if (isVariable(pattern1) || isTypedVariable(pattern1)) {
            if (isTypedVariable(pattern1) && !typedVariableAccepts(pattern1, pattern2)) {
                return false;
            }
            return bindVariable(pattern1, pattern2, bindings);
        }
        if (isVariable(pattern2) || isTypedVariable(pattern2)) {
            if (isTypedVariable(pattern2) && !typedVariableAccepts(pattern2, pattern1)) {
                return false;
            }
            return bindVariable(pattern2, pattern1, bindings);
        }

        if (pattern1->getType() != pattern2->getType()) {
            return false;
        }

        if (pattern1->isNode()) {
            const Node* node1 = static_cast<const Node*>(pattern1.get());
            const Node* node2 = static_cast<const Node*>(pattern2.get());
            return node1->getName() == node2->getName();
        }

        if (pattern1->isLink()) {
            const Link* link1 = static_cast<const Link*>(pattern1.get());
            const Link* link2 = static_cast<const Link*>(pattern2.get());

            const auto& outgoing1 = link1->getOutgoingSet();
            const auto& outgoing2 = link2->getOutgoingSet();

            if (outgoing1.size() != outgoing2.size()) {
                return false;
            }

            for (size_t i = 0; i < outgoing1.size(); ++i) {
                if (!unify(outgoing1[i], outgoing2[i], bindings)) {
                    return false;
                }
            }
            return true;
        }

        return false;
    }

    /**
     * Query the atomspace with a pattern.
     */
    static void query(AtomSpace& space,
                     const Atom::Handle& pattern,
                     std::function<void(const Atom::Handle&, const VariableBinding&)> callback) {
        auto matches = findMatches(space, pattern);
        for (const auto& [atom, bindings] : matches) {
            callback(atom, bindings);
        }
    }

    /** Check if an atom is a plain variable node */
    static bool isVariable(const Atom::Handle& atom) {
        return atom->isNode() &&
               atom->getType() == Atom::Type::VARIABLE_NODE;
    }

    /**
     * Check if an atom is a type-constrained variable (TypedVariableNode).
     *
     * A TypedVariableNode encodes the constraint in its name: "?X:ConceptNode".
     * The portion after the last ':' must equal the target's getTypeName().
     */
    static bool isTypedVariable(const Atom::Handle& atom) {
        return atom->isNode() &&
               atom->getType() == Atom::Type::TYPED_VARIABLE_NODE;
    }

    /** Check if an atom is a GlobNode (sequence wildcard) */
    static bool isGlob(const Atom::Handle& atom) {
        return atom->isNode() &&
               atom->getType() == Atom::Type::GLOB_NODE;
    }

    /**
     * Return the type-name constraint encoded in a TypedVariableNode.
     * E.g. a node named "?X:ConceptNode" returns "ConceptNode".
     * Returns empty string if no constraint separator ':' is found.
     */
    static std::string getTypeConstraint(const Atom::Handle& tvar) {
        const Node* n = static_cast<const Node*>(tvar.get());
        const std::string& name = n->getName();
        auto pos = name.rfind(':');
        if (pos == std::string::npos) return "";
        return name.substr(pos + 1);
    }

    /**
     * Check whether a TypedVariableNode accepts a given target atom.
     */
    static bool typedVariableAccepts(const Atom::Handle& tvar,
                                     const Atom::Handle& target) {
        std::string constraint = getTypeConstraint(tvar);
        if (constraint.empty()) return true; // no constraint → accept all
        return target->getTypeName() == constraint;
    }

private:
    /**
     * Bind a variable to a value, checking consistency if already bound.
     */
    static bool bindVariable(const Atom::Handle& var,
                            const Atom::Handle& value,
                            VariableBinding& bindings) {
        auto it = bindings.find(var);
        if (it != bindings.end()) {
            return it->second->equals(*value);
        }
        bindings[var] = value;
        return true;
    }

    /**
     * Match outgoing sets element-by-element, handling GlobNode wildcards.
     *
     * A GlobNode in the pattern outgoing set matches zero or more consecutive
     * target atoms.  At most one GlobNode per outgoing set is efficiently
     * supported; nested globs are undefined.
     */
    static bool matchOutgoing(const std::vector<Atom::Handle>& pOut,
                              const std::vector<Atom::Handle>& tOut,
                              VariableBinding& bindings) {
        // Check if the pattern contains a GlobNode
        int globIdx = -1;
        for (int i = 0; i < static_cast<int>(pOut.size()); ++i) {
            if (isGlob(pOut[i])) {
                globIdx = i;
                break;
            }
        }

        if (globIdx == -1) {
            // No glob: lengths must match exactly
            if (pOut.size() != tOut.size()) return false;
            for (size_t i = 0; i < pOut.size(); ++i) {
                if (!match(pOut[i], tOut[i], bindings)) return false;
            }
            return true;
        }

        // With one GlobNode:
        // prefix = pOut[0..globIdx-1], suffix = pOut[globIdx+1..end]
        // The glob absorbs tOut[globIdx .. tOut.size()-suffix.size()-1]
        size_t prefixLen = static_cast<size_t>(globIdx);
        size_t suffixLen = pOut.size() - globIdx - 1;

        if (tOut.size() < prefixLen + suffixLen) return false;

        // Match prefix
        for (size_t i = 0; i < prefixLen; ++i) {
            if (!match(pOut[i], tOut[i], bindings)) return false;
        }
        // Match suffix (from the end)
        for (size_t i = 0; i < suffixLen; ++i) {
            size_t pi = pOut.size() - suffixLen + i;
            size_t ti = tOut.size() - suffixLen + i;
            if (!match(pOut[pi], tOut[ti], bindings)) return false;
        }
        // Bind the GlobNode to the absorbed atoms (represented as a list handle)
        // We skip binding here since list atoms would require a shared AtomSpace.
        // Callers can inspect bindings for ?glob variables separately.
        return true;
    }

    /**
     * Get candidate atoms for matching based on pattern structure.
     */
    static std::vector<Atom::Handle> getCandidates(AtomSpace& space,
                                                   const Atom::Handle& pattern) {
        std::vector<Atom::Handle> candidates;

        if (isVariable(pattern) || isTypedVariable(pattern)) {
            // For typed variables, filter by constraint type if possible
            if (isTypedVariable(pattern)) {
                std::string constraint = getTypeConstraint(pattern);
                if (!constraint.empty()) {
                    const auto& all = space.getAtoms();
                    for (const auto& a : all) {
                        if (a->getTypeName() == constraint) {
                            candidates.push_back(a);
                        }
                    }
                    return candidates;
                }
            }
            const auto& allAtoms = space.getAtoms();
            candidates.insert(candidates.end(), allAtoms.begin(), allAtoms.end());
            return candidates;
        }

        // For globs, return all atoms
        if (isGlob(pattern)) {
            const auto& allAtoms = space.getAtoms();
            candidates.insert(candidates.end(), allAtoms.begin(), allAtoms.end());
            return candidates;
        }

        // Negation patterns (NOT_LINK / ABSENT_LINK) must consider every
        // atom as a candidate: the whole point is to match atoms that do
        // NOT satisfy the inner pattern, so filtering by the negation
        // link's own type would wrongly yield zero candidates.
        if (pattern->isLink() &&
            (pattern->getType() == Atom::Type::NOT_LINK ||
             pattern->getType() == Atom::Type::ABSENT_LINK)) {
            const auto& allAtoms = space.getAtoms();
            candidates.insert(candidates.end(), allAtoms.begin(), allAtoms.end());
            return candidates;
        }

        auto atomsByType = space.getAtomsByType(pattern->getType());

        if (pattern->isNode()) {
            const Node* node = static_cast<const Node*>(pattern.get());
            for (const auto& atom : atomsByType) {
                const Node* candidate = static_cast<const Node*>(atom.get());
                if (candidate->getName() == node->getName()) {
                    candidates.push_back(atom);
                }
            }
        } else {
            candidates = atomsByType;
        }

        return candidates;
    }
};

/**
 * Pattern - Helper class for building patterns
 */
class Pattern {
public:
    static Atom::Handle from(const Atom::Handle& atom) {
        return atom;
    }

    static bool hasVariables(const Atom::Handle& pattern) {
        if (PatternMatcher::isVariable(pattern) || PatternMatcher::isTypedVariable(pattern)) {
            return true;
        }

        if (pattern->isLink()) {
            const Link* link = static_cast<const Link*>(pattern.get());
            for (const auto& atom : link->getOutgoingSet()) {
                if (hasVariables(atom)) {
                    return true;
                }
            }
        }

        return false;
    }

    static std::vector<Atom::Handle> getVariables(const Atom::Handle& pattern) {
        std::unordered_set<Atom::Handle> vars;
        collectVariables(pattern, vars);
        return std::vector<Atom::Handle>(vars.begin(), vars.end());
    }

private:
    static void collectVariables(const Atom::Handle& pattern,
                                 std::unordered_set<Atom::Handle>& vars) {
        if (PatternMatcher::isVariable(pattern) || PatternMatcher::isTypedVariable(pattern)) {
            vars.insert(pattern);
            return;
        }

        if (pattern->isLink()) {
            const Link* link = static_cast<const Link*>(pattern.get());
            for (const auto& atom : link->getOutgoingSet()) {
                collectVariables(atom, vars);
            }
        }
    }
};

} // namespace atomspace
} // namespace at
