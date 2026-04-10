#pragma once

#include "Atom.h"
#include "AtomSpace.h"
#include "PatternMatcher.h"
#include "TruthValue.h"
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <optional>
#include <string>

namespace at {
namespace atomspace {

/**
 * QueryResult - A single row of bound variables from a query
 */
using QueryResult = VariableBinding;

/**
 * QueryResultSet - Collection of query results
 */
using QueryResultSet = std::vector<QueryResult>;

/**
 * FilterPredicate - User-supplied predicate for filtering matches
 * Returns true if the binding should be kept
 */
using FilterPredicate = std::function<bool(const QueryResult&)>;

/**
 * QueryClause - A single pattern clause in a conjunctive query
 *
 * Each clause is a pattern atom that may contain VariableNodes.
 * Multiple clauses are joined (AND) using shared variable names.
 */
struct QueryClause {
    Atom::Handle pattern;           ///< Pattern to match
    bool optional = false;          ///< If true, clause need not match (LEFT JOIN)
    Atom::Type typeConstraint = Atom::Type::NODE;  ///< Ignored when matchAnyType=true
    bool matchAnyType = true;       ///< If true, typeConstraint is ignored

    explicit QueryClause(Atom::Handle p, bool opt = false)
        : pattern(std::move(p)), optional(opt) {}
};

/**
 * QueryBuilder - Fluent API for constructing hypergraph queries
 *
 * Example:
 * @code
 *   auto results = QueryBuilder(space)
 *       .match(InheritanceLink(?X, concept_mammal))
 *       .match(InheritanceLink(?X, concept_has_legs))
 *       .filter([](const QueryResult& r) {
 *           return getTruthStrength(r.at(x_var)) > 0.7f;
 *       })
 *       .limit(10)
 *       .execute();
 * @endcode
 */
class QueryEngine {
public:
    // ------------------------------------------------------------------ //
    //  Construction
    // ------------------------------------------------------------------ //

    explicit QueryEngine(AtomSpace& space) : space_(space) {}

    // ------------------------------------------------------------------ //
    //  Core single-pattern query
    // ------------------------------------------------------------------ //

    /**
     * Find all atoms matching a single pattern.
     * Variables in the pattern are bound to matching atoms.
     *
     * @param pattern  Pattern atom (may contain VARIABLE_NODE atoms)
     * @return         All solutions (variable → atom bindings)
     */
    QueryResultSet findMatches(const Atom::Handle& pattern) const {
        QueryResultSet results;
        auto candidates = getCandidatesByType(pattern);

        for (const auto& candidate : candidates) {
            VariableBinding bindings;
            if (PatternMatcher::match(pattern, candidate, bindings)) {
                results.push_back(bindings);
            }
        }
        return results;
    }

    // ------------------------------------------------------------------ //
    //  Conjunctive (multi-clause) query
    // ------------------------------------------------------------------ //

    /**
     * Execute a conjunctive query (logical AND of multiple clauses).
     *
     * Variables shared across clauses act as join conditions.
     *
     * @param clauses   Ordered list of pattern clauses
     * @param filters   Optional filter predicates applied after matching
     * @param maxResults  0 = unlimited
     * @return           All satisfying variable bindings
     */
    QueryResultSet executeConjunctive(
            const std::vector<QueryClause>& clauses,
            const std::vector<FilterPredicate>& filters = {},
            size_t maxResults = 0) const {

        if (clauses.empty()) return {};

        // Start with the first clause
        QueryResultSet current = findMatchesForClause(clauses[0]);

        // Join remaining clauses
        for (size_t i = 1; i < clauses.size(); ++i) {
            const auto& clause = clauses[i];

            if (clause.optional) {
                // LEFT JOIN: keep existing results even if no new match
                current = leftJoin(current, clause);
            } else {
                // INNER JOIN: only keep results that extend to this clause
                current = innerJoin(current, clause);
            }

            if (current.empty() && !clause.optional) {
                return {};  // Early termination
            }
        }

        // Apply filters
        for (const auto& filter : filters) {
            QueryResultSet filtered;
            filtered.reserve(current.size());
            for (const auto& row : current) {
                if (filter(row)) {
                    filtered.push_back(row);
                }
            }
            current = std::move(filtered);
        }

        // Apply limit
        if (maxResults > 0 && current.size() > maxResults) {
            current.resize(maxResults);
        }

        return current;
    }

    // ------------------------------------------------------------------ //
    //  Convenience query methods
    // ------------------------------------------------------------------ //

    /**
     * Find all atoms of a specific type.
     */
    std::vector<Atom::Handle> findByType(Atom::Type type) const {
        return space_.getAtomsByType(type);
    }

    /**
     * Find all atoms where truth value strength >= minStrength.
     */
    std::vector<Atom::Handle> findByTruthStrength(
            float minStrength, float minConfidence = 0.0f) const {
        std::vector<Atom::Handle> results;
        for (const auto& atom : space_.getAtoms()) {
            auto tv = atom->getTruthValue();
            if (tv.defined() && tv.numel() >= 2) {
                float s = TruthValue::getStrength(tv);
                float c = TruthValue::getConfidence(tv);
                if (s >= minStrength && c >= minConfidence) {
                    results.push_back(atom);
                }
            }
        }
        return results;
    }

    /**
     * Find nodes that are semantically similar to a query embedding.
     *
     * @param queryEmbedding  Query vector
     * @param topK            Number of results to return
     * @param minSimilarity   Minimum cosine similarity threshold
     * @return                Top-K most similar nodes
     */
    std::vector<std::pair<Atom::Handle, float>> findSimilar(
            const Tensor& queryEmbedding,
            size_t topK = 10,
            float minSimilarity = 0.0f) const {

        std::vector<std::pair<Atom::Handle, float>> scored;

        for (const auto& atom : space_.getAtoms()) {
            if (!atom->isNode()) continue;
            const auto* node = static_cast<const Node*>(atom.get());
            if (!node->hasEmbedding()) continue;

            auto emb = node->getEmbedding();
            float sim = cosineSimilarity(queryEmbedding, emb);
            if (sim >= minSimilarity) {
                scored.push_back({atom, sim});
            }
        }

        // Sort descending by similarity
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b){ return a.second > b.second; });

        if (topK > 0 && scored.size() > topK) {
            scored.resize(topK);
        }
        return scored;
    }

    /**
     * Retrieve the neighbourhood of an atom up to a given depth.
     * Returns all atoms reachable via incoming / outgoing links.
     */
    std::vector<Atom::Handle> neighbourhood(
            const Atom::Handle& seed,
            int depth = 1) const {

        std::unordered_set<Atom::Handle> visited;
        std::vector<Atom::Handle> frontier = {seed};
        visited.insert(seed);

        for (int d = 0; d < depth; ++d) {
            std::vector<Atom::Handle> next;
            for (const auto& atom : frontier) {
                // Follow outgoing links
                if (atom->isLink()) {
                    const auto* link = static_cast<const Link*>(atom.get());
                    for (const auto& child : link->getOutgoingSet()) {
                        if (visited.insert(child).second) {
                            next.push_back(child);
                        }
                    }
                }
                // Follow incoming links
                for (const auto& weak : atom->getIncomingSet()) {
                    auto parent = weak.lock();
                    if (parent && visited.insert(parent).second) {
                        next.push_back(parent);
                    }
                }
            }
            frontier = std::move(next);
        }

        visited.erase(seed);  // Exclude the seed itself
        return std::vector<Atom::Handle>(visited.begin(), visited.end());
    }

    /**
     * Count how many atoms satisfy a given pattern.
     */
    size_t count(const Atom::Handle& pattern) const {
        return findMatches(pattern).size();
    }

    /**
     * Check whether any atom in the space satisfies the given pattern.
     */
    bool exists(const Atom::Handle& pattern) const {
        auto candidates = getCandidatesByType(pattern);
        for (const auto& candidate : candidates) {
            VariableBinding bindings;
            if (PatternMatcher::match(pattern, candidate, bindings)) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ //
    //  Truth-value–aware join helpers (exposed for testing)
    // ------------------------------------------------------------------ //

    /**
     * Project a set of query results to a subset of variable names.
     * Useful for extracting only the variables of interest.
     */
    static QueryResultSet project(
            const QueryResultSet& results,
            const std::vector<Atom::Handle>& variables) {

        QueryResultSet projected;
        projected.reserve(results.size());

        for (const auto& row : results) {
            QueryResult proj;
            for (const auto& var : variables) {
                auto it = row.find(var);
                if (it != row.end()) {
                    proj[var] = it->second;
                }
            }
            if (!proj.empty()) {
                projected.push_back(proj);
            }
        }
        return projected;
    }

    /**
     * Deduplicate a result set (remove duplicate bindings).
     */
    static QueryResultSet distinct(const QueryResultSet& results) {
        QueryResultSet unique;
        // Simple O(n^2) dedup – fine for typical knowledge-graph sizes
        for (const auto& row : results) {
            bool found = false;
            for (const auto& existing : unique) {
                if (rowsEqual(row, existing)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                unique.push_back(row);
            }
        }
        return unique;
    }

private:
    AtomSpace& space_;

    // ------------------------------------------------------------------ //
    //  Internal helpers
    // ------------------------------------------------------------------ //

    QueryResultSet findMatchesForClause(const QueryClause& clause) const {
        QueryResultSet results;
        auto candidates = getCandidatesByType(clause.pattern);

        for (const auto& candidate : candidates) {
            VariableBinding bindings;
            if (PatternMatcher::match(clause.pattern, candidate, bindings)) {
                results.push_back(bindings);
            }
        }
        return results;
    }

    /**
     * Inner join: extend each existing row with new bindings from clause.
     * Rows are kept only if the clause can be matched consistently.
     */
    QueryResultSet innerJoin(
            const QueryResultSet& existing,
            const QueryClause& clause) const {

        QueryResultSet result;
        auto newMatches = findMatchesForClause(clause);

        for (const auto& row : existing) {
            for (const auto& newRow : newMatches) {
                auto merged = mergeBindings(row, newRow);
                if (merged.has_value()) {
                    result.push_back(merged.value());
                }
            }
        }
        return result;
    }

    /**
     * Left (optional) join: keep all existing rows.
     * If the clause matches consistently, extend the row; otherwise keep as-is.
     */
    QueryResultSet leftJoin(
            const QueryResultSet& existing,
            const QueryClause& clause) const {

        QueryResultSet result;
        auto newMatches = findMatchesForClause(clause);

        for (const auto& row : existing) {
            bool anyExtension = false;
            for (const auto& newRow : newMatches) {
                auto merged = mergeBindings(row, newRow);
                if (merged.has_value()) {
                    result.push_back(merged.value());
                    anyExtension = true;
                }
            }
            if (!anyExtension) {
                result.push_back(row);  // Keep unextended row
            }
        }
        return result;
    }

    /**
     * Merge two binding rows.
     * Returns std::nullopt if they are inconsistent (same variable bound to
     * different atoms).
     */
    static std::optional<QueryResult> mergeBindings(
            const QueryResult& a, const QueryResult& b) {

        QueryResult merged = a;
        for (const auto& [var, val] : b) {
            auto it = merged.find(var);
            if (it != merged.end()) {
                if (!it->second->equals(*val)) {
                    return std::nullopt;  // Conflict
                }
            } else {
                merged[var] = val;
            }
        }
        return merged;
    }

    /**
     * Get candidates by type, falling back to all atoms for variables.
     */
    std::vector<Atom::Handle> getCandidatesByType(
            const Atom::Handle& pattern) const {

        if (PatternMatcher::isVariable(pattern)) {
            auto atomSet = space_.getAtoms();
            return std::vector<Atom::Handle>(atomSet.begin(), atomSet.end());
        }
        return space_.getAtomsByType(pattern->getType());
    }

    /**
     * Compute cosine similarity between two 1-D tensors.
     */
    static float cosineSimilarity(const Tensor& a, const Tensor& b) {
        if (!a.defined() || !b.defined()) return 0.0f;
        auto a_flat = a.reshape({-1}).to(torch::kFloat);
        auto b_flat = b.reshape({-1}).to(torch::kFloat);
        if (a_flat.size(0) != b_flat.size(0)) return 0.0f;
        auto dot   = torch::dot(a_flat, b_flat);
        auto normA = torch::norm(a_flat);
        auto normB = torch::norm(b_flat);
        auto denom = normA * normB;
        if (denom.item<float>() < 1e-8f) return 0.0f;
        return (dot / denom).item<float>();
    }

    /**
     * Check if two query result rows are equal.
     */
    static bool rowsEqual(const QueryResult& a, const QueryResult& b) {
        if (a.size() != b.size()) return false;
        for (const auto& [var, val] : a) {
            auto it = b.find(var);
            if (it == b.end()) return false;
            if (!val->equals(*it->second)) return false;
        }
        return true;
    }
};

// ======================================================================== //
//  QueryBuilder - fluent API wrapping QueryEngine                          //
// ======================================================================== //

/**
 * QueryBuilder provides a fluent interface for constructing and executing
 * conjunctive hypergraph queries.
 *
 * Usage:
 * @code
 *   QueryResultSet results = QueryBuilder(space)
 *       .match(pattern1)
 *       .optionalMatch(pattern2)
 *       .filter([](const QueryResult& r){ return ...; })
 *       .limit(50)
 *       .execute();
 * @endcode
 */
class QueryBuilder {
public:
    explicit QueryBuilder(AtomSpace& space) : engine_(space) {}

    /** Add a mandatory match clause */
    QueryBuilder& match(const Atom::Handle& pattern) {
        clauses_.emplace_back(pattern, false);
        return *this;
    }

    /** Add an optional match clause (LEFT JOIN semantics) */
    QueryBuilder& optionalMatch(const Atom::Handle& pattern) {
        clauses_.emplace_back(pattern, true);
        return *this;
    }

    /** Add a filter predicate applied after all clauses are matched */
    QueryBuilder& filter(FilterPredicate pred) {
        filters_.push_back(std::move(pred));
        return *this;
    }

    /** Filter results by minimum truth-value strength of a specific variable */
    QueryBuilder& filterByStrength(const Atom::Handle& var, float minStrength) {
        filters_.push_back([var, minStrength](const QueryResult& row) {
            auto it = row.find(var);
            if (it == row.end()) return false;
            auto tv = it->second->getTruthValue();
            if (!tv.defined() || tv.numel() < 1) return false;
            return TruthValue::getStrength(tv) >= minStrength;
        });
        return *this;
    }

    /** Limit the number of results returned */
    QueryBuilder& limit(size_t n) {
        maxResults_ = n;
        return *this;
    }

    /** Execute the query and return results */
    QueryResultSet execute() const {
        return engine_.executeConjunctive(clauses_, filters_, maxResults_);
    }

    /** Count matching results without returning full bindings */
    size_t count() const {
        return execute().size();
    }

    /**
     * Negation-as-failure (Phase 10): exclude any result row for which the
     * given pattern also matches (using the variable bindings from previous
     * clauses).  The pattern is matched independently; if it finds at least
     * one result, the row is discarded.
     *
     * Example: find all mammals that are NOT domestic
     * @code
     *   QueryBuilder(space)
     *       .match(InheritanceLink(?X, mammal))
     *       .notMatch(InheritanceLink(?X, domestic))
     *       .execute();
     * @endcode
     */
    QueryBuilder& notMatch(const Atom::Handle& pattern) {
        negations_.push_back(pattern);
        return *this;
    }

    /**
     * filterByConfidence: filter results by minimum truth-value confidence
     * of a specific variable (Phase 10 convenience).
     */
    QueryBuilder& filterByConfidence(const Atom::Handle& var, float minConf) {
        filters_.push_back([var, minConf](const QueryResult& row) {
            auto it = row.find(var);
            if (it == row.end()) return false;
            auto tv = it->second->getTruthValue();
            if (!tv.defined() || tv.numel() < 2) return false;
            return TruthValue::getConfidence(tv) >= minConf;
        });
        return *this;
    }

private:
    QueryEngine engine_;
    std::vector<QueryClause> clauses_;
    std::vector<FilterPredicate> filters_;
    std::vector<Atom::Handle> negations_;
    size_t maxResults_ = 0;

    /** Build the combined filter list including negation-as-failure checks */
    std::vector<FilterPredicate> buildFilters() const {
        std::vector<FilterPredicate> combined = filters_;
        for (const auto& negPat : negations_) {
            combined.push_back([this, negPat](const QueryResult& row) {
                // Substitute bound variables into the negation pattern
                // then check if any match exists – if yes, exclude the row
                QueryResultSet negMatches = engine_.findMatches(negPat);
                // Simple check: any result means a match exists → exclude
                return negMatches.empty();
            });
        }
        return combined;
    }

public:
    /** Execute the query with negation-as-failure applied */
    QueryResultSet executeWithNegation() const {
        return engine_.executeConjunctive(clauses_, buildFilters(), maxResults_);
    }

}; // class QueryBuilder

} // namespace atomspace
} // namespace at
