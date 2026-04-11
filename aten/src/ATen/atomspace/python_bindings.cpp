/**
 * Python Bindings for ATenSpace
 *
 * Provides Python API access to ATenSpace cognitive architecture:
 * - Core AtomSpace (knowledge representation)
 * - PLN (Probabilistic Logic Networks)
 * - ECAN (Economic Attention Networks)
 * - NLU (Natural Language Understanding)
 * - Vision (Visual Perception)
 * - CognitiveEngine (Integrated reasoning)
 * - ModelLoader (TorchScript model loading) - Phase 8
 * - QueryEngine, BinarySerializer, InferencePipeline, HebbianLearner - Phase 9
 * - TypedVariable / GlobNode / negation-as-failure - Phase 10
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <torch/extension.h>

#include "ATenSpace.h"
#include "ModelLoader.h"
#include "Tokenizer.h"
#include "QueryEngine.h"
#include "BinarySerializer.h"
#include "InferencePipeline.h"
#include "HebbianLearner.h"

namespace py = pybind11;
using namespace at::atomspace;

// Module definition
PYBIND11_MODULE(atenspace, m) {
    m.doc() = "ATenSpace - Tensor-based Cognitive Architecture";

    // ============================================================
    // CORE ATOMSPACE
    // ============================================================
    
    // Atom::Type enum
    py::enum_<Atom::Type>(m, "AtomType")
        .value("NODE", Atom::Type::NODE)
        .value("LINK", Atom::Type::LINK)
        .value("CONCEPT_NODE", Atom::Type::CONCEPT_NODE)
        .value("PREDICATE_NODE", Atom::Type::PREDICATE_NODE)
        .value("VARIABLE_NODE", Atom::Type::VARIABLE_NODE)
        .value("INHERITANCE_LINK", Atom::Type::INHERITANCE_LINK)
        .value("SIMILARITY_LINK", Atom::Type::SIMILARITY_LINK)
        .value("EVALUATION_LINK", Atom::Type::EVALUATION_LINK)
        .value("IMPLICATION_LINK", Atom::Type::IMPLICATION_LINK)
        .value("AND_LINK", Atom::Type::AND_LINK)
        .value("OR_LINK", Atom::Type::OR_LINK)
        .value("NOT_LINK", Atom::Type::NOT_LINK)
        .value("MEMBER_LINK", Atom::Type::MEMBER_LINK)
        .value("SUBSET_LINK", Atom::Type::SUBSET_LINK)
        .value("SEQUENTIAL_LINK", Atom::Type::SEQUENTIAL_LINK)
        .value("SIMULTANEOUS_LINK", Atom::Type::SIMULTANEOUS_LINK)
        .value("CONTEXT_LINK", Atom::Type::CONTEXT_LINK)
        .value("EXECUTION_LINK", Atom::Type::EXECUTION_LINK)
        // Phase 10 additions
        .value("HEBBIAN_LINK",            Atom::Type::HEBBIAN_LINK)
        .value("SYMMETRIC_HEBBIAN_LINK",  Atom::Type::SYMMETRIC_HEBBIAN_LINK)
        .value("ASYMMETRIC_HEBBIAN_LINK", Atom::Type::ASYMMETRIC_HEBBIAN_LINK)
        .value("INVERSE_HEBBIAN_LINK",    Atom::Type::INVERSE_HEBBIAN_LINK)
        .value("TYPED_VARIABLE_NODE",     Atom::Type::TYPED_VARIABLE_NODE)
        .value("GLOB_NODE",               Atom::Type::GLOB_NODE)
        .export_values();

    // Atom base class
    py::class_<Atom, std::shared_ptr<Atom>>(m, "Atom")
        .def("get_type", &Atom::getType)
        .def("get_name", &Atom::getName)
        .def("get_outgoing", &Atom::getOutgoing)
        .def("get_incoming", &Atom::getIncoming)
        .def("get_arity", &Atom::getArity)
        .def("has_embedding", &Atom::hasEmbedding)
        .def("get_embedding", &Atom::getEmbedding)
        .def("set_embedding", &Atom::setEmbedding)
        .def("get_truth_value", &Atom::getTruthValue)
        .def("set_truth_value", &Atom::setTruthValue)
        .def("to_string", &Atom::toString)
        .def("__repr__", &Atom::toString);

    // Node class
    py::class_<Node, Atom, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<Atom::Type, const std::string&>(),
             py::arg("type"), py::arg("name"))
        .def(py::init<Atom::Type, const std::string&, const Tensor&>(),
             py::arg("type"), py::arg("name"), py::arg("embedding"));

    // Link class
    py::class_<Link, Atom, std::shared_ptr<Link>>(m, "Link")
        .def(py::init<Atom::Type, const std::vector<AtomPtr>&>(),
             py::arg("type"), py::arg("outgoing"));

    // AtomSpace class
    py::class_<AtomSpace>(m, "AtomSpace")
        .def(py::init<>())
        .def("add_node", static_cast<NodePtr(AtomSpace::*)(Atom::Type, const std::string&)>(&AtomSpace::addNode),
             py::arg("type"), py::arg("name"))
        .def("add_node", static_cast<NodePtr(AtomSpace::*)(Atom::Type, const std::string&, const Tensor&)>(&AtomSpace::addNode),
             py::arg("type"), py::arg("name"), py::arg("embedding"))
        .def("add_link", &AtomSpace::addLink,
             py::arg("type"), py::arg("outgoing"))
        .def("get_atom", &AtomSpace::getAtom,
             py::arg("type"), py::arg("name"))
        .def("get_atoms_by_type", &AtomSpace::getAtomsByType,
             py::arg("type"))
        .def("get_all_atoms", &AtomSpace::getAllAtoms)
        .def("get_size", &AtomSpace::getSize)
        .def("clear", &AtomSpace::clear)
        .def("query_similar", &AtomSpace::querySimilar,
             py::arg("query_embedding"), py::arg("k") = 5)
        .def("__len__", &AtomSpace::getSize);

    // Helper functions for creating atoms
    m.def("create_concept_node", 
          static_cast<NodePtr(*)(AtomSpace&, const std::string&)>(&createConceptNode),
          py::arg("space"), py::arg("name"));
    m.def("create_concept_node", 
          static_cast<NodePtr(*)(AtomSpace&, const std::string&, const Tensor&)>(&createConceptNode),
          py::arg("space"), py::arg("name"), py::arg("embedding"));
    m.def("create_predicate_node", &createPredicateNode,
          py::arg("space"), py::arg("name"));
    m.def("create_variable_node", &createVariableNode,
          py::arg("space"), py::arg("name"));
    m.def("create_inheritance_link", &createInheritanceLink,
          py::arg("space"), py::arg("child"), py::arg("parent"));
    m.def("create_evaluation_link", &createEvaluationLink,
          py::arg("space"), py::arg("predicate"), py::arg("arguments"));
    m.def("create_implication_link", &createImplicationLink,
          py::arg("space"), py::arg("antecedent"), py::arg("consequent"));
    m.def("create_and_link", &createAndLink,
          py::arg("space"), py::arg("atoms"));
    m.def("create_or_link", &createOrLink,
          py::arg("space"), py::arg("atoms"));
    m.def("create_not_link", &createNotLink,
          py::arg("space"), py::arg("atom"));

    // ============================================================
    // TIME SERVER
    // ============================================================
    
    py::class_<TimeServer>(m, "TimeServer")
        .def(py::init<>())
        .def("record_creation", &TimeServer::recordCreation,
             py::arg("atom"))
        .def("record_access", &TimeServer::recordAccess,
             py::arg("atom"))
        .def("record_modification", &TimeServer::recordModification,
             py::arg("atom"))
        .def("record_event", &TimeServer::recordEvent,
             py::arg("atom"), py::arg("event_type"))
        .def("get_creation_time", &TimeServer::getCreationTime,
             py::arg("atom"))
        .def("get_last_access_time", &TimeServer::getLastAccessTime,
             py::arg("atom"))
        .def("get_last_modification_time", &TimeServer::getLastModificationTime,
             py::arg("atom"))
        .def("get_atoms_created_between", &TimeServer::getAtomsCreatedBetween,
             py::arg("start"), py::arg("end"))
        .def("get_atoms_accessed_between", &TimeServer::getAtomsAccessedBetween,
             py::arg("start"), py::arg("end"))
        .def("get_atoms_modified_between", &TimeServer::getAtomsModifiedBetween,
             py::arg("start"), py::arg("end"));

    // ============================================================
    // ATTENTION BANK
    // ============================================================
    
    py::class_<AttentionValue>(m, "AttentionValue")
        .def(py::init<float, float, float>(),
             py::arg("sti") = 0.0f, py::arg("lti") = 0.0f, py::arg("vlti") = 0.0f)
        .def_readwrite("sti", &AttentionValue::sti)
        .def_readwrite("lti", &AttentionValue::lti)
        .def_readwrite("vlti", &AttentionValue::vlti);

    py::class_<AttentionBank>(m, "AttentionBank")
        .def(py::init<>())
        .def("set_attention_value", &AttentionBank::setAttentionValue,
             py::arg("atom"), py::arg("av"))
        .def("get_attention_value", &AttentionBank::getAttentionValue,
             py::arg("atom"))
        .def("stimulate", &AttentionBank::stimulate,
             py::arg("atom"), py::arg("amount"))
        .def("get_attentional_focus", &AttentionBank::getAttentionalFocus,
             py::arg("k") = 100)
        .def("update_attentional_focus", &AttentionBank::updateAttentionalFocus)
        .def("get_atoms_above_sti_threshold", &AttentionBank::getAtomsAboveSTIThreshold,
             py::arg("threshold"))
        .def("decay_sti", &AttentionBank::decaySTI,
             py::arg("rate") = 0.1f)
        .def("transfer_sti", &AttentionBank::transferSTI,
             py::arg("from"), py::arg("to"), py::arg("amount"));

    // ============================================================
    // SERIALIZATION
    // ============================================================
    
    py::class_<Serializer>(m, "Serializer")
        .def_static("save", &Serializer::save,
                    py::arg("space"), py::arg("filename"))
        .def_static("load", &Serializer::load,
                    py::arg("space"), py::arg("filename"))
        .def_static("to_string", &Serializer::toString,
                    py::arg("space"));

    // ============================================================
    // PLN - PATTERN MATCHING
    // ============================================================
    
    py::class_<VariableMap>(m, "VariableMap")
        .def(py::init<>())
        .def("__getitem__", [](const VariableMap& vm, const std::string& key) {
            auto it = vm.find(key);
            if (it == vm.end()) throw py::key_error("Variable not found: " + key);
            return it->second;
        })
        .def("__setitem__", [](VariableMap& vm, const std::string& key, AtomPtr value) {
            vm[key] = value;
        })
        .def("__contains__", [](const VariableMap& vm, const std::string& key) {
            return vm.find(key) != vm.end();
        })
        .def("keys", [](const VariableMap& vm) {
            std::vector<std::string> keys;
            for (const auto& pair : vm) keys.push_back(pair.first);
            return keys;
        });

    // VariableBinding: typed wrapper around unordered_map<Atom::Handle, Atom::Handle>
    // Keys are VariableNode atoms; values are the atoms they are bound to.
    // Python access is by variable name string OR by Atom handle.
    py::class_<VariableBinding>(m, "VariableBinding",
        "Typed mapping from VariableNode atoms to their bound atoms.\n\n"
        "Supports lookup by variable name (string) as well as by Atom handle.")
        .def(py::init<>())
        .def("__len__", [](const VariableBinding& vb) {
            return vb.size();
        })
        .def("__contains__",
             [](const VariableBinding& vb, const Atom::Handle& var) {
                 return vb.find(var) != vb.end();
             }, py::arg("var"),
             "Return True if the variable atom is bound.")
        .def("get",
             [](const VariableBinding& vb,
                const Atom::Handle& var,
                py::object default_) -> py::object {
                 auto it = vb.find(var);
                 if (it == vb.end()) return default_;
                 return py::cast(it->second);
             }, py::arg("var"), py::arg("default") = py::none(),
             "Return the binding for var, or default if not found.")
        .def("__getitem__",
             [](const VariableBinding& vb, const Atom::Handle& var) {
                 auto it = vb.find(var);
                 if (it == vb.end())
                     throw py::key_error("Variable not bound.");
                 return it->second;
             }, py::arg("var"),
             "Return the bound atom for a VariableNode key.")
        .def("get_by_name",
             [](const VariableBinding& vb, const std::string& name) -> py::object {
                 for (const auto& kv : vb) {
                     if (!kv.first->isNode()) continue;
                     const Node* n = static_cast<const Node*>(kv.first.get());
                     if (n->getName() == name)
                         return py::cast(kv.second);
                 }
                 return py::none();
             }, py::arg("name"),
             "Look up a binding by variable name string (e.g. '?X').")
        .def("keys",
             [](const VariableBinding& vb) {
                 std::vector<Atom::Handle> keys;
                 keys.reserve(vb.size());
                 for (const auto& kv : vb) keys.push_back(kv.first);
                 return keys;
             }, "Return a list of bound VariableNode atoms.")
        .def("values",
             [](const VariableBinding& vb) {
                 std::vector<Atom::Handle> vals;
                 vals.reserve(vb.size());
                 for (const auto& kv : vb) vals.push_back(kv.second);
                 return vals;
             }, "Return a list of atoms that variables are bound to.")
        .def("items",
             [](const VariableBinding& vb) {
                 std::vector<std::pair<Atom::Handle, Atom::Handle>> items;
                 items.reserve(vb.size());
                 for (const auto& kv : vb) items.emplace_back(kv.first, kv.second);
                 return items;
             }, "Return a list of (variable_atom, bound_atom) pairs.")
        .def("variable_names",
             [](const VariableBinding& vb) {
                 std::vector<std::string> names;
                 for (const auto& kv : vb) {
                     if (!kv.first->isNode()) continue;
                     const Node* n = static_cast<const Node*>(kv.first.get());
                     names.push_back(n->getName());
                 }
                 return names;
             }, "Return the names of all bound variables.")
        .def("to_dict",
             [](const VariableBinding& vb) {
                 py::dict d;
                 for (const auto& kv : vb) {
                     if (!kv.first->isNode()) continue;
                     const Node* n = static_cast<const Node*>(kv.first.get());
                     d[py::str(n->getName())] = kv.second;
                 }
                 return d;
             }, "Convert to a plain Python dict mapping variable names to atoms.")
        .def("__repr__",
             [](const VariableBinding& vb) {
                 std::string s = "VariableBinding({";
                 bool first = true;
                 for (const auto& kv : vb) {
                     if (!first) s += ", ";
                     first = false;
                     std::string varName = kv.first->isNode()
                         ? static_cast<const Node*>(kv.first.get())->getName()
                         : "<link>";
                     std::string valName = kv.second->isNode()
                         ? static_cast<const Node*>(kv.second.get())->getName()
                         : "<link>";
                     s += varName + " -> " + valName;
                 }
                 s += "})";
                 return s;
             });

    py::class_<PatternMatcher>(m, "PatternMatcher",
        "Static-method class for pattern matching and unification.")
        // match(pattern, target) → bool
        .def_static("match",
            [](const Atom::Handle& pattern, const Atom::Handle& target) {
                VariableBinding bindings;
                return PatternMatcher::match(pattern, target, bindings);
            }, py::arg("pattern"), py::arg("target"),
            "Return True if pattern matches target (bindings discarded).")
        // match_with_bindings(pattern, target) → (bool, dict)
        .def_static("match_with_bindings",
            [](const Atom::Handle& pattern, const Atom::Handle& target) {
                VariableBinding bindings;
                bool ok = PatternMatcher::match(pattern, target, bindings);
                return std::make_pair(ok, bindings);
            }, py::arg("pattern"), py::arg("target"),
            "Return (matched, bindings_dict) after matching pattern against target.")
        // find_matches(space, pattern) → list[(atom, bindings_dict)]
        .def_static("find_matches",
            [](AtomSpace& space, const Atom::Handle& pattern) {
                return PatternMatcher::findMatches(space, pattern);
            }, py::arg("space"), py::arg("pattern"),
            "Search the AtomSpace for all atoms matching pattern; "
            "returns list of (atom, bindings_dict).")
        // substitute(pattern, bindings, space) → atom
        .def_static("substitute",
            [](const Atom::Handle& pattern,
               const VariableBinding& bindings,
               AtomSpace& space) {
                return PatternMatcher::substitute(pattern, bindings, space);
            }, py::arg("pattern"), py::arg("bindings"), py::arg("space"),
            "Apply variable bindings to a pattern, producing a ground atom.")
        // unify(p1, p2) → (bool, bindings_dict)
        .def_static("unify",
            [](const Atom::Handle& p1, const Atom::Handle& p2) {
                VariableBinding bindings;
                bool ok = PatternMatcher::unify(p1, p2, bindings);
                return std::make_pair(ok, bindings);
            }, py::arg("pattern1"), py::arg("pattern2"),
            "Unify two patterns, returning (unified, bindings_dict).")
        .def_static("is_variable", &PatternMatcher::isVariable,
            py::arg("atom"), "True if atom is a plain VariableNode.")
        .def_static("is_typed_variable", &PatternMatcher::isTypedVariable,
            py::arg("atom"), "True if atom is a TypedVariableNode.")
        .def_static("is_glob", &PatternMatcher::isGlob,
            py::arg("atom"), "True if atom is a GlobNode.")
        .def_static("get_type_constraint", &PatternMatcher::getTypeConstraint,
            py::arg("atom"),
            "Return the type-constraint string for a TypedVariableNode.")
        // Legacy callback-based query kept for backward compatibility
        .def_static("query",
            [](AtomSpace& space,
               const Atom::Handle& pattern,
               py::function callback) {
                PatternMatcher::query(
                    space, pattern,
                    [&](const Atom::Handle& a, const VariableBinding& b) {
                        callback(a, b);
                    });
            }, py::arg("space"), py::arg("pattern"), py::arg("callback"));

    py::class_<Pattern>(m, "Pattern",
        "Helper utilities for inspecting and manipulating patterns.")
        .def_static("has_variables", &Pattern::hasVariables,
            py::arg("atom"),
            "Return True if the pattern contains any VariableNode atoms.")
        .def_static("get_variables", &Pattern::getVariables,
            py::arg("atom"),
            "Return a list of all VariableNode atoms found in the pattern.");

    // ============================================================
    // PLN - TRUTH VALUES
    // ============================================================
    
    py::class_<TruthValue>(m, "TruthValueOps")
        .def_static("deduction", &TruthValue::deduction,
                    py::arg("ab"), py::arg("bc"))
        .def_static("induction", &TruthValue::induction,
                    py::arg("ab"), py::arg("ac"))
        .def_static("abduction", &TruthValue::abduction,
                    py::arg("ab"), py::arg("cb"))
        .def_static("revision", &TruthValue::revision,
                    py::arg("tv1"), py::arg("tv2"))
        .def_static("conjunction", &TruthValue::conjunction,
                    py::arg("tv1"), py::arg("tv2"))
        .def_static("disjunction", &TruthValue::disjunction,
                    py::arg("tv1"), py::arg("tv2"))
        .def_static("negation", &TruthValue::negation,
                    py::arg("tv"))
        .def_static("implication", &TruthValue::implication,
                    py::arg("tv_p"), py::arg("tv_q"));

    // ============================================================
    // PLN - FORWARD CHAINING
    // ============================================================
    
    py::class_<InferenceRule, std::shared_ptr<InferenceRule>>(m, "InferenceRule");

    py::class_<ForwardChainer>(m, "ForwardChainer")
        .def(py::init<AtomSpace&>())
        .def("add_rule", &ForwardChainer::addRule,
             py::arg("rule"))
        .def("step", &ForwardChainer::step)
        .def("run", &ForwardChainer::run,
             py::arg("max_iterations") = 100)
        .def("get_results", &ForwardChainer::getResults);

    // ============================================================
    // PLN - BACKWARD CHAINING
    // ============================================================
    
    py::class_<ProofNode, std::shared_ptr<ProofNode>>(m, "ProofNode")
        .def_readonly("atom", &ProofNode::atom)
        .def_readonly("premises", &ProofNode::premises)
        .def_readonly("rule_name", &ProofNode::ruleName);

    py::class_<BackwardChainer>(m, "BackwardChainer")
        .def(py::init<AtomSpace&>())
        .def("prove", &BackwardChainer::prove,
             py::arg("goal"), py::arg("max_depth") = 10)
        .def("get_proof_tree", &BackwardChainer::getProofTree)
        .def("get_truth_value", &BackwardChainer::getTruthValue,
             py::arg("goal"));

    // ============================================================
    // ECAN - ECONOMIC ATTENTION NETWORKS
    // ============================================================
    
    py::class_<HebbianLink>(m, "HebbianLink")
        .def_readwrite("atom1", &HebbianLink::atom1)
        .def_readwrite("atom2", &HebbianLink::atom2)
        .def_readwrite("asymmetric_weight", &HebbianLink::asymmetricWeight)
        .def_readwrite("symmetric_weight", &HebbianLink::symmetricWeight)
        .def("update", &HebbianLink::update,
             py::arg("sti1"), py::arg("sti2"));

    py::class_<ImportanceSpreading>(m, "ImportanceSpreading")
        .def(py::init<>())
        .def("spread", &ImportanceSpreading::spread,
             py::arg("bank"), py::arg("hebbian_links"), py::arg("amount") = 1.0f);

    py::class_<ForgettingAgent>(m, "ForgettingAgent")
        .def(py::init<float>(), py::arg("threshold") = -100.0f)
        .def("forget", &ForgettingAgent::forget,
             py::arg("space"), py::arg("bank"))
        .def("get_forgotten_count", &ForgettingAgent::getForgottenCount);

    py::class_<RentAgent>(m, "RentAgent")
        .def(py::init<float>(), py::arg("rent_rate") = 1.0f)
        .def("collect_rent", &RentAgent::collectRent,
             py::arg("bank"));

    py::class_<WageAgent>(m, "WageAgent")
        .def(py::init<float>(), py::arg("wage_rate") = 1.0f)
        .def("pay_wages", &WageAgent::payWages,
             py::arg("bank"), py::arg("used_atoms"));

    // ============================================================
    // TENSOR LOGIC ENGINE
    // ============================================================
    
    py::class_<TensorLogicEngine>(m, "TensorLogicEngine")
        .def(py::init<>())
        .def("batch_and", &TensorLogicEngine::batchAND,
             py::arg("truth_values"))
        .def("batch_or", &TensorLogicEngine::batchOR,
             py::arg("truth_values"))
        .def("batch_not", &TensorLogicEngine::batchNOT,
             py::arg("truth_values"))
        .def("batch_implies", &TensorLogicEngine::batchIMPLIES,
             py::arg("premises"), py::arg("conclusions"))
        .def("batch_deduction", &TensorLogicEngine::batchDeduction,
             py::arg("ab_tvs"), py::arg("bc_tvs"))
        .def("batch_similarity", &TensorLogicEngine::batchSimilarity,
             py::arg("embeddings1"), py::arg("embeddings2"))
        .def("batch_pattern_match", &TensorLogicEngine::batchPatternMatch,
             py::arg("patterns"), py::arg("targets"))
        .def("compute_statistics", &TensorLogicEngine::computeStatistics,
             py::arg("truth_values"))
        .def("filter_by_confidence", &TensorLogicEngine::filterByConfidence,
             py::arg("atoms"), py::arg("threshold"));

    // ============================================================
    // COGNITIVE ENGINE
    // ============================================================
    
    py::class_<CognitiveEngine>(m, "CognitiveEngine")
        .def(py::init<AtomSpace&>())
        .def("set_time_server", &CognitiveEngine::setTimeServer,
             py::arg("ts"))
        .def("set_attention_bank", &CognitiveEngine::setAttentionBank,
             py::arg("ab"))
        .def("set_forward_chainer", &CognitiveEngine::setForwardChainer,
             py::arg("fc"))
        .def("set_backward_chainer", &CognitiveEngine::setBackwardChainer,
             py::arg("bc"))
        .def("enable_forgetting", &CognitiveEngine::enableForgetting,
             py::arg("threshold") = -100.0f)
        .def("enable_economic_attention", &CognitiveEngine::enableEconomicAttention,
             py::arg("rent_rate") = 1.0f, py::arg("wage_rate") = 1.0f)
        .def("cognitive_cycle", &CognitiveEngine::cognitiveCycle,
             py::arg("inference_steps") = 10)
        .def("run", &CognitiveEngine::run,
             py::arg("num_cycles") = 100, py::arg("inference_per_cycle") = 10)
        .def("get_cycle_count", &CognitiveEngine::getCycleCount)
        .def("get_metrics", &CognitiveEngine::getMetrics);

    // ============================================================
    // NLU - NATURAL LANGUAGE UNDERSTANDING
    // ============================================================
    
    py::class_<Token>(m, "Token")
        .def_readonly("text", &Token::text)
        .def_readonly("pos_tag", &Token::posTag)
        .def_readonly("start", &Token::start)
        .def_readonly("end", &Token::end);

    py::class_<TextProcessor>(m, "TextProcessor")
        .def(py::init<>())
        .def("tokenize", &TextProcessor::tokenize,
             py::arg("text"))
        .def("normalize", &TextProcessor::normalize,
             py::arg("text"))
        .def("extract_sentences", &TextProcessor::extractSentences,
             py::arg("text"));

    py::class_<Entity>(m, "Entity")
        .def_readonly("text", &Entity::text)
        .def_readonly("type", &Entity::type)
        .def_readonly("start", &Entity::start)
        .def_readonly("end", &Entity::end)
        .def_readonly("confidence", &Entity::confidence);

    py::class_<EntityRecognizer>(m, "EntityRecognizer")
        .def(py::init<>())
        .def("recognize", &EntityRecognizer::recognize,
             py::arg("text"))
        .def("recognize_with_model", &EntityRecognizer::recognizeWithModel,
             py::arg("text"), py::arg("model_fn"));

    py::class_<Relation>(m, "Relation")
        .def_readonly("subject", &Relation::subject)
        .def_readonly("predicate", &Relation::predicate)
        .def_readonly("object", &Relation::object_)
        .def_readonly("confidence", &Relation::confidence);

    py::class_<RelationExtractor>(m, "RelationExtractor")
        .def(py::init<>())
        .def("extract", &RelationExtractor::extract,
             py::arg("text"), py::arg("entities"));

    py::class_<SemanticExtractor>(m, "SemanticExtractor")
        .def(py::init<AtomSpace&>())
        .def("extract_from_text", &SemanticExtractor::extractFromText,
             py::arg("text"))
        .def("extract_from_text_with_embeddings", 
             &SemanticExtractor::extractFromTextWithEmbeddings,
             py::arg("text"), py::arg("embedding_fn"));

    py::class_<LanguageGenerator>(m, "LanguageGenerator")
        .def(py::init<AtomSpace&>())
        .def("generate_from_atom", &LanguageGenerator::generateFromAtom,
             py::arg("atom"))
        .def("generate_summary", &LanguageGenerator::generateSummary,
             py::arg("atoms"));

    // ============================================================
    // VISION - VISUAL PERCEPTION
    // ============================================================
    
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<float, float, float, float>(),
             py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"))
        .def_readwrite("x", &BoundingBox::x)
        .def_readwrite("y", &BoundingBox::y)
        .def_readwrite("width", &BoundingBox::width)
        .def_readwrite("height", &BoundingBox::height)
        .def("center_x", &BoundingBox::centerX)
        .def("center_y", &BoundingBox::centerY)
        .def("area", &BoundingBox::area)
        .def("iou", &BoundingBox::iou,
             py::arg("other"));

    py::class_<DetectedObject>(m, "DetectedObject")
        .def(py::init<const std::string&, const BoundingBox&, float>(),
             py::arg("label"), py::arg("bbox"), py::arg("confidence") = 1.0f)
        .def_readonly("label", &DetectedObject::label)
        .def_readonly("bbox", &DetectedObject::bbox)
        .def_readonly("confidence", &DetectedObject::confidence)
        .def_readwrite("features", &DetectedObject::features);

    py::class_<SpatialRelation>(m, "SpatialRelation")
        .def_readonly("object1", &SpatialRelation::object1)
        .def_readonly("object2", &SpatialRelation::object2)
        .def_readonly("relation_type", &SpatialRelation::relationType)
        .def_readonly("confidence", &SpatialRelation::confidence);

    py::class_<SpatialAnalyzer>(m, "SpatialAnalyzer")
        .def(py::init<>())
        .def("analyze_spatial_relations", &SpatialAnalyzer::analyzeSpatialRelations,
             py::arg("objects"));

    py::class_<SceneUnderstanding>(m, "SceneUnderstanding")
        .def(py::init<AtomSpace&>())
        .def("build_scene_graph", &SceneUnderstanding::buildSceneGraph,
             py::arg("objects"), py::arg("relations"))
        .def("describe_scene", &SceneUnderstanding::describeScene,
             py::arg("objects"), py::arg("relations"));

    py::class_<VisualReasoning>(m, "VisualReasoning")
        .def(py::init<AtomSpace&>())
        .def("ground_concept", &VisualReasoning::groundConcept,
             py::arg("concept_name"), py::arg("visual_examples"))
        .def("query_visual_knowledge", &VisualReasoning::queryVisualKnowledge,
             py::arg("query_concept"));

    py::class_<VisionProcessor>(m, "VisionProcessor")
        .def(py::init<AtomSpace&>())
        .def("process_image", &VisionProcessor::processImage,
             py::arg("image"), py::arg("detector_fn"))
        .def("process_video", &VisionProcessor::processVideo,
             py::arg("frames"), py::arg("detector_fn"), py::arg("time_server"));

    py::class_<MultimodalIntegration>(m, "MultimodalIntegration")
        .def(py::init<AtomSpace&>())
        .def("caption_image", &MultimodalIntegration::captionImage,
             py::arg("objects"), py::arg("relations"))
        .def("answer_visual_question", &MultimodalIntegration::answerVisualQuestion,
             py::arg("question"), py::arg("visual_atoms"));

    // ============================================================
    // MODEL LOADER - Phase 8
    // ============================================================
    
    py::class_<nn::LoadedModelConfig>(m, "LoadedModelConfig")
        .def(py::init<>())
        .def_readwrite("model_name", &nn::LoadedModelConfig::model_name)
        .def_readwrite("hidden_size", &nn::LoadedModelConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &nn::LoadedModelConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &nn::LoadedModelConfig::num_attention_heads)
        .def_readwrite("vocab_size", &nn::LoadedModelConfig::vocab_size)
        .def_readwrite("max_seq_length", &nn::LoadedModelConfig::max_seq_length)
        .def_readwrite("max_position_embeddings", &nn::LoadedModelConfig::max_position_embeddings)
        .def_readwrite("type_vocab_size", &nn::LoadedModelConfig::type_vocab_size);

    py::class_<nn::TorchScriptModel, std::shared_ptr<nn::TorchScriptModel>>(m, "TorchScriptModel")
        .def("forward", &nn::TorchScriptModel::forward,
             py::arg("inputs"))
        .def("to", &nn::TorchScriptModel::to,
             py::arg("device"))
        .def("get_device", &nn::TorchScriptModel::getDevice);

    py::class_<nn::ModelLoader>(m, "ModelLoader")
        .def(py::init<>())
        .def("load_torchscript_model", &nn::ModelLoader::loadTorchScriptModel,
             py::arg("model_path"),
             py::arg("device") = torch::kCPU,
             py::arg("use_cache") = true)
        .def("load_model_config", &nn::ModelLoader::loadModelConfig,
             py::arg("config_path"))
        .def("clear_cache", &nn::ModelLoader::clearCache)
        .def_static("model_exists", &nn::ModelLoader::modelExists,
                    py::arg("model_path"))
        .def("get_default_device", &nn::ModelLoader::getDefaultDevice);

    // Helper functions for loading specific model types
    m.def("load_bert_model", &nn::loadBERTModel,
          py::arg("model_path") = "models/bert_base.pt");
    m.def("load_gpt2_model", &nn::loadGPT2Model,
          py::arg("model_path") = "models/gpt2.pt");
    m.def("load_vit_model", &nn::loadViTModel,
          py::arg("model_path") = "models/vit_base.pt");
    m.def("load_yolo_model", &nn::loadYOLOModel,
          py::arg("model_path") = "models/yolov5.pt");

    // ============================================================
    // TOKENIZER - Phase 8 (tokenization support)
    // ============================================================

    py::class_<nn::Vocabulary>(m, "Vocabulary")
        .def(py::init<>())
        .def("load_from_txt",  &nn::Vocabulary::loadFromTxt,
             py::arg("path"))
        .def("load_from_json", &nn::Vocabulary::loadFromJson,
             py::arg("path"))
        .def("token_to_id", &nn::Vocabulary::tokenToId,
             py::arg("token"), py::arg("unk_id") = 100)
        .def("id_to_token", &nn::Vocabulary::idToToken,
             py::arg("id"))
        .def("size",     &nn::Vocabulary::size)
        .def("contains", &nn::Vocabulary::contains,
             py::arg("token"));

    py::class_<nn::WordPieceTokenizer, std::shared_ptr<nn::WordPieceTokenizer>>(
            m, "WordPieceTokenizer")
        .def("tokenize", &nn::WordPieceTokenizer::tokenize,
             py::arg("text"))
        .def("encode",
             &nn::WordPieceTokenizer::encode,
             py::arg("text"),
             py::arg("max_length")         = 512,
             py::arg("add_special_tokens") = true)
        .def("encode_to_bert_tensors",
             &nn::WordPieceTokenizer::encodeToBertTensors,
             py::arg("text"), py::arg("max_length") = 512)
        .def("batch_encode",
             &nn::WordPieceTokenizer::batchEncode,
             py::arg("texts"), py::arg("max_length") = 512);

    py::class_<nn::BPETokenizer, std::shared_ptr<nn::BPETokenizer>>(m, "BPETokenizer")
        .def("tokenize", &nn::BPETokenizer::tokenize,
             py::arg("text"))
        .def("encode", &nn::BPETokenizer::encode,
             py::arg("text"))
        .def("decode", &nn::BPETokenizer::decode,
             py::arg("ids"))
        .def("encode_to_gpt_tensors",
             &nn::BPETokenizer::encodeToGPTTensors,
             py::arg("text"), py::arg("max_length") = 1024);

    m.def("load_bert_tokenizer",
          &nn::TokenizerFactory::loadBertTokenizer,
          py::arg("dir"),
          "Load a WordPieceTokenizer (BERT) from a directory containing vocab.txt");

    m.def("load_gpt2_tokenizer",
          &nn::TokenizerFactory::loadGPT2Tokenizer,
          py::arg("dir"),
          "Load a BPETokenizer (GPT-2) from a directory containing vocab.json + merges.txt");

    // Module version
    m.attr("__version__") = "0.13.0";

    // ============================================================
    // PHASE 9 + 10 BINDINGS
    // ============================================================

    // ---- QueryEngine -------------------------------------------

    py::class_<QueryClause>(m, "QueryClause",
        "A single pattern clause for use in a conjunctive query.")
        .def(py::init<Atom::Handle, bool>(),
             py::arg("pattern"), py::arg("optional") = false)
        .def_readwrite("optional", &QueryClause::optional)
        .def_readwrite("match_any_type", &QueryClause::matchAnyType);

    py::class_<QueryEngine>(m, "QueryEngine",
        "Advanced multi-pattern conjunctive query engine over the hypergraph.")
        .def(py::init<AtomSpace&>(), py::arg("space"))
        .def("find_matches", &QueryEngine::findMatches,
             py::arg("pattern"),
             "Find all atoms matching a single pattern (returns list of VariableBinding dicts).")
        .def("execute_conjunctive",
             [](const QueryEngine& qe,
                const std::vector<QueryClause>& clauses,
                size_t maxResults) {
                 return qe.executeConjunctive(clauses, {}, maxResults);
             },
             py::arg("clauses"), py::arg("max_results") = 0)
        .def("find_by_type", &QueryEngine::findByType,
             py::arg("type"))
        .def("find_by_truth_strength", &QueryEngine::findByTruthStrength,
             py::arg("min_strength"), py::arg("min_confidence") = 0.0f)
        .def("find_similar", &QueryEngine::findSimilar,
             py::arg("embedding"), py::arg("top_k") = 10)
        .def("neighbourhood", &QueryEngine::neighbourhood,
             py::arg("seed"), py::arg("depth") = 1)
        .def("count",
             [](const QueryEngine& qe, const Atom::Handle& pat) {
                 return qe.count(pat);
             }, py::arg("pattern"))
        .def("exists",
             [](const QueryEngine& qe, const Atom::Handle& pat) {
                 return qe.exists(pat);
             }, py::arg("pattern"));

    py::class_<QueryBuilder>(m, "QueryBuilder",
        "Fluent builder for constructing and executing hypergraph queries.")
        .def(py::init<AtomSpace&>(), py::arg("space"))
        .def("match",
             [](QueryBuilder& qb, const Atom::Handle& pat) -> QueryBuilder& {
                 return qb.match(pat);
             }, py::arg("pattern"), py::return_value_policy::reference)
        .def("optional_match",
             [](QueryBuilder& qb, const Atom::Handle& pat) -> QueryBuilder& {
                 return qb.optionalMatch(pat);
             }, py::arg("pattern"), py::return_value_policy::reference)
        .def("not_match",
             [](QueryBuilder& qb, const Atom::Handle& pat) -> QueryBuilder& {
                 return qb.notMatch(pat);
             }, py::arg("pattern"), py::return_value_policy::reference)
        .def("filter",
             [](QueryBuilder& qb, py::function pred) -> QueryBuilder& {
                 return qb.filter([pred](const QueryResult& row) -> bool {
                     return pred(row).cast<bool>();
                 });
             }, py::arg("predicate"), py::return_value_policy::reference)
        .def("filter_by_strength",
             [](QueryBuilder& qb,
                const Atom::Handle& var,
                float minS) -> QueryBuilder& {
                 return qb.filterByStrength(var, minS);
             }, py::arg("var"), py::arg("min_strength"),
             py::return_value_policy::reference)
        .def("filter_by_confidence",
             [](QueryBuilder& qb,
                const Atom::Handle& var,
                float minC) -> QueryBuilder& {
                 return qb.filterByConfidence(var, minC);
             }, py::arg("var"), py::arg("min_confidence"),
             py::return_value_policy::reference)
        .def("limit",
             [](QueryBuilder& qb, size_t n) -> QueryBuilder& {
                 return qb.limit(n);
             }, py::arg("n"), py::return_value_policy::reference)
        .def("execute",         &QueryBuilder::execute)
        .def("execute_with_negation", &QueryBuilder::executeWithNegation)
        .def("count",           &QueryBuilder::count);

    // Phase 10 factory helpers
    m.def("create_typed_variable_node", &createTypedVariableNode,
          py::arg("space"), py::arg("var_name"), py::arg("constraint_type_name"),
          "Create a TypedVariableNode that only binds to atoms of the given type name.");
    m.def("create_glob_node", &createGlobNode,
          py::arg("space"), py::arg("name") = "@",
          "Create a GlobNode (sequence wildcard) for use inside link patterns.");

    // ---- BinarySerializer --------------------------------------

    // Expose as module-level functions (all methods are static)
    m.def("save_atomspace",
          [](const AtomSpace& space, const std::string& path) {
              return BinarySerializer::save(space, path);
          },
          py::arg("space"), py::arg("path"),
          "Serialize AtomSpace to a binary file. Returns True on success.");

    m.def("load_atomspace",
          [](AtomSpace& space, const std::string& path) {
              return BinarySerializer::load(space, path);
          },
          py::arg("space"), py::arg("path"),
          "Deserialize AtomSpace from a binary file. Returns True on success.");

    m.def("serialize_atomspace",
          [](const AtomSpace& space) -> py::bytes {
              auto buf = BinarySerializer::serialize(space);
              return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
          },
          py::arg("space"),
          "Serialize AtomSpace to an in-memory bytes object.");

    m.def("deserialize_atomspace",
          [](AtomSpace& space, py::bytes data) {
              std::string s = data;
              std::vector<uint8_t> buf(s.begin(), s.end());
              BinarySerializer::deserialize(space, buf);
          },
          py::arg("space"), py::arg("data"),
          "Deserialize AtomSpace from a bytes object.");

    // ---- InferencePipeline -------------------------------------

    py::class_<StepStats>(m, "StepStats",
        "Per-step statistics for an InferencePipeline run.")
        .def_readonly("step_name",       &StepStats::stepName)
        .def_readonly("produced",        &StepStats::produced)
        .def_readonly("working_set_size",&StepStats::workingSetSize)
        .def_readonly("elapsed_ms",      &StepStats::elapsedMs);

    py::class_<PipelineResult>(m, "PipelineResult",
        "Result of a full InferencePipeline run.")
        .def_readonly("atoms",          &PipelineResult::atoms)
        .def_readonly("stats",          &PipelineResult::stats)
        .def_readonly("converged",      &PipelineResult::converged)
        .def_readonly("iterations_run", &PipelineResult::iterationsRun)
        .def("total_ms",                &PipelineResult::totalMs);

    // InferenceStep abstract base with Python trampoline
    {
        struct PyInferenceStep : public InferenceStep {
            using InferenceStep::InferenceStep;
            std::string getName() const override {
                PYBIND11_OVERRIDE_PURE(std::string, InferenceStep, getName,);
            }
            bool execute(std::vector<Atom::Handle>& workingSet,
                         AtomSpace& space) override {
                PYBIND11_OVERRIDE_PURE(bool, InferenceStep, execute,
                                       workingSet, space);
            }
        };
        py::class_<InferenceStep, PyInferenceStep, std::shared_ptr<InferenceStep>>(
                m, "InferenceStep",
                "Abstract base class for a single inference pipeline step.")
            .def(py::init<>())
            .def("get_name", &InferenceStep::getName)
            .def("execute",  &InferenceStep::execute,
                 py::arg("working_set"), py::arg("space"));
    }

    py::class_<ForwardChainingStep, InferenceStep,
               std::shared_ptr<ForwardChainingStep>>(m, "ForwardChainingStep")
        .def(py::init<int>(), py::arg("max_rounds") = 1);

    py::class_<BackwardChainingStep, InferenceStep,
               std::shared_ptr<BackwardChainingStep>>(m, "BackwardChainingStep")
        .def(py::init<Atom::Handle, int>(),
             py::arg("goal"), py::arg("max_depth") = 5);

    py::class_<TruthValueThresholdStep, InferenceStep,
               std::shared_ptr<TruthValueThresholdStep>>(m, "TruthValueThresholdStep")
        .def(py::init<float, float>(),
             py::arg("min_strength"), py::arg("min_confidence") = 0.0f);

    py::class_<PatternMatchStep, InferenceStep,
               std::shared_ptr<PatternMatchStep>>(m, "PatternMatchStep")
        .def(py::init<Atom::Handle>(), py::arg("pattern"));

    py::class_<AttentionBoostStep, InferenceStep,
               std::shared_ptr<AttentionBoostStep>>(m, "AttentionBoostStep")
        .def(py::init<AttentionBank&, float>(),
             py::arg("bank"), py::arg("boost") = 10.0f);

    py::class_<FilterStep, InferenceStep,
               std::shared_ptr<FilterStep>>(m, "FilterStep",
        "Filter atoms from the working set using a Python predicate.")
        .def(py::init([](const std::string& name, py::function pred) {
                 return std::make_shared<FilterStep>(
                     name,
                     [pred](const Atom::Handle& a) -> bool {
                         return pred(a).cast<bool>();
                     });
             }),
             py::arg("name"), py::arg("predicate"));

    py::class_<CustomStep, InferenceStep,
               std::shared_ptr<CustomStep>>(m, "CustomStep",
        "Wrap an arbitrary Python callable as an InferenceStep.")
        .def(py::init([](const std::string& name, py::function fn) {
                 return std::make_shared<CustomStep>(
                     name,
                     [fn](std::vector<Atom::Handle>& ws, AtomSpace& sp) -> bool {
                         return fn(ws, sp).cast<bool>();
                     });
             }),
             py::arg("name"), py::arg("fn"));

    // ============================================================
    // PLN pipeline steps (Phase 11)
    // ============================================================

    py::class_<PLNDeductionStep, InferenceStep,
               std::shared_ptr<PLNDeductionStep>>(m, "PLNDeductionStep",
        "Apply PLN deduction (A→B, B→C ⊢ A→C) to link pairs in the working set.")
        .def(py::init<float>(), py::arg("min_confidence") = 0.0f);

    py::class_<PLNRevisionStep, InferenceStep,
               std::shared_ptr<PLNRevisionStep>>(m, "PLNRevisionStep",
        "Merge truth values of structurally identical atoms via PLN revision.")
        .def(py::init<>());

    py::class_<PLNAbductionStep, InferenceStep,
               std::shared_ptr<PLNAbductionStep>>(m, "PLNAbductionStep",
        "Infer explanatory atoms using PLN abduction (B, A→B ⊢ A).")
        .def(py::init<float, float>(),
             py::arg("min_observation_strength") = 0.7f,
             py::arg("min_confidence") = 0.0f);

    py::class_<PLNInductionStep, InferenceStep,
               std::shared_ptr<PLNInductionStep>>(m, "PLNInductionStep",
        "Generalise from instances using PLN induction, emitting MemberLinks.")
        .def(py::init<Atom::Type>(),
             py::arg("link_type") = Atom::Type::INHERITANCE_LINK);

    // Phase 12 PLN steps
    py::class_<PLNConjunctionStep, InferenceStep,
               std::shared_ptr<PLNConjunctionStep>>(m, "PLNConjunctionStep",
        "Compute AND_LINK truth values using the PLN conjunction formula.")
        .def(py::init<float>(),
             py::arg("min_strength") = 0.5f);

    py::class_<PLNDisjunctionStep, InferenceStep,
               std::shared_ptr<PLNDisjunctionStep>>(m, "PLNDisjunctionStep",
        "Compute OR_LINK truth values using the PLN disjunction formula.")
        .def(py::init<float>(),
             py::arg("min_strength") = 0.3f);

    py::class_<PLNSimilarityStep, InferenceStep,
               std::shared_ptr<PLNSimilarityStep>>(m, "PLNSimilarityStep",
        "Create SIMILARITY_LINK atoms for cognate atoms above a similarity threshold.")
        .def(py::init<float>(),
             py::arg("min_similarity") = 0.5f);

    // Phase 13 PLN steps
    py::class_<PLNImplicationStep, InferenceStep,
               std::shared_ptr<PLNImplicationStep>>(m, "PLNImplicationStep",
        "Evaluate and create IMPLICATION_LINK truth values using the PLN "
        "material-implication formula (s = 1 - sA + sA*sB, c = min(cA,cB)).")
        .def(py::init<float, float>(),
             py::arg("min_antecedent_strength")  = 0.5f,
             py::arg("min_implication_strength") = 0.5f);

    py::class_<PLNImplicationChainStep, InferenceStep,
               std::shared_ptr<PLNImplicationChainStep>>(m, "PLNImplicationChainStep",
        "Compute the multi-hop transitive closure of IMPLICATION_LINKs up to "
        "max_depth hops, deriving new links via the PLN deduction formula.")
        .def(py::init<int, float>(),
             py::arg("max_depth")          = 3,
             py::arg("min_chain_confidence") = 0.0f);

    py::class_<InferencePipeline>(m, "InferencePipeline",
        "Composable, ordered sequence of inference steps.")
        .def(py::init<AtomSpace&>(), py::arg("space"))
        .def("add_step",
             [](InferencePipeline& p,
                std::shared_ptr<InferenceStep> step) -> InferencePipeline& {
                 return p.addStep(std::move(step));
             }, py::arg("step"), py::return_value_policy::reference)
        .def("forward_chain",
             [](InferencePipeline& p, int rounds) -> InferencePipeline& {
                 return p.forwardChain(rounds);
             }, py::arg("rounds") = 1, py::return_value_policy::reference)
        .def("backward_chain",
             [](InferencePipeline& p,
                Atom::Handle goal,
                int depth) -> InferencePipeline& {
                 return p.backwardChain(std::move(goal), depth);
             }, py::arg("goal"), py::arg("max_depth") = 5,
             py::return_value_policy::reference)
        .def("match_pattern",
             [](InferencePipeline& p,
                Atom::Handle pat) -> InferencePipeline& {
                 return p.matchPattern(std::move(pat));
             }, py::arg("pattern"), py::return_value_policy::reference)
        .def("filter_by_tv",
             [](InferencePipeline& p,
                float minS, float minC) -> InferencePipeline& {
                 return p.filterByTV(minS, minC);
             }, py::arg("min_strength"), py::arg("min_confidence") = 0.0f,
             py::return_value_policy::reference)
        .def("run",
             [](InferencePipeline& p,
                std::vector<Atom::Handle> seeds,
                bool untilFixed,
                int maxIter) {
                 return p.run(std::move(seeds), untilFixed, maxIter);
             },
             py::arg("seeds") = std::vector<Atom::Handle>{},
             py::arg("until_fixed_point") = false,
             py::arg("max_iterations") = 100)
        .def("size", &InferencePipeline::size)
        .def("step_names", &InferencePipeline::stepNames)
        .def("clear",
             [](InferencePipeline& p) -> InferencePipeline& {
                 return p.clear();
             }, py::return_value_policy::reference)
        // PLN step shortcuts (Phase 11)
        .def("pln_deduction",
             [](InferencePipeline& p, float minConf) -> InferencePipeline& {
                 return p.plnDeduction(minConf);
             }, py::arg("min_confidence") = 0.0f,
             py::return_value_policy::reference,
             "Append a PLNDeductionStep.")
        .def("pln_revision",
             [](InferencePipeline& p) -> InferencePipeline& {
                 return p.plnRevision();
             }, py::return_value_policy::reference,
             "Append a PLNRevisionStep.")
        .def("pln_abduction",
             [](InferencePipeline& p,
                float minObs, float minConf) -> InferencePipeline& {
                 return p.plnAbduction(minObs, minConf);
             }, py::arg("min_observation_strength") = 0.7f,
             py::arg("min_confidence") = 0.0f,
             py::return_value_policy::reference,
             "Append a PLNAbductionStep.")
        .def("pln_induction",
             [](InferencePipeline& p,
                Atom::Type lt) -> InferencePipeline& {
                 return p.plnInduction(lt);
             }, py::arg("link_type") = Atom::Type::INHERITANCE_LINK,
             py::return_value_policy::reference,
             "Append a PLNInductionStep.")
        // PLN step shortcuts (Phase 12)
        .def("pln_conjunction",
             [](InferencePipeline& p, float minS) -> InferencePipeline& {
                 return p.plnConjunction(minS);
             }, py::arg("min_strength") = 0.5f,
             py::return_value_policy::reference,
             "Append a PLNConjunctionStep.")
        .def("pln_disjunction",
             [](InferencePipeline& p, float minS) -> InferencePipeline& {
                 return p.plnDisjunction(minS);
             }, py::arg("min_strength") = 0.3f,
             py::return_value_policy::reference,
             "Append a PLNDisjunctionStep.")
        .def("pln_similarity",
             [](InferencePipeline& p, float minSim) -> InferencePipeline& {
                 return p.plnSimilarity(minSim);
             }, py::arg("min_similarity") = 0.5f,
             py::return_value_policy::reference,
             "Append a PLNSimilarityStep.")
        // PLN step shortcuts (Phase 13)
        .def("pln_implication",
             [](InferencePipeline& p,
                float minAnt, float minImp) -> InferencePipeline& {
                 return p.plnImplication(minAnt, minImp);
             },
             py::arg("min_antecedent_strength")  = 0.5f,
             py::arg("min_implication_strength") = 0.5f,
             py::return_value_policy::reference,
             "Append a PLNImplicationStep.")
        .def("pln_implication_chain",
             [](InferencePipeline& p,
                int maxDepth, float minConf) -> InferencePipeline& {
                 return p.plnImplicationChain(maxDepth, minConf);
             },
             py::arg("max_depth")           = 3,
             py::arg("min_chain_confidence") = 0.0f,
             py::return_value_policy::reference,
             "Append a PLNImplicationChainStep.");

    m.def("make_forward_reasoning_pipeline",
          &makeForwardReasoningPipeline,
          py::arg("space"), py::arg("seed_pattern"),
          py::arg("tv_threshold") = 0.5f, py::arg("fc_rounds") = 3,
          "Create a standard forward-reasoning pipeline.");

    m.def("make_hypothesis_verification_pipeline",
          &makeHypothesisVerificationPipeline,
          py::arg("space"), py::arg("goal"),
          py::arg("min_confidence") = 0.6f, py::arg("max_depth") = 5,
          "Create a hypothesis verification pipeline.");

    m.def("make_pln_reasoning_pipeline",
          &makePLNReasoningPipeline,
          py::arg("space"),
          py::arg("tv_threshold") = 0.0f,
          py::arg("min_confidence") = 0.0f,
          "Create a PLN reasoning pipeline (deduction → revision → filter).");

    m.def("make_pln_full_pipeline",
          &makePLNFullPipeline,
          py::arg("space"),
          py::arg("tv_threshold")    = 0.0f,
          py::arg("min_confidence")  = 0.0f,
          py::arg("min_strength")    = 0.3f,
          py::arg("min_similarity")  = 0.5f,
          "Create a full PLN pipeline (deduction→conjunction→disjunction→similarity→revision→filter).");

    m.def("make_pln_complete_pipeline",
          &makePLNCompletePipeline,
          py::arg("space"),
          py::arg("tv_threshold")          = 0.0f,
          py::arg("min_confidence")        = 0.0f,
          py::arg("min_strength")          = 0.3f,
          py::arg("min_similarity")        = 0.5f,
          py::arg("min_antecedent_strength")  = 0.5f,
          py::arg("min_implication_strength") = 0.5f,
          py::arg("chain_max_depth")       = 3,
          "Create a complete PLN pipeline (deduction→conjunction→disjunction→"
          "similarity→implication→implication_chain→revision→filter).");

    // ---- HebbianLearner ----------------------------------------

    py::class_<HebbianLearnerConfig>(m, "HebbianLearnerConfig",
        "Configuration for HebbianLearner.")
        .def(py::init<>())
        .def_readwrite("learning_rate",     &HebbianLearnerConfig::learningRate)
        .def_readwrite("decay_rate",        &HebbianLearnerConfig::decayRate)
        .def_readwrite("prune_threshold",   &HebbianLearnerConfig::pruneThreshold)
        .def_readwrite("max_strength",      &HebbianLearnerConfig::maxStrength)
        .def_readwrite("asymmetric",        &HebbianLearnerConfig::asymmetric)
        .def_readwrite("oja_rule",          &HebbianLearnerConfig::ojaRule)
        .def_readwrite("min_activation_sti",&HebbianLearnerConfig::minActivationSTI);

    py::class_<HebbianLearner>(m, "HebbianLearner",
        "Associative learning from co-activation (Hebbian learning).")
        .def(py::init<AtomSpace&, AttentionBank&>(),
             py::arg("space"), py::arg("bank"))
        .def(py::init<AtomSpace&, AttentionBank&, const HebbianLearnerConfig&>(),
             py::arg("space"), py::arg("bank"), py::arg("config"))
        .def("record_co_activation", &HebbianLearner::recordCoActivation,
             py::arg("source"), py::arg("target"), py::arg("amount") = 1.0f)
        .def("learn_from_attentional_focus",
             &HebbianLearner::learnFromAttentionalFocus)
        .def("decay", &HebbianLearner::decay)
        .def("run_cycles", &HebbianLearner::runCycles, py::arg("cycles") = 1)
        .def("reset", &HebbianLearner::reset)
        .def("get_link", &HebbianLearner::getLink,
             py::arg("a"), py::arg("b"))
        .def("get_strength", &HebbianLearner::getStrength,
             py::arg("a"), py::arg("b"))
        .def("get_associates", &HebbianLearner::getAssociates,
             py::arg("atom"), py::arg("min_strength") = 0.0f)
        .def("get_all_hebbian_links", &HebbianLearner::getAllHebbianLinks)
        .def("total_co_activations", &HebbianLearner::totalCoActivations)
        .def("co_activation_count", &HebbianLearner::coActivationCount,
             py::arg("a"), py::arg("b"))
        .def("get_config", &HebbianLearner::getConfig,
             py::return_value_policy::reference)
        .def("set_config", &HebbianLearner::setConfig, py::arg("config"));

}
