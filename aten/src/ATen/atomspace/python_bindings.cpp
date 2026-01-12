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
 * 
 * Phase 6 - Production Integration
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <torch/extension.h>

#include "ATenSpace.h"

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

    py::class_<PatternMatcher>(m, "PatternMatcher")
        .def(py::init<AtomSpace&>())
        .def("match", &PatternMatcher::match,
             py::arg("pattern"), py::arg("target"))
        .def("query", &PatternMatcher::query,
             py::arg("pattern"), py::arg("callback"));

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

    // Module version
    m.attr("__version__") = "0.6.0";
}
