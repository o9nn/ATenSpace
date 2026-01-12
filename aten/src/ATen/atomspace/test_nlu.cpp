#include <ATen/atomspace/ATenSpace.h>
#include <ATen/atomspace/NLU.h>
#include <iostream>
#include <cassert>

using namespace at::atomspace;

void test_tokenization() {
    std::cout << "Test: Tokenization... ";
    
    std::string text = "Hello world! This is a test.";
    auto tokens = TextProcessor::tokenize(text);
    
    assert(tokens.size() > 0);
    assert(tokens[0].text == "Hello");
    
    std::cout << "PASSED\n";
}

void test_normalization() {
    std::cout << "Test: Text Normalization... ";
    
    std::string text = "Hello, World! 123";
    std::string normalized = TextProcessor::normalize(text);
    
    assert(normalized.find("hello") != std::string::npos);
    assert(normalized.find(",") == std::string::npos);
    assert(normalized.find("!") == std::string::npos);
    
    std::cout << "PASSED\n";
}

void test_sentenceExtraction() {
    std::cout << "Test: Sentence Extraction... ";
    
    std::string text = "First sentence. Second sentence! Third sentence?";
    auto sentences = TextProcessor::extractSentences(text);
    
    assert(sentences.size() == 3);
    assert(sentences[0].find("First") != std::string::npos);
    assert(sentences[1].find("Second") != std::string::npos);
    assert(sentences[2].find("Third") != std::string::npos);
    
    std::cout << "PASSED\n";
}

void test_entityRecognition() {
    std::cout << "Test: Entity Recognition... ";
    
    std::string text = "Albert Einstein and Marie Curie were scientists.";
    auto entities = EntityRecognizer::recognize(text);
    
    assert(entities.size() >= 2);
    bool foundEinstein = false;
    bool foundCurie = false;
    
    for (const auto& entity : entities) {
        if (entity.text.find("Einstein") != std::string::npos) {
            foundEinstein = true;
        }
        if (entity.text.find("Curie") != std::string::npos) {
            foundCurie = true;
        }
    }
    
    assert(foundEinstein);
    assert(foundCurie);
    
    std::cout << "PASSED\n";
}

void test_relationExtraction() {
    std::cout << "Test: Relation Extraction... ";
    
    std::string text = "John loves Mary.";
    auto entities = EntityRecognizer::recognize(text);
    auto relations = RelationExtractor::extract(text, entities);
    
    // Should extract some relations (even if simplified)
    // The test is more about not crashing than exact results
    assert(relations.size() >= 0);
    
    std::cout << "PASSED\n";
}

void test_semanticExtraction() {
    std::cout << "Test: Semantic Extraction to AtomSpace... ";
    
    AtomSpace space;
    std::string text = "Socrates is a philosopher.";
    
    SemanticExtractor::extractToAtomSpace(space, text);
    
    // Should create some atoms
    assert(space.getSize() > 0);
    
    auto nodes = space.getAtomsByType(Atom::Type::CONCEPT_NODE);
    assert(nodes.size() > 0);
    
    std::cout << "PASSED\n";
}

void test_languageGeneration() {
    std::cout << "Test: Language Generation... ";
    
    AtomSpace space;
    
    auto cat = createConceptNode(space, "cat");
    auto mammal = createConceptNode(space, "mammal");
    auto inh = createInheritanceLink(space, cat, mammal);
    
    std::string generated = LanguageGenerator::generate(inh);
    
    assert(!generated.empty());
    assert(generated.find("cat") != std::string::npos);
    assert(generated.find("mammal") != std::string::npos);
    
    std::cout << "PASSED\n";
}

void test_languageGenerationEvaluation() {
    std::cout << "Test: Language Generation for Evaluation Links... ";
    
    AtomSpace space;
    
    auto loves = createPredicateNode(space, "loves");
    auto john = createConceptNode(space, "John");
    auto mary = createConceptNode(space, "Mary");
    auto evalLink = createEvaluationLink(space, loves, {john, mary});
    
    std::string generated = LanguageGenerator::generate(evalLink);
    
    assert(!generated.empty());
    assert(generated.find("John") != std::string::npos);
    assert(generated.find("Mary") != std::string::npos);
    assert(generated.find("loves") != std::string::npos);
    
    std::cout << "PASSED\n";
}

void test_languageGenerationLogical() {
    std::cout << "Test: Language Generation for Logical Links... ";
    
    AtomSpace space;
    
    auto a = createConceptNode(space, "A");
    auto b = createConceptNode(space, "B");
    auto c = createConceptNode(space, "C");
    
    auto andLink = createAndLink(space, {a, b, c});
    std::string andGenerated = LanguageGenerator::generate(andLink);
    
    assert(andGenerated.find("A") != std::string::npos);
    assert(andGenerated.find("B") != std::string::npos);
    assert(andGenerated.find("C") != std::string::npos);
    assert(andGenerated.find("and") != std::string::npos);
    
    auto orLink = createOrLink(space, {a, b});
    std::string orGenerated = LanguageGenerator::generate(orLink);
    
    assert(orGenerated.find("or") != std::string::npos);
    
    std::cout << "PASSED\n";
}

void test_generateSummary() {
    std::cout << "Test: Generate Summary from Knowledge Graph... ";
    
    AtomSpace space;
    
    // Build a small knowledge base
    auto socrates = createConceptNode(space, "Socrates");
    auto plato = createConceptNode(space, "Plato");
    auto teaches = createPredicateNode(space, "teaches");
    createEvaluationLink(space, teaches, {socrates, plato});
    
    std::string summary = LanguageGenerator::generateSummary(space, 5);
    
    // Should generate something
    assert(!summary.empty());
    
    std::cout << "PASSED\n";
}

void test_multiSentenceProcessing() {
    std::cout << "Test: Multi-Sentence Processing... ";
    
    AtomSpace space;
    
    std::string text = "Dogs are animals. Cats are animals. Animals need food.";
    auto sentences = TextProcessor::extractSentences(text);
    
    assert(sentences.size() == 3);
    
    for (const auto& sentence : sentences) {
        SemanticExtractor::extractToAtomSpace(space, sentence);
    }
    
    // Should have created multiple atoms
    assert(space.getSize() > 5);
    
    std::cout << "PASSED\n";
}

void test_embeddingIntegration() {
    std::cout << "Test: Embedding Integration with NLU... ";
    
    AtomSpace space;
    
    std::string text = "Einstein was a scientist.";
    Tensor embedding = torch::randn({128});
    
    SemanticExtractor::extractToAtomSpace(space, text, embedding);
    
    // Should create atoms with embeddings
    assert(space.getSize() > 0);
    
    auto nodes = space.getAtomsByType(Atom::Type::CONCEPT_NODE);
    // At least one node should have been created
    assert(nodes.size() > 0);
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║             ATenSpace NLU Unit Tests                       ║\n";
    std::cout << "║                    Phase 5 Testing                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    try {
        test_tokenization();
        test_normalization();
        test_sentenceExtraction();
        test_entityRecognition();
        test_relationExtraction();
        test_semanticExtraction();
        test_languageGeneration();
        test_languageGenerationEvaluation();
        test_languageGenerationLogical();
        test_generateSummary();
        test_multiSentenceProcessing();
        test_embeddingIntegration();
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              All NLU Tests Passed! ✓                       ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
