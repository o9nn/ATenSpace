#include <ATen/atomspace/ATenSpace.h>
#include <ATen/atomspace/NLU.h>
#include <iostream>
#include <iomanip>

using namespace at::atomspace;

void printSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

/**
 * Example 1: Text Tokenization
 */
void example1_tokenization() {
    printSeparator("Example 1: Text Tokenization");
    
    std::string text = "The quick brown fox jumps over the lazy dog.";
    std::cout << "Text: " << text << "\n\n";
    
    auto tokens = TextProcessor::tokenize(text);
    
    std::cout << "Tokens:\n";
    for (const auto& token : tokens) {
        std::cout << "  " << std::setw(15) << std::left << token.text 
                  << " | Lemma: " << std::setw(15) << token.lemma
                  << " | POS: " << token.pos << "\n";
    }
}

/**
 * Example 2: Named Entity Recognition
 */
void example2_ner() {
    printSeparator("Example 2: Named Entity Recognition");
    
    std::string text = "Albert Einstein worked in Princeton. Marie Curie was a scientist.";
    std::cout << "Text: " << text << "\n\n";
    
    auto entities = EntityRecognizer::recognize(text);
    
    std::cout << "Recognized Entities:\n";
    for (const auto& entity : entities) {
        std::cout << "  " << std::setw(20) << std::left << entity.text
                  << " | Type: " << std::setw(10) << entity.type
                  << " | Confidence: " << entity.confidence << "\n";
    }
}

/**
 * Example 3: Relation Extraction
 */
void example3_relations() {
    printSeparator("Example 3: Relation Extraction");
    
    std::string text = "John loves Mary. Sarah teaches mathematics.";
    std::cout << "Text: " << text << "\n\n";
    
    // Extract entities first
    auto entities = EntityRecognizer::recognize(text);
    
    // Extract relations
    auto relations = RelationExtractor::extract(text, entities);
    
    std::cout << "Extracted Relations:\n";
    for (const auto& rel : relations) {
        std::cout << "  " << rel.subject << " --[" << rel.predicate << "]--> " 
                  << rel.object << " (confidence: " << rel.confidence << ")\n";
    }
}

/**
 * Example 4: Semantic Extraction to Knowledge Graph
 */
void example4_semanticExtraction() {
    printSeparator("Example 4: Semantic Extraction to Knowledge Graph");
    
    AtomSpace space;
    
    std::string text = "Socrates is a philosopher. Plato studied under Socrates. "
                       "Aristotle learned from Plato.";
    std::cout << "Text: " << text << "\n\n";
    
    // Extract knowledge to AtomSpace
    SemanticExtractor::extractToAtomSpace(space, text);
    
    std::cout << "Knowledge Graph Statistics:\n";
    std::cout << "  Total atoms: " << space.getSize() << "\n";
    
    auto nodes = space.getAtomsByType(Atom::Type::CONCEPT_NODE);
    std::cout << "  Concept nodes: " << nodes.size() << "\n";
    for (const auto& node : nodes) {
        std::cout << "    - " << node->getName() << "\n";
    }
    
    auto evalLinks = space.getAtomsByType(Atom::Type::EVALUATION_LINK);
    std::cout << "\n  Evaluation links (relations): " << evalLinks.size() << "\n";
    for (const auto& link : evalLinks) {
        const auto& outgoing = link->getOutgoing();
        if (outgoing.size() == 2) {
            auto predicate = outgoing[0];
            auto args = outgoing[1];
            std::cout << "    - Relation: " << predicate->getName() << "\n";
        }
    }
}

/**
 * Example 5: Language Generation from Knowledge Graph
 */
void example5_generation() {
    printSeparator("Example 5: Language Generation from Knowledge Graph");
    
    AtomSpace space;
    
    // Build a simple knowledge graph
    auto socrates = createConceptNode(space, "Socrates");
    auto philosopher = createConceptNode(space, "philosopher");
    auto human = createConceptNode(space, "human");
    
    auto inh1 = createInheritanceLink(space, socrates, philosopher);
    auto inh2 = createInheritanceLink(space, philosopher, human);
    
    std::cout << "Knowledge Graph:\n";
    std::cout << "  " << LanguageGenerator::generate(inh1) << "\n";
    std::cout << "  " << LanguageGenerator::generate(inh2) << "\n";
    
    // Create evaluation link
    auto teaches = createPredicateNode(space, "teaches");
    auto plato = createConceptNode(space, "Plato");
    auto evalLink = createEvaluationLink(space, teaches, {socrates, plato});
    
    std::cout << "\n  " << LanguageGenerator::generate(evalLink) << "\n";
}

/**
 * Example 6: Multi-Sentence Processing
 */
void example6_multiSentence() {
    printSeparator("Example 6: Multi-Sentence Processing");
    
    AtomSpace space;
    
    std::string text = 
        "Cats are mammals. Dogs are mammals. "
        "Mammals are animals. Animals need food.";
    
    std::cout << "Text: " << text << "\n\n";
    
    // Extract sentences
    auto sentences = TextProcessor::extractSentences(text);
    std::cout << "Extracted " << sentences.size() << " sentences:\n";
    for (const auto& sentence : sentences) {
        std::cout << "  - " << sentence << "\n";
    }
    
    // Process each sentence
    std::cout << "\nBuilding knowledge graph from sentences...\n";
    for (const auto& sentence : sentences) {
        SemanticExtractor::extractToAtomSpace(space, sentence);
    }
    
    std::cout << "\nFinal Knowledge Graph:\n";
    std::cout << "  Total atoms: " << space.getSize() << "\n";
    std::cout << "  Concepts: " << space.getAtomsByType(Atom::Type::CONCEPT_NODE).size() << "\n";
    std::cout << "  Relations: " << space.getAtomsByType(Atom::Type::EVALUATION_LINK).size() << "\n";
    
    // Generate summary
    std::cout << "\nGenerated Summary:\n";
    std::string summary = LanguageGenerator::generateSummary(space, 3);
    std::cout << "  " << summary << "\n";
}

/**
 * Example 7: Knowledge Graph Querying with NLU
 */
void example7_querying() {
    printSeparator("Example 7: Knowledge Graph Querying with NLU");
    
    AtomSpace space;
    
    // Build knowledge base
    std::string knowledge = 
        "Paris is the capital of France. "
        "London is the capital of England. "
        "Berlin is the capital of Germany.";
    
    std::cout << "Knowledge Base:\n" << knowledge << "\n\n";
    SemanticExtractor::extractToAtomSpace(space, knowledge);
    
    // Query the knowledge
    std::cout << "Querying knowledge graph...\n\n";
    
    // Find all capitals
    auto evalLinks = space.getAtomsByType(Atom::Type::EVALUATION_LINK);
    std::cout << "All relations in knowledge base:\n";
    for (const auto& link : evalLinks) {
        std::cout << "  " << LanguageGenerator::generate(link) << "\n";
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          ATenSpace Natural Language Understanding          ║\n";
    std::cout << "║                      Phase 5 Examples                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    
    try {
        example1_tokenization();
        example2_ner();
        example3_relations();
        example4_semanticExtraction();
        example5_generation();
        example6_multiSentence();
        example7_querying();
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║            All NLU Examples Completed Successfully!        ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
