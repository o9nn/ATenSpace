#pragma once

#include "Atom.h"
#include "AtomSpace.h"
#include "TruthValue.h"
#include <ATen/ATen.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <regex>

namespace at {
namespace atomspace {

/**
 * NLU (Natural Language Understanding) - Language processing for ATenSpace
 * 
 * This module provides natural language understanding capabilities,
 * enabling conversion between text and knowledge graph representations.
 * It integrates with transformer embeddings and semantic extraction.
 * 
 * Key features:
 * - Text tokenization and preprocessing
 * - Semantic extraction to knowledge graphs
 * - Language generation from knowledge graphs
 * - Integration with neural embeddings
 * - Named entity recognition
 * - Relation extraction
 */

/**
 * Token - Represents a linguistic token
 */
struct Token {
    std::string text;
    std::string lemma;
    std::string pos;        // Part of speech
    size_t startIdx;
    size_t endIdx;
    
    Token(const std::string& t, const std::string& l, const std::string& p, 
          size_t start, size_t end)
        : text(t), lemma(l), pos(p), startIdx(start), endIdx(end) {}
};

/**
 * Entity - Represents a named entity
 */
struct Entity {
    std::string text;
    std::string type;       // PERSON, LOCATION, ORGANIZATION, etc.
    size_t startIdx;
    size_t endIdx;
    float confidence;
    
    Entity(const std::string& t, const std::string& ty, 
           size_t start, size_t end, float conf = 1.0f)
        : text(t), type(ty), startIdx(start), endIdx(end), confidence(conf) {}
};

/**
 * Relation - Represents a semantic relation between entities
 */
struct Relation {
    std::string subject;
    std::string predicate;
    std::string object;
    float confidence;
    
    Relation(const std::string& subj, const std::string& pred, 
             const std::string& obj, float conf = 1.0f)
        : subject(subj), predicate(pred), object(obj), confidence(conf) {}
};

/**
 * TextProcessor - Tokenization and preprocessing
 */
class TextProcessor {
public:
    /**
     * Tokenize text into tokens
     */
    static std::vector<Token> tokenize(const std::string& text) {
        std::vector<Token> tokens;
        
        // Simple whitespace and punctuation tokenization
        std::regex wordRegex(R"(\w+|[^\w\s])");
        auto wordsBegin = std::sregex_iterator(text.begin(), text.end(), wordRegex);
        auto wordsEnd = std::sregex_iterator();
        
        for (auto it = wordsBegin; it != wordsEnd; ++it) {
            std::smatch match = *it;
            std::string word = match.str();
            size_t pos = match.position();
            
            // Simple lemmatization (lowercase)
            std::string lemma = toLowerCase(word);
            
            // Simple POS tagging (simplified)
            std::string posTag = inferPOS(word);
            
            tokens.emplace_back(word, lemma, posTag, pos, pos + word.length());
        }
        
        return tokens;
    }
    
    /**
     * Normalize text (lowercase, remove special chars, etc.)
     */
    static std::string normalize(const std::string& text) {
        std::string normalized;
        for (char c : text) {
            if (std::isalnum(c) || std::isspace(c)) {
                normalized += std::tolower(c);
            } else if (c == '-' || c == '_') {
                normalized += c;
            }
        }
        return normalized;
    }
    
    /**
     * Extract sentences from text
     */
    static std::vector<std::string> extractSentences(const std::string& text) {
        std::vector<std::string> sentences;
        std::regex sentenceRegex(R"([^.!?]+[.!?]+)");
        auto sentencesBegin = std::sregex_iterator(text.begin(), text.end(), sentenceRegex);
        auto sentencesEnd = std::sregex_iterator();
        
        for (auto it = sentencesBegin; it != sentencesEnd; ++it) {
            std::string sentence = it->str();
            // Trim whitespace
            sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
            sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);
            if (!sentence.empty()) {
                sentences.push_back(sentence);
            }
        }
        
        // If no sentences found, treat whole text as one sentence
        if (sentences.empty() && !text.empty()) {
            sentences.push_back(text);
        }
        
        return sentences;
    }

private:
    static std::string toLowerCase(const std::string& str) {
        std::string result;
        for (char c : str) {
            result += std::tolower(c);
        }
        return result;
    }
    
    static std::string inferPOS(const std::string& word) {
        // Very simplified POS tagging
        if (word.length() == 1 && !std::isalnum(word[0])) {
            return "PUNCT";
        }
        if (word.length() > 2 && word.substr(word.length() - 2) == "ed") {
            return "VERB";
        }
        if (word.length() > 3 && word.substr(word.length() - 3) == "ing") {
            return "VERB";
        }
        if (word.length() > 2 && word.substr(word.length() - 2) == "ly") {
            return "ADV";
        }
        return "NOUN";  // Default
    }
};

/**
 * EntityRecognizer - Named entity recognition
 */
class EntityRecognizer {
public:
    /**
     * Extract named entities from text
     */
    static std::vector<Entity> recognize(const std::string& text) {
        std::vector<Entity> entities;
        
        // Simple pattern-based NER
        // This is a simplified version - real NER would use ML models
        
        // Recognize capitalized words as potential entities
        std::regex entityRegex(R"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)");
        auto entitiesBegin = std::sregex_iterator(text.begin(), text.end(), entityRegex);
        auto entitiesEnd = std::sregex_iterator();
        
        for (auto it = entitiesBegin; it != entitiesEnd; ++it) {
            std::smatch match = *it;
            std::string entityText = match.str();
            size_t pos = match.position();
            
            // Simple type inference based on patterns
            std::string type = inferEntityType(entityText);
            
            entities.emplace_back(entityText, type, pos, 
                                  pos + entityText.length(), 0.8f);
        }
        
        return entities;
    }

private:
    static std::string inferEntityType(const std::string& entity) {
        // Very simplified entity type inference
        // Real systems would use gazetteers, ML models, etc.
        
        // Common person name patterns
        if (entity.find(" ") != std::string::npos) {
            return "PERSON";
        }
        
        // Single capitalized word - could be anything
        return "ENTITY";
    }
};

/**
 * RelationExtractor - Extract semantic relations from text
 */
class RelationExtractor {
public:
    /**
     * Extract relations from text using entities
     */
    static std::vector<Relation> extract(const std::string& text, 
                                          const std::vector<Entity>& entities) {
        std::vector<Relation> relations;
        
        // Simple pattern-based relation extraction
        // This is simplified - real systems would use dependency parsing, ML, etc.
        
        // Look for simple patterns: Entity1 verb Entity2
        for (size_t i = 0; i < entities.size(); ++i) {
            for (size_t j = i + 1; j < entities.size(); ++j) {
                const auto& ent1 = entities[i];
                const auto& ent2 = entities[j];
                
                // Extract text between entities
                if (ent1.endIdx < ent2.startIdx) {
                    size_t start = ent1.endIdx;
                    size_t end = ent2.startIdx;
                    std::string between = text.substr(start, end - start);
                    
                    // Extract verb as predicate
                    std::string predicate = extractVerb(between);
                    if (!predicate.empty()) {
                        relations.emplace_back(ent1.text, predicate, ent2.text, 0.7f);
                    }
                }
            }
        }
        
        return relations;
    }

private:
    static std::string extractVerb(const std::string& text) {
        // Simple verb extraction - look for common verb patterns
        // Note: This is a simplified heuristic. Real implementation would use
        // dependency parsing or ML-based POS tagging
        std::regex verbRegex(R"(\b\w*(?:ed|ing)\b)");  // Past tense and gerund forms
        auto verbsBegin = std::sregex_iterator(text.begin(), text.end(), verbRegex);
        auto verbsEnd = std::sregex_iterator();
        
        for (auto it = verbsBegin; it != verbsEnd; ++it) {
            std::string word = it->str();
            // Filter out obvious non-verbs
            if (word.length() > 3) {
                return TextProcessor::normalize(word);
            }
        }
        
        // Fallback: return empty string if no verb found
        return "";
    }
};

/**
 * SemanticExtractor - Convert text to knowledge graphs
 */
class SemanticExtractor {
public:
    /**
     * Extract knowledge from text and add to AtomSpace
     */
    static void extractToAtomSpace(AtomSpace& space, const std::string& text,
                                    const Tensor& textEmbedding = Tensor()) {
        // Tokenize text
        auto tokens = TextProcessor::tokenize(text);
        
        // Recognize entities
        auto entities = EntityRecognizer::recognize(text);
        
        // Extract relations
        auto relations = RelationExtractor::extract(text, entities);
        
        // Add entities to AtomSpace as ConceptNodes
        std::unordered_map<std::string, Atom::Handle> entityNodes;
        size_t entityCounter = 0;
        
        for (const auto& entity : entities) {
            Atom::Handle node;
            if (textEmbedding.defined() && textEmbedding.numel() > 0) {
                // Use provided embedding
                node = space.addNode(Atom::Type::CONCEPT_NODE, entity.text, textEmbedding);
            } else {
                node = space.addNode(Atom::Type::CONCEPT_NODE, entity.text);
            }
            
            // Set truth value based on confidence
            node->setTruthValue(TruthValue::create(entity.confidence, 0.9f));
            
            // Create unique key using entity text and position for disambiguation
            std::string key = entity.text + "_" + std::to_string(entity.startIdx) + "_" + 
                              std::to_string(entityCounter++);
            entityNodes[key] = node;
        }
        
        // Add relations to AtomSpace
        for (const auto& relation : relations) {
            // Get or create nodes
            auto subjectNode = entityNodes.count(relation.subject) > 0
                ? entityNodes[relation.subject]
                : space.addNode(Atom::Type::CONCEPT_NODE, relation.subject);
                
            auto objectNode = entityNodes.count(relation.object) > 0
                ? entityNodes[relation.object]
                : space.addNode(Atom::Type::CONCEPT_NODE, relation.object);
            
            // Create predicate
            auto predicateNode = space.addNode(Atom::Type::PREDICATE_NODE, relation.predicate);
            
            // Create evaluation link: predicate(subject, object)
            auto listLink = space.addLink(Atom::Type::LIST_LINK, {subjectNode, objectNode});
            auto evalLink = space.addLink(Atom::Type::EVALUATION_LINK, {predicateNode, listLink});
            
            // Set truth value based on confidence
            evalLink->setTruthValue(TruthValue::create(relation.confidence, 0.8f));
        }
    }
    
    /**
     * Extract knowledge with custom embedding function
     */
    static void extractToAtomSpace(AtomSpace& space, const std::string& text,
                                    std::function<Tensor(const std::string&)> embeddingFn) {
        // Get embedding for the text
        Tensor textEmbedding = embeddingFn(text);
        extractToAtomSpace(space, text, textEmbedding);
    }
};

/**
 * LanguageGenerator - Generate text from knowledge graphs
 */
class LanguageGenerator {
public:
    /**
     * Generate natural language from an atom
     */
    static std::string generate(Atom::Handle atom) {
        if (!atom) {
            return "";
        }
        
        // Handle different atom types
        if (atom->isNode()) {
            return atom->getName();
        }
        
        // Handle links
        const auto& outgoing = atom->getOutgoing();
        
        switch (atom->getType()) {
            case Atom::Type::INHERITANCE_LINK:
                if (outgoing.size() == 2) {
                    return generate(outgoing[0]) + " is a " + generate(outgoing[1]);
                }
                break;
                
            case Atom::Type::EVALUATION_LINK:
                if (outgoing.size() == 2) {
                    auto predicate = generate(outgoing[0]);
                    auto args = outgoing[1];
                    if (args->getType() == Atom::Type::LIST_LINK) {
                        const auto& argList = args->getOutgoing();
                        if (argList.size() == 2) {
                            return generate(argList[0]) + " " + predicate + " " + generate(argList[1]);
                        }
                    }
                }
                break;
                
            case Atom::Type::AND_LINK:
                {
                    std::string result;
                    for (size_t i = 0; i < outgoing.size(); ++i) {
                        if (i > 0) {
                            result += (i == outgoing.size() - 1) ? " and " : ", ";
                        }
                        result += generate(outgoing[i]);
                    }
                    return result;
                }
                break;
                
            case Atom::Type::OR_LINK:
                {
                    std::string result;
                    for (size_t i = 0; i < outgoing.size(); ++i) {
                        if (i > 0) {
                            result += (i == outgoing.size() - 1) ? " or " : ", ";
                        }
                        result += generate(outgoing[i]);
                    }
                    return result;
                }
                break;
                
            default:
                // Generic link representation
                {
                    std::string result = atom->toString() + "(";
                    for (size_t i = 0; i < outgoing.size(); ++i) {
                        if (i > 0) result += ", ";
                        result += generate(outgoing[i]);
                    }
                    result += ")";
                    return result;
                }
        }
        
        return atom->toString();
    }
    
    /**
     * Generate a description of the knowledge graph
     */
    static std::string generateSummary(AtomSpace& space, size_t maxSentences = 5) {
        std::vector<std::string> sentences;
        
        // Generate sentences from important atoms
        auto atoms = space.getAtomsByType(Atom::Type::EVALUATION_LINK);
        
        // Sort by attention/importance if available
        // For now, just take first maxSentences
        size_t count = 0;
        for (const auto& atom : atoms) {
            if (count >= maxSentences) break;
            
            std::string sentence = generate(atom);
            if (!sentence.empty()) {
                sentences.push_back(sentence + ".");
                count++;
            }
        }
        
        // Combine sentences
        std::string summary;
        for (const auto& sentence : sentences) {
            summary += sentence + " ";
        }
        
        return summary;
    }
};

} // namespace atomspace
} // namespace at
