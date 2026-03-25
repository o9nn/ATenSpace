/**
 * test_tokenization.cpp — Tokenization Support Tests (Phase 8)
 *
 * Validates:
 *   - Vocabulary loading (txt and json)
 *   - WordPieceTokenizer: tokenize / encode / encodeToBertTensors / batchEncode
 *   - BPETokenizer: tokenize / encode / decode / encodeToGPTTensors
 *   - BERTModel::encodeText() integration
 *   - GPTModel::generateText() integration
 *   - Graceful error handling when tokenizer files are absent
 */

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cassert>
#include <fstream>

// Bring in LibTorch so the tensor helpers in Tokenizer.h are compiled.
#include <torch/torch.h>

#include "Tokenizer.h"
#include "PretrainedModels.h"

// ============================================================================
// Helpers
// ============================================================================

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(name, body)                                        \
    do {                                                            \
        ++tests_run;                                                \
        std::cout << "  [TEST] " << name << " ... " << std::flush; \
        try {                                                        \
            body                                                    \
            std::cout << "PASS\n";                                  \
            ++tests_passed;                                         \
        } catch (const std::exception& e) {                        \
            std::cout << "FAIL\n";                                  \
            std::cerr << "    Error: " << e.what() << "\n";        \
            ++tests_failed;                                         \
        }                                                           \
    } while (0)

// Write a minimal vocab.txt for testing (BERT-style)
static void writeMiniVocabTxt(const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    // id 0 = [PAD], 100 = [UNK], 101 = [CLS], 102 = [SEP], 103 = [MASK]
    const std::vector<std::string> tokens = {
        "[PAD]",                              // 0
        "[unused1]","[unused2]","[unused3]",  // 1-3
        "[unused4]","[unused5]","[unused6]",  // 4-6
        "[unused7]","[unused8]","[unused9]",  // 7-9
        "[unused10]","[unused11]","[unused12]",// 10-12
        "[unused13]","[unused14]","[unused15]",// 13-15
        "[unused16]","[unused17]","[unused18]",// 16-18
        "[unused19]","[unused20]","[unused21]",// 19-21
        "[unused22]","[unused23]","[unused24]",// 22-24
        "[unused25]","[unused26]","[unused27]",// 25-27
        "[unused28]","[unused29]","[unused30]",// 28-30
        "[unused31]","[unused32]","[unused33]",// 31-33
        "[unused34]","[unused35]","[unused36]",// 34-36
        "[unused37]","[unused38]","[unused39]",// 37-39
        "[unused40]","[unused41]","[unused42]",// 40-42
        "[unused43]","[unused44]","[unused45]",// 43-45
        "[unused46]","[unused47]","[unused48]",// 46-48
        "[unused49]","[unused50]","[unused51]",// 49-51
        "[unused52]","[unused53]","[unused54]",// 52-54
        "[unused55]","[unused56]","[unused57]",// 55-57
        "[unused58]","[unused59]","[unused60]",// 58-60
        "[unused61]","[unused62]","[unused63]",// 61-63
        "[unused64]","[unused65]","[unused66]",// 64-66
        "[unused67]","[unused68]","[unused69]",// 67-69
        "[unused70]","[unused71]","[unused72]",// 70-72
        "[unused73]","[unused74]","[unused75]",// 73-75
        "[unused76]","[unused77]","[unused78]",// 76-78
        "[unused79]","[unused80]","[unused81]",// 79-81
        "[unused82]","[unused83]","[unused84]",// 82-84
        "[unused85]","[unused86]","[unused87]",// 85-87
        "[unused88]","[unused89]","[unused90]",// 88-90
        "[unused91]","[unused92]","[unused93]",// 91-93
        "[unused94]","[unused95]","[unused96]",// 94-96
        "[unused97]","[unused98]","[unused99]",// 97-99
        "[UNK]",                              // 100
        "[CLS]",                              // 101
        "[SEP]",                              // 102
        "[MASK]",                             // 103
        "hello",                              // 104
        ",",                                  // 105
        "how",                                // 106
        "are",                                // 107
        "you",                                // 108
        "?",                                  // 109
        "the",                                // 110
        "quick",                              // 111
        "brown",                              // 112
        "fox",                                // 113
        "##s",                                // 114
        "world",                              // 115
    };
    for (const auto& t : tokens) f << t << "\n";
}

// Write a minimal vocab.json for GPT-2 testing
static void writeMiniVocabJson(const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    f << "{\n";
    // Map a handful of byte-encoded characters and the Ġ-prefixed word forms
    // that the BPE tokenizer will produce for "The quick brown fox".
    // Ġ = U+0120 = UTF-8 \xC4\xA0
    f << "  \"T\": 0,\n";
    f << "  \"h\": 1,\n";
    f << "  \"e\": 2,\n";
    f << "  \"Ġquick\": 3,\n";
    f << "  \"Ġbrown\": 4,\n";
    f << "  \"Ġfox\": 5,\n";
    f << "  \"The\": 6,\n";
    f << "  \"<|endoftext|>\": 50256\n";
    f << "}\n";
}

// Write a minimal merges.txt for GPT-2 testing
static void writeMiniMergesTxt(const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    f << "#version: 0.2\n";
    // Merge T + h → Th, then Th + e → The
    f << "T h\n";
    f << "Th e\n";
}

// ============================================================================
// Test sections
// ============================================================================

static void testVocabulary() {
    std::cout << "\n--- Vocabulary ---\n";

    RUN_TEST("loadFromTxt missing file throws", {
        at::atomspace::nn::Vocabulary vocab;
        bool threw = false;
        try { vocab.loadFromTxt("/nonexistent/vocab.txt"); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw && "Expected runtime_error for missing file");
    });

    RUN_TEST("loadFromTxt succeeds and maps tokens", {
        at::atomspace::nn::Vocabulary vocab;
        writeMiniVocabTxt("/tmp/test_vocab.txt");
        vocab.loadFromTxt("/tmp/test_vocab.txt");
        assert(vocab.size() > 0);
        assert(vocab.tokenToId("[CLS]") == 101);
        assert(vocab.tokenToId("[SEP]") == 102);
        assert(vocab.tokenToId("[PAD]") == 0);
        assert(vocab.tokenToId("[UNK]") == 100);
        assert(vocab.idToToken(101) == "[CLS]");
        std::cout << "  vocab size = " << vocab.size() << "\n";
    });

    RUN_TEST("tokenToId returns unk_id for unknown token", {
        at::atomspace::nn::Vocabulary vocab;
        vocab.loadFromTxt("/tmp/test_vocab.txt");
        assert(vocab.tokenToId("zzzunknown", 42) == 42);
    });

    RUN_TEST("loadFromJson succeeds", {
        at::atomspace::nn::Vocabulary vocab;
        writeMiniVocabJson("/tmp/test_vocab.json");
        vocab.loadFromJson("/tmp/test_vocab.json");
        assert(vocab.size() > 0);
        assert(vocab.tokenToId("<|endoftext|>", 0) == 50256);
        assert(vocab.idToToken(50256) == "<|endoftext|>");
        std::cout << "  vocab size = " << vocab.size() << "\n";
    });
}

static void testWordPieceTokenizer() {
    std::cout << "\n--- WordPieceTokenizer ---\n";

    // Build tokenizer
    at::atomspace::nn::Vocabulary vocab;
    writeMiniVocabTxt("/tmp/test_vocab.txt");
    vocab.loadFromTxt("/tmp/test_vocab.txt");
    at::atomspace::nn::WordPieceTokenizer tokenizer(std::move(vocab));

    RUN_TEST("tokenize simple phrase", {
        auto tokens = tokenizer.tokenize("Hello, how are you?");
        std::cout << "  tokens: ";
        for (const auto& t : tokens) std::cout << "[" << t << "] ";
        std::cout << "\n";
        assert(!tokens.empty());
    });

    RUN_TEST("encode adds CLS and SEP", {
        auto ids = tokenizer.encode("hello", 512, true);
        assert(ids.front() == 101);  // [CLS]
        assert(ids.back()  == 102);  // [SEP]
        std::cout << "  ids:";
        for (int id : ids) std::cout << " " << id;
        std::cout << "\n";
    });

    RUN_TEST("encode without special tokens", {
        auto ids = tokenizer.encode("hello", 512, false);
        assert(ids.front() != 101);
        assert(ids.back()  != 102);
    });

    RUN_TEST("encode respects max_length", {
        std::string long_text = "hello hello hello hello hello hello hello hello "
                                "hello hello hello hello hello hello hello hello";
        auto ids = tokenizer.encode(long_text, 8, true);
        assert(static_cast<int>(ids.size()) <= 8);
    });

    RUN_TEST("encodeToBertTensors returns correct shapes", {
        auto [input_ids, attention_mask, token_type_ids] =
            tokenizer.encodeToBertTensors("hello world", 16);
        assert(input_ids.sizes()      == torch::IntArrayRef({1, 16}));
        assert(attention_mask.sizes() == torch::IntArrayRef({1, 16}));
        assert(token_type_ids.sizes() == torch::IntArrayRef({1, 16}));
        // First token should be [CLS]=101
        assert(input_ids[0][0].item<int64_t>() == 101);
        std::cout << "  shapes: " << input_ids.sizes() << "\n";
    });

    RUN_TEST("batchEncode returns correct batch size", {
        std::vector<std::string> texts = {"hello", "how are you", "the quick brown fox"};
        auto [input_ids, attention_mask, token_type_ids] =
            tokenizer.batchEncode(texts, 16);
        assert(input_ids.size(0) == 3);
        assert(input_ids.size(1) == 16);
        std::cout << "  shapes: " << input_ids.sizes() << "\n";
    });

    RUN_TEST("unknown token maps to UNK id", {
        auto ids = tokenizer.encode("zzznonsenseword", 512, false);
        // Should contain [UNK]=100
        bool has_unk = false;
        for (int id : ids) if (id == 100) has_unk = true;
        assert(has_unk);
    });
}

static void testBPETokenizer() {
    std::cout << "\n--- BPETokenizer ---\n";

    // Build tokenizer
    at::atomspace::nn::Vocabulary vocab;
    writeMiniVocabJson("/tmp/test_vocab.json");
    vocab.loadFromJson("/tmp/test_vocab.json");

    writeMiniMergesTxt("/tmp/test_merges.txt");
    std::ifstream mf("/tmp/test_merges.txt");
    std::vector<std::string> merges;
    std::string line;
    while (std::getline(mf, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        merges.push_back(line);
    }

    at::atomspace::nn::BPETokenizer tokenizer(std::move(vocab), std::move(merges));

    RUN_TEST("tokenize returns non-empty tokens", {
        auto tokens = tokenizer.tokenize("The quick brown fox");
        std::cout << "  tokens:";
        for (const auto& t : tokens) std::cout << " [" << t << "]";
        std::cout << "\n";
        assert(!tokens.empty());
    });

    RUN_TEST("encode returns non-empty ids", {
        auto ids = tokenizer.encode("The quick brown fox");
        std::cout << "  ids:";
        for (int id : ids) std::cout << " " << id;
        std::cout << "\n";
        assert(!ids.empty());
    });

    RUN_TEST("decode round-trip is non-empty", {
        auto ids = tokenizer.encode("The quick brown fox");
        std::string decoded = tokenizer.decode(ids);
        std::cout << "  decoded: [" << decoded << "]\n";
        // The decode may not be perfect with a mini-vocab,
        // but it must return a non-empty string.
        assert(!decoded.empty());
    });

    RUN_TEST("encodeToGPTTensors shape", {
        auto [input_ids, attention_mask] =
            tokenizer.encodeToGPTTensors("The quick brown fox", 1024);
        assert(input_ids.size(0) == 1);
        assert(input_ids.size(1) == attention_mask.size(1));
        assert(attention_mask.sum().item<int64_t>() == input_ids.size(1));
        std::cout << "  shapes: " << input_ids.sizes() << "\n";
    });

    RUN_TEST("max_length truncation", {
        auto [input_ids, attention_mask] =
            tokenizer.encodeToGPTTensors("The quick brown fox", 2);
        assert(input_ids.size(1) <= 2);
    });
}

static void testTokenizerFactory() {
    std::cout << "\n--- TokenizerFactory ---\n";

    RUN_TEST("loadBertTokenizer missing directory throws", {
        bool threw = false;
        try {
            at::atomspace::nn::TokenizerFactory::loadBertTokenizer(
                "/nonexistent/bert/dir/");
        } catch (const std::runtime_error&) { threw = true; }
        assert(threw && "Expected runtime_error for missing vocab.txt");
    });

    RUN_TEST("loadGPT2Tokenizer missing directory throws", {
        bool threw = false;
        try {
            at::atomspace::nn::TokenizerFactory::loadGPT2Tokenizer(
                "/nonexistent/gpt2/dir/");
        } catch (const std::runtime_error&) { threw = true; }
        assert(threw && "Expected runtime_error for missing vocab.json");
    });

    RUN_TEST("loadBertTokenizer from valid directory", {
        // vocab.txt was written to /tmp/ in earlier tests
        auto tok = at::atomspace::nn::TokenizerFactory::loadBertTokenizer("/tmp/");
        assert(tok != nullptr);
        assert(tok->encode("[CLS]", 8, false).front() == 101);
        std::cout << "  tokenizer loaded OK\n";
    });

    RUN_TEST("loadGPT2Tokenizer from valid directory", {
        auto tok = at::atomspace::nn::TokenizerFactory::loadGPT2Tokenizer("/tmp/");
        assert(tok != nullptr);
        auto ids = tok->encode("The quick brown fox");
        assert(!ids.empty());
        std::cout << "  tokenizer loaded, " << ids.size() << " ids\n";
    });
}

static void testBERTModelIntegration() {
    std::cout << "\n--- BERTModel::encodeText() ---\n";

    RUN_TEST("encodeText throws without tokenizer", {
        at::atomspace::nn::ModelConfig cfg;
        cfg.model_name = "bert-test";
        cfg.vocab_size  = 200;
        cfg.hidden_size = 32;
        cfg.num_layers  = 1;
        cfg.max_seq_length = 64;
        cfg.device = torch::kCPU;

        at::atomspace::nn::BERTModel model(cfg);
        bool threw = false;
        try { model.encodeText("hello world"); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw);
        std::cout << "  correctly threw\n";
    });

    RUN_TEST("loadTokenizer and encodeText shape", {
        at::atomspace::nn::ModelConfig cfg;
        cfg.model_name = "bert-test";
        cfg.vocab_size  = 200;
        cfg.hidden_size = 32;
        cfg.num_layers  = 1;
        cfg.max_seq_length = 64;
        cfg.device = torch::kCPU;

        at::atomspace::nn::BERTModel model(cfg);
        model.loadTokenizer("/tmp/");
        assert(model.hasTokenizer());
        auto output = model.encodeText("hello world", 16);
        // Output should be [1, seq_len, 32]
        assert(output.size(0) == 1);
        std::cout << "  output shape: " << output.sizes() << "\n";
    });
}

static void testGPTModelIntegration() {
    std::cout << "\n--- GPTModel::generateText() ---\n";

    RUN_TEST("generateText throws without tokenizer", {
        at::atomspace::nn::ModelConfig cfg;
        cfg.model_name = "gpt-test";
        cfg.vocab_size  = 100;
        cfg.hidden_size = 32;
        cfg.num_layers  = 1;
        cfg.max_seq_length = 64;
        cfg.device = torch::kCPU;

        at::atomspace::nn::GPTModel model(cfg);
        bool threw = false;
        try { model.generateText("The quick"); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw);
        std::cout << "  correctly threw\n";
    });

    RUN_TEST("loadTokenizer and generateText returns string", {
        at::atomspace::nn::ModelConfig cfg;
        cfg.model_name = "gpt-test";
        cfg.vocab_size  = 100;
        cfg.hidden_size = 32;
        cfg.num_layers  = 1;
        cfg.max_seq_length = 64;
        cfg.device = torch::kCPU;

        at::atomspace::nn::GPTModel model(cfg);
        model.loadTokenizer("/tmp/");
        assert(model.hasTokenizer());
        // Generate just 3 new tokens so the test is fast
        std::string result = model.generateText("The quick brown fox", 3);
        std::cout << "  generated: [" << result << "]\n";
        // Result is a string (may be empty if all generated ids are EOS — that is OK)
    });
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "=========================================\n";
    std::cout << " ATenSpace Phase 8 — Tokenization Tests \n";
    std::cout << "=========================================\n";

    testVocabulary();
    testWordPieceTokenizer();
    testBPETokenizer();
    testTokenizerFactory();
    testBERTModelIntegration();
    testGPTModelIntegration();

    std::cout << "\n=========================================\n";
    std::cout << " Results: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) {
        std::cout << "  (" << tests_failed << " FAILED)";
    }
    std::cout << "\n=========================================\n";

    return (tests_failed == 0) ? 0 : 1;
}
