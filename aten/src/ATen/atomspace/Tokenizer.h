/**
 * Tokenizer - C++ Tokenization Support for ATenSpace
 *
 * Provides WordPiece (BERT) and BPE (GPT-2) tokenization in a
 * single-header implementation.  LibTorch is optional: the core
 * tokenization logic only requires the C++ STL; tensor helpers are
 * compiled in when HAVE_TORCH is defined (set automatically when
 * including via ATenSpace headers that pull in <torch/torch.h>).
 *
 * Features:
 * - Vocabulary loading from vocab.txt (BERT) or vocab.json (GPT-2)
 * - WordPieceTokenizer: greedy longest-match subword, "##" continuations
 * - BPETokenizer: byte-level BPE with merge rules from merges.txt
 * - Tensor helpers: encodeToBertTensors / encodeToGPTTensors / batchEncode
 * - TokenizerFactory: static convenience loaders
 *
 * Usage (WordPiece):
 *   auto tok = TokenizerFactory::loadBertTokenizer("path/to/bert/");
 *   auto tokens = tok->tokenize("Hello, world!");
 *   auto [ids, mask, type_ids] = tok->encodeToBertTensors("Hello, world!");
 *
 * Usage (BPE):
 *   auto tok = TokenizerFactory::loadGPT2Tokenizer("path/to/gpt2/");
 *   auto ids = tok->encode("The quick brown fox");
 *   std::string text = tok->decode(ids);
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// Pull in torch headers only when the rest of ATenSpace already has them.
#if defined(HAVE_TORCH) || defined(_TORCH_SCRIPT_H) || defined(TORCH_LIBRARY)
#  define TOKENIZER_HAVE_TORCH 1
#  include <torch/torch.h>
#endif

namespace at {
namespace atomspace {
namespace nn {

// ============================================================================
// Vocabulary
// ============================================================================

/**
 * Vocabulary — bidirectional token ↔ integer-id mapping.
 *
 * Supports two file formats:
 *  - vocab.txt  (BERT):  one token per line, index = line number
 *  - vocab.json (GPT-2): minimal JSON object {"token": id, ...}
 */
class Vocabulary {
public:
    Vocabulary() = default;

    /**
     * Load a BERT-style vocab.txt file.
     * Each line is a single token; line N receives id N.
     * @param path  Path to vocab.txt
     */
    void loadFromTxt(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Vocabulary: cannot open file: " + path);
        }
        token_to_id_.clear();
        id_to_token_.clear();

        std::string token;
        int id = 0;
        while (std::getline(file, token)) {
            // Strip trailing carriage-return (Windows line endings)
            if (!token.empty() && token.back() == '\r') {
                token.pop_back();
            }
            token_to_id_[token] = id;
            id_to_token_[id] = token;
            ++id;
        }
    }

    /**
     * Load a GPT-2-style vocab.json file.
     * Minimal JSON parser: expects a flat object {"token": id, ...}.
     * @param path  Path to vocab.json
     */
    void loadFromJson(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Vocabulary: cannot open file: " + path);
        }
        token_to_id_.clear();
        id_to_token_.clear();

        std::stringstream buf;
        buf << file.rdbuf();
        parseJson(buf.str());
    }

    /**
     * Look up a token's integer id.
     * Returns unk_id if the token is not in the vocabulary.
     */
    int tokenToId(const std::string& token, int unk_id = 100) const {
        auto it = token_to_id_.find(token);
        return (it != token_to_id_.end()) ? it->second : unk_id;
    }

    /**
     * Look up the string token for an integer id.
     * Returns "[UNK]" if the id is not in the vocabulary.
     */
    std::string idToToken(int id) const {
        auto it = id_to_token_.find(id);
        return (it != id_to_token_.end()) ? it->second : "[UNK]";
    }

    /** Return the number of entries in the vocabulary. */
    std::size_t size() const { return token_to_id_.size(); }

    /** Check if a token exists in the vocabulary. */
    bool contains(const std::string& token) const {
        return token_to_id_.find(token) != token_to_id_.end();
    }

    /** Direct access to the token→id map (e.g. for iteration). */
    const std::unordered_map<std::string, int>& getTokenToId() const {
        return token_to_id_;
    }

private:
    std::unordered_map<std::string, int>  token_to_id_;
    std::unordered_map<int, std::string>  id_to_token_;

    // ------------------------------------------------------------------
    // Minimal JSON parser
    // Handles flat objects: { "key": integer, ... }
    // Keys may contain escaped quotes (\\").
    // ------------------------------------------------------------------
    void parseJson(const std::string& json) {
        std::size_t pos = 0;
        auto skip_ws = [&]() {
            while (pos < json.size() &&
                   (json[pos] == ' ' || json[pos] == '\t' ||
                    json[pos] == '\n' || json[pos] == '\r')) {
                ++pos;
            }
        };

        auto read_string = [&]() -> std::string {
            // Expect opening quote
            if (pos >= json.size() || json[pos] != '"') {
                throw std::runtime_error("Vocabulary JSON: expected '\"'");
            }
            ++pos;
            std::string result;
            while (pos < json.size() && json[pos] != '"') {
                if (json[pos] == '\\' && pos + 1 < json.size()) {
                    ++pos;
                    if      (json[pos] == '"')  { result += '"';  }
                    else if (json[pos] == '\\') { result += '\\'; }
                    else if (json[pos] == '/')  { result += '/';  }
                    else if (json[pos] == 'n')  { result += '\n'; }
                    else if (json[pos] == 't')  { result += '\t'; }
                    else if (json[pos] == 'r')  { result += '\r'; }
                    else if (json[pos] == 'b')  { result += '\b'; }
                    else if (json[pos] == 'f')  { result += '\f'; }
                    else { result += json[pos]; }
                } else {
                    result += json[pos];
                }
                ++pos;
            }
            if (pos < json.size()) ++pos;  // consume closing quote
            return result;
        };

        auto read_int = [&]() -> int {
            skip_ws();
            bool negative = (pos < json.size() && json[pos] == '-');
            if (negative) ++pos;
            std::string num;
            while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
                num += json[pos++];
            }
            if (num.empty()) throw std::runtime_error("Vocabulary JSON: expected integer");
            int value = std::stoi(num);
            return negative ? -value : value;
        };

        skip_ws();
        if (pos >= json.size() || json[pos] != '{') {
            throw std::runtime_error("Vocabulary JSON: expected '{'");
        }
        ++pos;

        while (pos < json.size()) {
            skip_ws();
            if (pos >= json.size()) break;
            if (json[pos] == '}') break;
            if (json[pos] == ',') { ++pos; continue; }

            std::string token = read_string();
            skip_ws();
            if (pos >= json.size() || json[pos] != ':') {
                throw std::runtime_error("Vocabulary JSON: expected ':'");
            }
            ++pos;
            skip_ws();
            int id = read_int();

            token_to_id_[token] = id;
            id_to_token_[id]    = token;
        }
    }
};

// ============================================================================
// WordPieceTokenizer  (BERT)
// ============================================================================

/**
 * WordPieceTokenizer — BERT-style tokenization.
 *
 * Pipeline:
 *  1. Lowercase the input text.
 *  2. Strip Unicode accents (basic ASCII accent handling).
 *  3. Insert spaces around CJK characters and punctuation.
 *  4. Split on whitespace to obtain "basic" tokens.
 *  5. Apply greedy longest-match WordPiece on each token,
 *     using "##" as the continuation prefix.
 *
 * Standard BERT special-token ids (hard-coded to match the official
 * bert-base-uncased vocabulary):
 *   [PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102, [MASK]=103
 */
class WordPieceTokenizer {
public:
    // Standard BERT special-token ids
    static constexpr int ID_PAD  = 0;
    static constexpr int ID_UNK  = 100;
    static constexpr int ID_CLS  = 101;
    static constexpr int ID_SEP  = 102;
    static constexpr int ID_MASK = 103;

    /**
     * Construct from a pre-loaded Vocabulary.
     * @param vocab       Loaded BERT vocabulary
     * @param max_chars   WordPiece: maximum characters per word (default 100)
     */
    explicit WordPieceTokenizer(Vocabulary vocab, int max_chars = 100)
        : vocab_(std::move(vocab)), max_chars_(max_chars) {}

    // ------------------------------------------------------------------
    // Core interface
    // ------------------------------------------------------------------

    /**
     * Tokenize text into a list of WordPiece token strings.
     * @param text  Input text (UTF-8)
     * @return      Vector of token strings
     */
    std::vector<std::string> tokenize(const std::string& text) const {
        std::vector<std::string> output;
        for (const auto& basic_token : basicTokenize(text)) {
            auto wp = wordpieceTokenize(basic_token);
            output.insert(output.end(), wp.begin(), wp.end());
        }
        return output;
    }

    /**
     * Encode text to token ids.
     * @param text              Input text
     * @param max_length        Maximum sequence length (default 512)
     * @param add_special_tokens  Prepend [CLS] / append [SEP] (default true)
     * @return                  Vector of integer token ids
     */
    std::vector<int> encode(const std::string& text,
                             int  max_length          = 512,
                             bool add_special_tokens  = true) const {
        auto tokens = tokenize(text);
        std::vector<int> ids;
        ids.reserve(tokens.size() + 2);

        if (add_special_tokens) {
            ids.push_back(ID_CLS);
        }
        for (const auto& tok : tokens) {
            ids.push_back(vocab_.tokenToId(tok, ID_UNK));
        }
        if (add_special_tokens) {
            ids.push_back(ID_SEP);
        }

        // Truncate
        if (static_cast<int>(ids.size()) > max_length) {
            ids.resize(static_cast<std::size_t>(max_length));
        }
        return ids;
    }

#ifdef TOKENIZER_HAVE_TORCH
    /**
     * Encode text and return the three BERT input tensors.
     * @param text              Input text
     * @param max_length        Maximum sequence length (default 512)
     * @return  tuple of (input_ids, attention_mask, token_type_ids),
     *          each of shape [1, max_length], dtype torch::kLong
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    encodeToBertTensors(const std::string& text, int max_length = 512) const {
        auto ids = encode(text, max_length, /*add_special_tokens=*/true);
        return idsToTensors({ids}, max_length);
    }

    /**
     * Batch-encode a list of texts into padded BERT tensors.
     * @param texts       Input texts
     * @param max_length  Maximum sequence length (default 512)
     * @return  tuple of (input_ids, attention_mask, token_type_ids),
     *          each of shape [batch_size, max_length], dtype torch::kLong
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    batchEncode(const std::vector<std::string>& texts, int max_length = 512) const {
        std::vector<std::vector<int>> batch_ids;
        batch_ids.reserve(texts.size());
        for (const auto& t : texts) {
            batch_ids.push_back(encode(t, max_length, /*add_special_tokens=*/true));
        }
        return idsToTensors(batch_ids, max_length);
    }
#endif  // TOKENIZER_HAVE_TORCH

private:
    Vocabulary vocab_;
    int        max_chars_;

    // ------------------------------------------------------------------
    // Basic tokenization helpers
    // ------------------------------------------------------------------

    /** Convert to lowercase (ASCII only for simplicity). */
    static std::string toLower(const std::string& s) {
        std::string result = s;
        for (auto& c : result) {
            c = static_cast<char>(
                std::tolower(static_cast<unsigned char>(c)));
        }
        return result;
    }

    /** Return true if the character is ASCII punctuation or a separator. */
    static bool isPunctuation(unsigned char c) {
        return (c >= 33 && c <= 47) ||  // !"#$%&'()*+,-./
               (c >= 58 && c <= 64) ||  // :;<=>?@
               (c >= 91 && c <= 96) ||  // [\]^_`
               (c >= 123 && c <= 126);  // {|}~
    }

    /**
     * Split text on whitespace and punctuation, lowercase the result.
     * CJK characters and punctuation are each treated as individual tokens.
     */
    std::vector<std::string> basicTokenize(const std::string& text) const {
        std::string lower = toLower(text);
        std::vector<std::string> tokens;
        std::string current;

        for (std::size_t i = 0; i < lower.size(); ) {
            unsigned char c = static_cast<unsigned char>(lower[i]);

            // Handle ASCII punctuation: flush current token, emit punct
            if (isPunctuation(c)) {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
                tokens.push_back(std::string(1, static_cast<char>(c)));
                ++i;
                continue;
            }

            // Handle whitespace: flush current token
            if (std::isspace(c)) {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
                ++i;
                continue;
            }

            // Multi-byte UTF-8: pass through as-is (handles CJK, etc.)
            if (c >= 0x80) {
                // Determine byte length
                int bytes = 1;
                if      ((c & 0xE0) == 0xC0) bytes = 2;
                else if ((c & 0xF0) == 0xE0) bytes = 3;
                else if ((c & 0xF8) == 0xF0) bytes = 4;

                // For CJK unified ideographs, treat as individual tokens
                bool is_cjk = false;
                if (bytes == 3 && i + 2 < lower.size()) {
                    // 3-byte UTF-8: U+0800 .. U+FFFF
                    unsigned char b1 = c;
                    unsigned char b2 = static_cast<unsigned char>(lower[i + 1]);
                    unsigned char b3 = static_cast<unsigned char>(lower[i + 2]);
                    uint32_t cp = ((b1 & 0x0F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
                    is_cjk = (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK Unified Ideographs
                             (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK Extension A
                             (cp >= 0xF900 && cp <= 0xFAFF);     // CJK Compatibility Ideographs
                } else if (bytes == 4 && i + 3 < lower.size()) {
                    // 4-byte UTF-8: U+10000 .. U+10FFFF
                    // Covers CJK Extension B and CJK Compatibility Ideographs Supplement
                    unsigned char b1 = c;
                    unsigned char b2 = static_cast<unsigned char>(lower[i + 1]);
                    unsigned char b3 = static_cast<unsigned char>(lower[i + 2]);
                    unsigned char b4 = static_cast<unsigned char>(lower[i + 3]);
                    uint32_t cp = ((b1 & 0x07) << 18) | ((b2 & 0x3F) << 12) |
                                  ((b3 & 0x3F) << 6)  |  (b4 & 0x3F);
                    is_cjk = (cp >= 0x20000 && cp <= 0x2A6DF) ||  // CJK Extension B
                             (cp >= 0x2A700 && cp <= 0x2B73F) ||  // CJK Extension C
                             (cp >= 0x2B740 && cp <= 0x2B81F) ||  // CJK Extension D
                             (cp >= 0x2B820 && cp <= 0x2CEAF) ||  // CJK Extension E
                             (cp >= 0x2F800 && cp <= 0x2FA1F);   // CJK Compat. Supplement
                }

                if (is_cjk) {
                    if (!current.empty()) { tokens.push_back(current); current.clear(); }
                    tokens.push_back(lower.substr(i, static_cast<std::size_t>(bytes)));
                } else {
                    current += lower.substr(i, static_cast<std::size_t>(bytes));
                }
                i += static_cast<std::size_t>(bytes);
                continue;
            }

            current += static_cast<char>(c);
            ++i;
        }
        if (!current.empty()) tokens.push_back(current);
        return tokens;
    }

    /**
     * Apply greedy longest-match WordPiece to a single pre-tokenized word.
     * Returns {"[UNK]"} if the word cannot be represented.
     */
    std::vector<std::string> wordpieceTokenize(const std::string& word) const {
        if (static_cast<int>(word.size()) > max_chars_) {
            return {"[UNK]"};
        }

        std::vector<std::string> sub_tokens;
        bool is_bad = false;
        std::size_t start = 0;

        while (start < word.size()) {
            std::size_t end = word.size();
            std::string cur_substr;
            bool found = false;

            while (start < end) {
                std::string substr = word.substr(start, end - start);
                if (start > 0) substr = "##" + substr;

                if (vocab_.contains(substr)) {
                    cur_substr = substr;
                    found = true;
                    break;
                }
                // Move end back by one UTF-8 character
                --end;
                // Skip UTF-8 continuation bytes
                while (end > start &&
                       (static_cast<unsigned char>(word[end]) & 0xC0) == 0x80) {
                    --end;
                }
            }

            if (!found) {
                is_bad = true;
                break;
            }
            sub_tokens.push_back(cur_substr);
            start = end;
        }

        if (is_bad) return {"[UNK]"};
        return sub_tokens;
    }

#ifdef TOKENIZER_HAVE_TORCH
    /**
     * Convert a batch of id vectors (variable length) to padded tensors.
     * Padding uses ID_PAD (0); attention mask is 1 for real tokens, 0 for pad.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    idsToTensors(const std::vector<std::vector<int>>& batch_ids, int max_length) const {
        int64_t batch_size = static_cast<int64_t>(batch_ids.size());
        int64_t seq_len    = static_cast<int64_t>(max_length);

        auto input_ids     = torch::zeros({batch_size, seq_len}, torch::kLong);
        auto attention_mask = torch::zeros({batch_size, seq_len}, torch::kLong);
        auto token_type_ids = torch::zeros({batch_size, seq_len}, torch::kLong);

        for (int64_t b = 0; b < batch_size; ++b) {
            const auto& ids = batch_ids[static_cast<std::size_t>(b)];
            int64_t len = std::min(static_cast<int64_t>(ids.size()), seq_len);
            for (int64_t i = 0; i < len; ++i) {
                input_ids[b][i]      = ids[static_cast<std::size_t>(i)];
                attention_mask[b][i] = 1;
            }
        }
        return {input_ids, attention_mask, token_type_ids};
    }
#endif  // TOKENIZER_HAVE_TORCH
};

// ============================================================================
// BPETokenizer  (GPT-2)
// ============================================================================

/**
 * BPETokenizer — GPT-2-style byte-level BPE tokenization.
 *
 * Loads:
 *  - vocab.json  (token → id mapping)
 *  - merges.txt  (merge rules, one per line: "a b", first line may be a
 *                 "#version" comment and is skipped automatically)
 *
 * Special tokens:
 *   EOS = BOS = 50256  (GPT-2 uses a single <|endoftext|> token)
 *   PAD = -1           (GPT-2 has no dedicated padding token)
 */
class BPETokenizer {
public:
    static constexpr int ID_EOS = 50256;
    static constexpr int ID_BOS = 50256;
    static constexpr int ID_PAD = -1;

    /**
     * Construct from pre-loaded vocabulary and merge rules.
     * @param vocab   Loaded GPT-2 vocabulary (vocab.json)
     * @param merges  Ordered list of merge rules as "A B" strings
     */
    BPETokenizer(Vocabulary vocab, std::vector<std::string> merges)
        : vocab_(std::move(vocab)) {
        buildMergeMap(std::move(merges));
        buildByteEncoder();
    }

    // ------------------------------------------------------------------
    // Core interface
    // ------------------------------------------------------------------

    /**
     * Tokenize text into a list of BPE token strings.
     * @param text  Input text (UTF-8)
     * @return      Vector of token strings (byte-level BPE)
     */
    std::vector<std::string> tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        // Split on whitespace-prefixed word boundaries (GPT-2 prepends Ġ)
        for (const auto& word : splitWords(text)) {
            auto wp = bpeEncode(word);
            tokens.insert(tokens.end(), wp.begin(), wp.end());
        }
        return tokens;
    }

    /**
     * Encode text to a vector of token ids.
     * @param text  Input text
     * @return      Vector of integer token ids
     */
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        for (const auto& tok : tokenize(text)) {
            ids.push_back(vocab_.tokenToId(tok, ID_UNK_));
        }
        return ids;
    }

    /**
     * Decode a sequence of token ids back to a string.
     * @param ids  Token ids
     * @return     Decoded text (UTF-8)
     */
    std::string decode(const std::vector<int>& ids) const {
        // Collect byte-encoded tokens, then map back through byte_decoder_
        std::string byte_stream;
        for (int id : ids) {
            if (id == ID_EOS) continue;
            std::string tok = vocab_.idToToken(id);
            // Replace Ġ → space, etc. using byte decoder
            byte_stream += byteDecode(tok);
        }
        return byte_stream;
    }

#ifdef TOKENIZER_HAVE_TORCH
    /**
     * Encode text and return GPT-2 input tensors.
     * @param text        Input text
     * @param max_length  Maximum sequence length (default 1024)
     * @return  pair of (input_ids, attention_mask),
     *          each of shape [1, seq_len], dtype torch::kLong
     *          (seq_len = min(actual_len, max_length))
     */
    std::pair<torch::Tensor, torch::Tensor>
    encodeToGPTTensors(const std::string& text, int max_length = 1024) const {
        auto ids = encode(text);
        if (static_cast<int>(ids.size()) > max_length) {
            ids.resize(static_cast<std::size_t>(max_length));
        }
        int64_t seq_len = static_cast<int64_t>(ids.size());

        auto input_ids      = torch::zeros({1, seq_len}, torch::kLong);
        auto attention_mask = torch::ones({1, seq_len},  torch::kLong);

        for (int64_t i = 0; i < seq_len; ++i) {
            input_ids[0][i] = ids[static_cast<std::size_t>(i)];
        }
        return {input_ids, attention_mask};
    }
#endif  // TOKENIZER_HAVE_TORCH

private:
    // GPT-2 uses a different unknown token; its vocab rarely has OOV because
    // byte-level encoding is complete, but keep a fallback.
    static constexpr int ID_UNK_ = 0;

    Vocabulary vocab_;

    // merge_priority_[{"A","B"}] = rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_priority_;

    // Byte encoder / decoder (maps bytes 0-255 ↔ unicode characters)
    std::unordered_map<int, std::string>  byte_encoder_;   // byte value → char string
    std::unordered_map<std::string, int>  byte_decoder_;   // char string → byte value

    // ------------------------------------------------------------------
    // Build helpers
    // ------------------------------------------------------------------

    /** Build a priority map from the ordered merge list. */
    void buildMergeMap(std::vector<std::string> merges) {
        merge_priority_.reserve(merges.size());
        for (int rank = 0; rank < static_cast<int>(merges.size()); ++rank) {
            merge_priority_[merges[static_cast<std::size_t>(rank)]] = rank;
        }
    }

    /**
     * Build the byte-level encoder used by GPT-2.
     * Maps the 256 possible byte values to human-readable unicode strings
     * so that the vocabulary covers all possible byte sequences.
     */
    void buildByteEncoder() {
        // Characters in ranges [33-126], [161-172], [174-255] map to themselves.
        // All other bytes (0-32, 127-160, 173) map to code points starting at 256.
        std::vector<int> bs;
        for (int i = 33; i <= 126; ++i)  bs.push_back(i);  // ! – ~
        for (int i = 161; i <= 172; ++i) bs.push_back(i);  // ¡ – ¬
        for (int i = 174; i <= 255; ++i) bs.push_back(i);  // ® – ÿ

        std::vector<int> cs = bs;  // copy for code points
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        }

        for (std::size_t i = 0; i < bs.size(); ++i) {
            std::string s = utf8Encode(static_cast<uint32_t>(cs[i]));
            byte_encoder_[bs[i]] = s;
            byte_decoder_[s]     = bs[i];
        }
    }

    /** Encode a Unicode code point as a UTF-8 string. */
    static std::string utf8Encode(uint32_t cp) {
        std::string s;
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            s += static_cast<char>(0xF0 | (cp >> 18));
            s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return s;
    }

    // ------------------------------------------------------------------
    // BPE core
    // ------------------------------------------------------------------

    /**
     * Split text into word units.
     * Non-first words are prefixed with a literal space character so that
     * wordToBytes() correctly maps it to the Ġ (U+0120) symbol via
     * byte_encoder_[32], matching GPT-2's convention.
     */
    std::vector<std::string> splitWords(const std::string& text) const {
        std::vector<std::string> words;
        std::string current;
        bool after_space = true;   // treat start-of-text like after whitespace

        for (std::size_t i = 0; i < text.size(); ) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            int bytes = 1;
            if      ((c & 0xE0) == 0xC0) bytes = 2;
            else if ((c & 0xF0) == 0xE0) bytes = 3;
            else if ((c & 0xF8) == 0xF0) bytes = 4;

            std::string ch = text.substr(i, static_cast<std::size_t>(bytes));
            i += static_cast<std::size_t>(bytes);

            if (ch == " " || ch == "\n" || ch == "\t" || ch == "\r") {
                if (!current.empty()) {
                    words.push_back(current);
                    current.clear();
                }
                after_space = true;
            } else {
                // Prefix non-first words with a literal space; wordToBytes()
                // will encode it as byte 32 → "Ġ".
                if (after_space && !words.empty()) {
                    current = " ";   // leading space byte for this word
                }
                current += ch;
                after_space = false;
            }
        }
        if (!current.empty()) words.push_back(current);
        return words;
    }

    /**
     * Convert a word string to its byte-level BPE symbol sequence.
     * Every raw byte is mapped to its corresponding unicode symbol string
     * via byte_encoder_.  In particular, byte 32 (space) maps to "Ġ".
     */
    std::vector<std::string> wordToBytes(const std::string& word) const {
        std::vector<std::string> chars;
        chars.reserve(word.size());
        for (std::size_t i = 0; i < word.size(); ++i) {
            unsigned char raw = static_cast<unsigned char>(word[i]);
            auto it = byte_encoder_.find(raw);
            if (it != byte_encoder_.end()) {
                chars.push_back(it->second);
            } else {
                // Fallback: keep the raw byte as a one-character string
                chars.push_back(std::string(1, static_cast<char>(raw)));
            }
        }
        return chars;
    }

    /**
     * Apply BPE merge rules to a sequence of symbol strings.
     * Returns the final list of BPE tokens for one word.
     */
    std::vector<std::string> bpeEncode(const std::string& word) const {
        auto chars = wordToBytes(word);
        if (chars.size() <= 1) return chars;

        // Iteratively apply the highest-priority merge
        while (chars.size() > 1) {
            int    best_rank  = std::numeric_limits<int>::max();
            std::size_t best_idx   = 0;
            bool   found_merge = false;

            for (std::size_t i = 0; i + 1 < chars.size(); ++i) {
                std::string pair_key = chars[i] + " " + chars[i + 1];
                auto it = merge_priority_.find(pair_key);
                if (it != merge_priority_.end() && it->second < best_rank) {
                    best_rank  = it->second;
                    best_idx   = i;
                    found_merge = true;
                }
            }

            if (!found_merge) break;

            // Merge the pair at best_idx: emit merged token, skip chars[best_idx+1].
            // The explicit ++i skips the right half of the pair; the for-loop ++i then
            // advances to best_idx+2, which is the correct next element.
            std::string merged = chars[best_idx] + chars[best_idx + 1];
            std::vector<std::string> next;
            next.reserve(chars.size() - 1);
            for (std::size_t i = 0; i < chars.size(); ++i) {
                if (i == best_idx) {
                    next.push_back(merged);
                    ++i;  // skip chars[best_idx+1] (right half of merged pair)
                } else {
                    next.push_back(chars[i]);
                }
            }
            chars = std::move(next);
        }
        return chars;
    }

    /**
     * Decode a single BPE token string back to raw bytes, then to UTF-8.
     * Handles the Ġ → ' ' mapping and the full byte_decoder_ table.
     */
    std::string byteDecode(const std::string& token) const {
        // Iterate over unicode code-points in `token`
        std::string result;
        std::size_t i = 0;
        while (i < token.size()) {
            unsigned char c = static_cast<unsigned char>(token[i]);
            int bytes = 1;
            if      ((c & 0xE0) == 0xC0) bytes = 2;
            else if ((c & 0xF0) == 0xE0) bytes = 3;
            else if ((c & 0xF8) == 0xF0) bytes = 4;

            std::string cp_str = token.substr(i, static_cast<std::size_t>(bytes));
            i += static_cast<std::size_t>(bytes);

            // Ġ (U+0120) → space
            if (cp_str == "\xC4\xA0") {
                result += ' ';
                continue;
            }

            auto it = byte_decoder_.find(cp_str);
            if (it != byte_decoder_.end()) {
                result += static_cast<char>(it->second);
            } else {
                // Pass through unchanged (already valid UTF-8)
                result += cp_str;
            }
        }
        return result;
    }
};

// ============================================================================
// TokenizerFactory
// ============================================================================

/**
 * TokenizerFactory — static helpers for loading tokenizers from a directory.
 *
 * Expected directory layout:
 *   BERT:  <dir>/vocab.txt
 *   GPT-2: <dir>/vocab.json  and  <dir>/merges.txt
 */
class TokenizerFactory {
public:
    /**
     * Load a WordPieceTokenizer (BERT) from a directory.
     * @param dir  Directory containing vocab.txt
     * @return     Shared pointer to the initialized tokenizer
     * @throws std::runtime_error if vocab.txt is not found
     */
    static std::shared_ptr<WordPieceTokenizer>
    loadBertTokenizer(const std::string& dir) {
        std::string vocab_path = ensureTrailingSep(dir) + "vocab.txt";
        Vocabulary vocab;
        vocab.loadFromTxt(vocab_path);
        return std::make_shared<WordPieceTokenizer>(std::move(vocab));
    }

    /**
     * Load a BPETokenizer (GPT-2) from a directory.
     * @param dir  Directory containing vocab.json and merges.txt
     * @return     Shared pointer to the initialized tokenizer
     * @throws std::runtime_error if either file is not found
     */
    static std::shared_ptr<BPETokenizer>
    loadGPT2Tokenizer(const std::string& dir) {
        std::string base = ensureTrailingSep(dir);

        Vocabulary vocab;
        vocab.loadFromJson(base + "vocab.json");

        std::vector<std::string> merges = loadMerges(base + "merges.txt");
        return std::make_shared<BPETokenizer>(std::move(vocab), std::move(merges));
    }

private:
    /** Ensure the path ends with a directory separator. */
    static std::string ensureTrailingSep(const std::string& dir) {
        if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') {
            return dir + "/";
        }
        return dir;
    }

    /**
     * Parse merges.txt: one merge rule per line ("A B"),
     * skipping lines that start with '#' (version comments).
     */
    static std::vector<std::string> loadMerges(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("TokenizerFactory: cannot open: " + path);
        }
        std::vector<std::string> merges;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            merges.push_back(line);
        }
        return merges;
    }
};

} // namespace nn
} // namespace atomspace
} // namespace at
