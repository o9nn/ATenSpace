#pragma once

#include "Atom.h"
#include "AtomSpace.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace at {
namespace atomspace {

/**
 * BinarySerializer - Production-grade binary persistence for AtomSpace
 *
 * Provides compact, fast, and complete serialization of the hypergraph:
 *  - All Atom types (nodes and links)
 *  - Tensor embeddings (float32, flattened)
 *  - Truth values
 *  - Attention values
 *  - Full link → atom reference resolution
 *
 * Binary format layout
 * --------------------
 * [4 bytes]  Magic number  "ATSP"
 * [4 bytes]  Format version (uint32_t)
 * [8 bytes]  Atom count    (uint64_t)
 * ---- per atom ----
 * [4 bytes]  Atom type     (uint32_t)
 * [1 byte]   Is node       (uint8_t)  0=link 1=node
 * --- if node ---
 * [4 bytes]  Name length   (uint32_t)
 * [N bytes]  Name UTF-8
 * [1 byte]   Has embedding (uint8_t)
 * --- if has_embedding ---
 * [4 bytes]  Embedding dim (uint32_t)
 * [dim*4]    float32 values
 * --- end if ---
 * --- if link ---
 * [4 bytes]  Arity         (uint32_t)
 * [arity*8]  Child IDs     (uint64_t each, index into atom list)
 * --- end per atom ---
 * [1 byte]   Has truth value (uint8_t)
 * [8 bytes]  TruthValue [strength(f32), confidence(f32)]
 * [4 bytes]  Attention value (float32)
 * ---- end per atom ----
 */
class BinarySerializer {
public:
    static constexpr uint32_t MAGIC   = 0x41545350u;  // "ATSP"
    static constexpr uint32_t VERSION = 1u;

    // ------------------------------------------------------------------ //
    //  Save
    // ------------------------------------------------------------------ //

    /**
     * Serialize the given AtomSpace to a binary file.
     *
     * @param space     AtomSpace to save
     * @param filename  Output file path
     * @return          true on success
     */
    static bool save(const AtomSpace& space, const std::string& filename) {
        std::ofstream out(filename, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) return false;

        auto bytes = serialize(space);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        return out.good();
    }

    /**
     * Deserialize an AtomSpace from a binary file.
     *
     * @param space     AtomSpace to populate
     * @param filename  Input file path
     * @return          true on success
     */
    static bool load(AtomSpace& space, const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;

        in.seekg(0, std::ios::end);
        auto size = static_cast<size_t>(in.tellg());
        in.seekg(0, std::ios::beg);

        std::vector<uint8_t> bytes(size);
        in.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(size));
        if (!in) return false;

        try {
            deserialize(space, bytes);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    // ------------------------------------------------------------------ //
    //  In-memory serialize / deserialize
    // ------------------------------------------------------------------ //

    /**
     * Serialize to an in-memory byte buffer.
     */
    static std::vector<uint8_t> serialize(const AtomSpace& space) {
        std::vector<uint8_t> buf;
        buf.reserve(4096);

        // Collect atoms: nodes first, then links in topological order
        auto allAtoms = space.getAtoms();
        std::vector<Atom::Handle> ordered;
        ordered.reserve(allAtoms.size());

        // Add all nodes first
        for (const auto& a : allAtoms) {
            if (a->isNode()) ordered.push_back(a);
        }

        // Add links in topological order (depth-first post-order)
        // so that any link referenced as a child appears before its parent
        std::unordered_set<Atom::Handle> visited;
        // Pre-seed visited with nodes
        for (const auto& a : ordered) visited.insert(a);

        // Topological sort of links via DFS
        std::function<void(const Atom::Handle&)> topoVisit =
            [&](const Atom::Handle& atom) {
                if (visited.count(atom)) return;
                if (!atom->isLink()) { visited.insert(atom); return; }
                // Visit children first
                const auto* lptr = static_cast<const Link*>(atom.get());
                for (const auto& child : lptr->getOutgoingSet()) {
                    topoVisit(child);
                }
                if (!visited.count(atom)) {
                    visited.insert(atom);
                    ordered.push_back(atom);
                }
            };

        for (const auto& a : allAtoms) {
            if (a->isLink()) topoVisit(a);
        }

        // Build index: atom pointer → position index
        std::unordered_map<Atom::Handle, uint64_t> index;
        for (uint64_t i = 0; i < ordered.size(); ++i) {
            index[ordered[i]] = i;
        }

        // Header
        writeU32(buf, MAGIC);
        writeU32(buf, VERSION);
        writeU64(buf, static_cast<uint64_t>(ordered.size()));

        // Atoms
        for (const auto& atom : ordered) {
            writeAtom(buf, atom, index);
        }

        return buf;
    }

    /**
     * Deserialize from an in-memory byte buffer.
     */
    static void deserialize(AtomSpace& space,
                            const std::vector<uint8_t>& buf) {
        size_t pos = 0;

        // Header
        uint32_t magic = readU32(buf, pos);
        if (magic != MAGIC) {
            throw std::runtime_error("BinarySerializer: bad magic number");
        }
        uint32_t version = readU32(buf, pos);
        if (version != VERSION) {
            throw std::runtime_error(
                "BinarySerializer: unsupported version " +
                std::to_string(version));
        }
        uint64_t atomCount = readU64(buf, pos);

        // Two-pass deserialization:
        // Pass 1: create all nodes (links reference nodes by index)
        // Pass 2: create all links using the index

        // Store raw atom descriptors
        struct AtomDesc {
            uint32_t type;
            bool isNode;
            // Node fields
            std::string name;
            bool hasEmbedding;
            std::vector<float> embData;
            uint32_t embDim;
            // Link fields
            std::vector<uint64_t> childIds;
            // Shared
            bool hasTv;
            float tvStrength;
            float tvConfidence;
            float attention;
        };

        std::vector<AtomDesc> descs(atomCount);
        for (uint64_t i = 0; i < atomCount; ++i) {
            AtomDesc& d = descs[i];
            d.type    = readU32(buf, pos);
            d.isNode  = (readU8(buf, pos) != 0);

            if (d.isNode) {
                uint32_t nameLen = readU32(buf, pos);
                d.name = readString(buf, pos, nameLen);
                d.hasEmbedding = (readU8(buf, pos) != 0);
                if (d.hasEmbedding) {
                    d.embDim = readU32(buf, pos);
                    d.embData.resize(d.embDim);
                    for (uint32_t j = 0; j < d.embDim; ++j) {
                        d.embData[j] = readF32(buf, pos);
                    }
                }
            } else {
                uint32_t arity = readU32(buf, pos);
                d.childIds.resize(arity);
                for (uint32_t j = 0; j < arity; ++j) {
                    d.childIds[j] = readU64(buf, pos);
                }
            }

            d.hasTv = (readU8(buf, pos) != 0);
            if (d.hasTv) {
                d.tvStrength   = readF32(buf, pos);
                d.tvConfidence = readF32(buf, pos);
            }
            d.attention = readF32(buf, pos);
        }

        // Build handles
        std::vector<Atom::Handle> handles(atomCount, nullptr);

        // Pass 1: nodes
        for (uint64_t i = 0; i < atomCount; ++i) {
            const AtomDesc& d = descs[i];
            if (!d.isNode) continue;

            auto type = static_cast<Atom::Type>(d.type);
            Atom::Handle h;
            if (d.hasEmbedding && !d.embData.empty()) {
                Tensor emb = torch::from_blob(
                    const_cast<float*>(d.embData.data()),
                    {static_cast<int64_t>(d.embDim)},
                    torch::kFloat).clone();
                h = space.addNode(type, d.name, emb);
            } else {
                h = space.addNode(type, d.name);
            }

            if (d.hasTv) {
                h->setTruthValue(TruthValue::create(d.tvStrength, d.tvConfidence));
            }
            h->setAttention(d.attention);
            handles[i] = h;
        }

        // Pass 2: links (topological order guaranteed by save() writing nodes first)
        for (uint64_t i = 0; i < atomCount; ++i) {
            const AtomDesc& d = descs[i];
            if (d.isNode) continue;

            auto type = static_cast<Atom::Type>(d.type);
            std::vector<Atom::Handle> outgoing;
            outgoing.reserve(d.childIds.size());

            for (uint64_t cid : d.childIds) {
                if (cid >= atomCount || !handles[cid]) {
                    throw std::runtime_error(
                        "BinarySerializer: invalid child atom index");
                }
                outgoing.push_back(handles[cid]);
            }

            Atom::Handle h = space.addLink(type, outgoing);
            if (d.hasTv) {
                h->setTruthValue(TruthValue::create(d.tvStrength, d.tvConfidence));
            }
            h->setAttention(d.attention);
            handles[i] = h;
        }
    }

private:
    // ------------------------------------------------------------------ //
    //  Per-atom serialization helper
    // ------------------------------------------------------------------ //

    static void writeAtom(
            std::vector<uint8_t>& buf,
            const Atom::Handle& atom,
            const std::unordered_map<Atom::Handle, uint64_t>& index) {

        writeU32(buf, static_cast<uint32_t>(atom->getType()));
        writeU8(buf, atom->isNode() ? 1u : 0u);

        if (atom->isNode()) {
            const auto* node = static_cast<const Node*>(atom.get());
            const std::string& name = node->getName();
            writeU32(buf, static_cast<uint32_t>(name.size()));
            writeBytes(buf,
                       reinterpret_cast<const uint8_t*>(name.data()),
                       name.size());

            bool hasEmb = node->hasEmbedding();
            writeU8(buf, hasEmb ? 1u : 0u);
            if (hasEmb) {
                auto emb = node->getEmbedding().reshape({-1}).to(torch::kFloat);
                uint32_t dim = static_cast<uint32_t>(emb.size(0));
                writeU32(buf, dim);
                auto acc = emb.accessor<float, 1>();
                for (uint32_t j = 0; j < dim; ++j) {
                    writeF32(buf, acc[j]);
                }
            }
        } else {
            const auto* link = static_cast<const Link*>(atom.get());
            const auto& out  = link->getOutgoingSet();
            writeU32(buf, static_cast<uint32_t>(out.size()));
            for (const auto& child : out) {
                auto it = index.find(child);
                if (it == index.end()) {
                    throw std::runtime_error(
                        "BinarySerializer: atom not found in index");
                }
                writeU64(buf, it->second);
            }
        }

        // Truth value
        auto tv = atom->getTruthValue();
        bool hasTv = tv.defined() && tv.numel() >= 2;
        writeU8(buf, hasTv ? 1u : 0u);
        if (hasTv) {
            writeF32(buf, TruthValue::getStrength(tv));
            writeF32(buf, TruthValue::getConfidence(tv));
        }

        // Attention
        writeF32(buf, atom->getAttention());
    }

    // ------------------------------------------------------------------ //
    //  Byte-level write helpers
    // ------------------------------------------------------------------ //

    static void writeU8(std::vector<uint8_t>& buf, uint8_t v) {
        buf.push_back(v);
    }
    static void writeU32(std::vector<uint8_t>& buf, uint32_t v) {
        buf.push_back(static_cast<uint8_t>(v));
        buf.push_back(static_cast<uint8_t>(v >> 8));
        buf.push_back(static_cast<uint8_t>(v >> 16));
        buf.push_back(static_cast<uint8_t>(v >> 24));
    }
    static void writeU64(std::vector<uint8_t>& buf, uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<uint8_t>(v >> (i * 8)));
        }
    }
    static void writeF32(std::vector<uint8_t>& buf, float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        writeU32(buf, bits);
    }
    static void writeBytes(std::vector<uint8_t>& buf,
                           const uint8_t* data, size_t n) {
        buf.insert(buf.end(), data, data + n);
    }

    // ------------------------------------------------------------------ //
    //  Byte-level read helpers
    // ------------------------------------------------------------------ //

    static void checkBounds(const std::vector<uint8_t>& buf,
                            size_t pos, size_t needed) {
        if (pos + needed > buf.size()) {
            throw std::runtime_error(
                "BinarySerializer: unexpected end of data");
        }
    }
    static uint8_t readU8(const std::vector<uint8_t>& buf, size_t& pos) {
        checkBounds(buf, pos, 1);
        return buf[pos++];
    }
    static uint32_t readU32(const std::vector<uint8_t>& buf, size_t& pos) {
        checkBounds(buf, pos, 4);
        uint32_t v = static_cast<uint32_t>(buf[pos])
                   | (static_cast<uint32_t>(buf[pos+1]) << 8)
                   | (static_cast<uint32_t>(buf[pos+2]) << 16)
                   | (static_cast<uint32_t>(buf[pos+3]) << 24);
        pos += 4;
        return v;
    }
    static uint64_t readU64(const std::vector<uint8_t>& buf, size_t& pos) {
        checkBounds(buf, pos, 8);
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= (static_cast<uint64_t>(buf[pos + i]) << (i * 8));
        }
        pos += 8;
        return v;
    }
    static float readF32(const std::vector<uint8_t>& buf, size_t& pos) {
        uint32_t bits = readU32(buf, pos);
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    static std::string readString(const std::vector<uint8_t>& buf,
                                  size_t& pos, uint32_t len) {
        checkBounds(buf, pos, len);
        std::string s(reinterpret_cast<const char*>(&buf[pos]), len);
        pos += len;
        return s;
    }
};

} // namespace atomspace
} // namespace at
