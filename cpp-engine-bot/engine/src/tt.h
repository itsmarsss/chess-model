#pragma once

#include "types.h"
#include <cstdint>
#include <cstring>
#include <memory>

enum TTFlag : uint8_t {
    TT_NONE = 0,
    TT_EXACT = 1,
    TT_ALPHA = 2,  // Upper bound (failed low)
    TT_BETA = 3    // Lower bound (failed high)
};

struct TTEntry {
    uint16_t key16;
    int16_t score;
    int16_t static_eval;
    Move best_move;
    uint8_t depth;
    uint8_t flags; // lower 2 bits = flag, upper 6 = generation

    TTFlag flag() const { return TTFlag(flags & 3); }
    uint8_t generation() const { return flags >> 2; }
    void set(uint16_t k, int16_t s, int16_t eval, Move m, uint8_t d, TTFlag f, uint8_t gen) {
        key16 = k;
        score = s;
        static_eval = eval;
        best_move = m;
        depth = d;
        flags = (gen << 2) | f;
    }
};

// 4 entries per bucket = 64 bytes = 1 cache line
struct TTBucket {
    TTEntry entries[4];
};

class TranspositionTable {
public:
    TranspositionTable() : table(nullptr), num_buckets(0), generation_(0) {}
    ~TranspositionTable() { delete[] table; }

    void resize(size_t mb) {
        delete[] table;
        num_buckets = (mb * 1024 * 1024) / sizeof(TTBucket);
        table = new TTBucket[num_buckets];
        clear();
    }

    void clear() {
        if (table) memset(table, 0, num_buckets * sizeof(TTBucket));
    }

    void new_search() { generation_ = (generation_ + 1) & 63; }

    TTEntry* probe(uint64_t key, bool& found) {
        TTBucket& bucket = table[key % num_buckets];
        uint16_t key16 = uint16_t(key >> 48);

        for (int i = 0; i < 4; i++) {
            if (bucket.entries[i].key16 == key16 && bucket.entries[i].flag() != TT_NONE) {
                found = true;
                return &bucket.entries[i];
            }
        }
        found = false;

        // Return the entry to replace (lowest depth, oldest generation)
        TTEntry* replace = &bucket.entries[0];
        for (int i = 1; i < 4; i++) {
            if (bucket.entries[i].flag() == TT_NONE) return &bucket.entries[i];
            int replace_score = replace->depth - 4 * (replace->generation() != generation_);
            int entry_score = bucket.entries[i].depth - 4 * (bucket.entries[i].generation() != generation_);
            if (entry_score < replace_score)
                replace = &bucket.entries[i];
        }
        return replace;
    }

    void store(uint64_t key, int score, int static_eval, Move m, int depth, TTFlag flag) {
        uint16_t key16 = uint16_t(key >> 48);
        bool found;
        TTEntry* tte = probe(key, found);

        // Don't overwrite deeper entries with the same key unless we have exact bound
        if (found && tte->depth > depth && flag != TT_EXACT) return;

        tte->set(key16, int16_t(score), int16_t(static_eval), m, uint8_t(depth), flag, generation_);
    }

    void prefetch(uint64_t key) const {
        __builtin_prefetch(&table[key % num_buckets]);
    }

    int hashfull() const {
        int cnt = 0;
        for (size_t i = 0; i < std::min(num_buckets, size_t(1000)); i++) {
            for (int j = 0; j < 4; j++) {
                if (table[i].entries[j].flag() != TT_NONE &&
                    table[i].entries[j].generation() == generation_)
                    cnt++;
            }
        }
        return cnt / 4;
    }

private:
    TTBucket* table;
    size_t num_buckets;
    uint8_t generation_;
};

extern TranspositionTable TT;
