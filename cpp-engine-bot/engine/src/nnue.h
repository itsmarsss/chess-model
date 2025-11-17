#pragma once

#include "types.h"
#include "position.h"
#include <string>
#include <cstdint>

namespace NNUE {

// Architecture constants matching training code
constexpr int L1 = 3072;           // Feature transformer output
constexpr int L2 = 15;             // First hidden (+ 1 skip = 16 total)
constexpr int L3 = 32;             // Second hidden
constexpr int NUM_PSQT_BUCKETS = 8;
constexpr int NUM_LS_BUCKETS = 8;

// Feature set: HalfKAv2_hm
constexpr int NUM_SQ = 64;
constexpr int NUM_PT_REAL = 11;
constexpr int NUM_PLANES_REAL = NUM_SQ * NUM_PT_REAL; // 704
constexpr int NUM_INPUTS = NUM_PLANES_REAL * NUM_SQ / 2; // 22528

// Quantization constants
constexpr int QUANTIZED_ONE = 127;
constexpr int WEIGHT_SCALE_HIDDEN = 64;
constexpr int WEIGHT_SCALE_OUT = 16;
constexpr int NNUE2SCORE = 600;

// Accumulator for incremental updates
struct alignas(64) Accumulator {
    int16_t values[2][L1];      // [perspective][L1]
    int32_t psqt[2][NUM_PSQT_BUCKETS]; // [perspective][buckets]
    bool computed[2];
};

// Network weights (loaded from .nnue file)
struct Network {
    // Feature transformer
    alignas(64) int16_t ft_bias[L1];
    alignas(64) int16_t ft_weight[NUM_INPUTS][L1];
    alignas(64) int32_t ft_psqt[NUM_INPUTS][NUM_PSQT_BUCKETS];

    // Layer stacks (8 buckets)
    // L1 input: 2 * (L1/2) = L1 (after product pooling from both perspectives = 2 * L1/2)
    // Actually: input is L1 (1536 from stm + 1536 from nstm after product pooling)
    alignas(64) int32_t l1_bias[NUM_LS_BUCKETS][L2 + 1];
    alignas(64) int8_t  l1_weight[NUM_LS_BUCKETS][(L2 + 1)][L1]; // padded to 32

    alignas(64) int32_t l2_bias[NUM_LS_BUCKETS][L3];
    alignas(64) int8_t  l2_weight[NUM_LS_BUCKETS][L3][L2 * 2]; // padded to 32

    alignas(64) int32_t out_bias[NUM_LS_BUCKETS][1];
    alignas(64) int8_t  out_weight[NUM_LS_BUCKETS][1][L3]; // padded to 32

    bool loaded;
};

extern Network net;

// Initialize NNUE: load weights from .nnue file
bool init(const std::string& path);
bool is_loaded();

// Evaluate a position using NNUE
int evaluate(const Position& pos);

// Feature index calculation (HalfKAv2_hm)
int feature_index(bool is_white_pov, Square king_sq, Square sq, Piece piece);

// Refresh accumulator from scratch
void refresh_accumulator(const Position& pos, Accumulator& acc, Color perspective);

} // namespace NNUE
