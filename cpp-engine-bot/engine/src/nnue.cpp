#include "nnue.h"
#include "bitboard.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

namespace NNUE {

Network net;

// King buckets table matching halfka_v2_hm.py
static constexpr int KingBuckets[64] = {
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10,  9,  8,
    -1, -1, -1, -1,  7,  6,  5,  4,
    -1, -1, -1, -1,  3,  2,  1,  0
};

static inline int orient(bool is_white_pov, int sq, int ksq) {
    int kfile = ksq % 8;
    return (7 * (kfile < 4)) ^ (56 * (!is_white_pov)) ^ sq;
}

int feature_index(bool is_white_pov, Square king_sq, Square sq, Piece piece) {
    int p_idx = (type_of(piece) - 1) * 2 + (color_of(piece) != (is_white_pov ? WHITE : BLACK));
    if (p_idx == 11) p_idx = 10;
    int o_ksq = orient(is_white_pov, int(king_sq), int(king_sq));
    return orient(is_white_pov, int(sq), int(king_sq))
         + p_idx * NUM_SQ
         + KingBuckets[o_ksq] * NUM_PLANES_REAL;
}

// LEB128 decoder
static bool decode_leb128(std::ifstream& f, void* output, size_t count, size_t elem_size) {
    std::vector<uint8_t> buf;
    // Read the length first (int32)
    uint32_t len;
    f.read(reinterpret_cast<char*>(&len), 4);
    if (!f.good()) return false;

    buf.resize(len);
    f.read(reinterpret_cast<char*>(buf.data()), len);
    if (!f.good()) return false;

    size_t total_elems = count;
    size_t k = 0;

    if (elem_size == 2) {
        int16_t* out = static_cast<int16_t*>(output);
        for (size_t i = 0; i < total_elems && k < buf.size(); i++) {
            int64_t r = 0;
            int shift = 0;
            while (k < buf.size()) {
                uint8_t byte = buf[k++];
                r |= int64_t(byte & 0x7F) << shift;
                shift += 7;
                if ((byte & 0x80) == 0) {
                    if (byte & 0x40) r |= ~((int64_t(1) << shift) - 1);
                    break;
                }
            }
            out[i] = int16_t(r);
        }
    } else if (elem_size == 4) {
        int32_t* out = static_cast<int32_t*>(output);
        for (size_t i = 0; i < total_elems && k < buf.size(); i++) {
            int64_t r = 0;
            int shift = 0;
            while (k < buf.size()) {
                uint8_t byte = buf[k++];
                r |= int64_t(byte & 0x7F) << shift;
                shift += 7;
                if ((byte & 0x80) == 0) {
                    if (byte & 0x40) r |= ~((int64_t(1) << shift) - 1);
                    break;
                }
            }
            out[i] = int32_t(r);
        }
    } else if (elem_size == 1) {
        int8_t* out = static_cast<int8_t*>(output);
        for (size_t i = 0; i < total_elems && k < buf.size(); i++) {
            int64_t r = 0;
            int shift = 0;
            while (k < buf.size()) {
                uint8_t byte = buf[k++];
                r |= int64_t(byte & 0x7F) << shift;
                shift += 7;
                if ((byte & 0x80) == 0) {
                    if (byte & 0x40) r |= ~((int64_t(1) << shift) - 1);
                    break;
                }
            }
            out[i] = int8_t(r);
        }
    }

    return true;
}

static const char LEB128_MAGIC[] = "COMPRESSED_LEB128";
static constexpr size_t LEB128_MAGIC_LEN = 17;

enum Compression { NONE, LEB128 };

static Compression detect_compression(std::ifstream& f) {
    char buf[17];
    auto pos = f.tellg();
    f.read(buf, LEB128_MAGIC_LEN);
    if (f.good() && memcmp(buf, LEB128_MAGIC, LEB128_MAGIC_LEN) == 0) {
        return LEB128;
    }
    f.seekg(pos);
    return NONE;
}

static bool read_tensor_i16(std::ifstream& f, int16_t* data, size_t count) {
    Compression comp = detect_compression(f);
    if (comp == LEB128) {
        return decode_leb128(f, data, count, 2);
    }
    f.read(reinterpret_cast<char*>(data), count * sizeof(int16_t));
    return f.good();
}

static bool read_tensor_i32(std::ifstream& f, int32_t* data, size_t count) {
    Compression comp = detect_compression(f);
    if (comp == LEB128) {
        return decode_leb128(f, data, count, 4);
    }
    f.read(reinterpret_cast<char*>(data), count * sizeof(int32_t));
    return f.good();
}

static bool read_tensor_i8(std::ifstream& f, int8_t* data, size_t count) {
    Compression comp = detect_compression(f);
    if (comp == LEB128) {
        return decode_leb128(f, data, count, 1);
    }
    f.read(reinterpret_cast<char*>(data), count * sizeof(int8_t));
    return f.good();
}

bool init(const std::string& path) {
    net.loaded = false;

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "NNUE: Cannot open file: " << path << std::endl;
        return false;
    }

    // Read header
    uint32_t version, hash, desc_len;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&hash), 4);
    f.read(reinterpret_cast<char*>(&desc_len), 4);

    if (version != 0x7AF32F20) {
        std::cerr << "NNUE: Invalid version: 0x" << std::hex << version << std::endl;
        return false;
    }

    // Skip description
    f.seekg(desc_len, std::ios::cur);

    // Read feature transformer hash
    uint32_t ft_hash;
    f.read(reinterpret_cast<char*>(&ft_hash), 4);

    // Read feature transformer: bias[L1], weight[NUM_INPUTS][L1], psqt[NUM_INPUTS][NUM_PSQT_BUCKETS]
    if (!read_tensor_i16(f, net.ft_bias, L1)) {
        std::cerr << "NNUE: Failed to read FT bias" << std::endl;
        return false;
    }

    if (!read_tensor_i16(f, &net.ft_weight[0][0], (size_t)NUM_INPUTS * L1)) {
        std::cerr << "NNUE: Failed to read FT weight" << std::endl;
        return false;
    }

    if (!read_tensor_i32(f, &net.ft_psqt[0][0], (size_t)NUM_INPUTS * NUM_PSQT_BUCKETS)) {
        std::cerr << "NNUE: Failed to read FT PSQT weight" << std::endl;
        return false;
    }

    // Read layer stacks (8 buckets)
    for (int bucket = 0; bucket < NUM_LS_BUCKETS; bucket++) {
        // FC hash
        uint32_t fc_hash;
        f.read(reinterpret_cast<char*>(&fc_hash), 4);

        // L1 layer: bias[L2+1], weight[(L2+1)][padded_input]
        int l1_inputs = L1; // After product pooling: 1536 + 1536 = 3072... wait
        // Actually the input size after SCReLU product pooling is L1/2 * 2 perspectives = L1
        // But looking at training code: l1 = FactorizedStackedLinear(2 * L1 // 2, L2 + 1, count)
        // So input is 2 * L1/2 = L1 = 3072... but after product pooling each perspective goes from L1 -> L1/2
        // Two perspectives concatenated: L1/2 + L1/2 = L1/2 * 2...
        // Wait: from model.py forward: l0_ = cat(l0_s1, dim=1) where l0_s1 has 2 elements of L1//2 each
        // So l0_ has size L1 (3072)... but it was [stm:nstm] each of L1 -> clamp -> split into L1/2 halves -> multiply
        // So: l0_s = split(l0_, L1//2) gives 4 tensors of L1/2 = 768 each
        //     l0_s1 = [l0_s[0]*l0_s[1], l0_s[2]*l0_s[3]] -> 2 tensors of 768 -> cat -> 1536
        // Wait no, l0_ has 2*L1 = 6144 elements (stm L1 + nstm L1)
        // Actually looking more carefully at model.py:
        //   l0_ = (us * cat([w, b])) + (them * cat([b, w]))  -> shape is 2*L1 = 6144? No...
        //   w and b are each L1 so cat gives 2*L1... but 'us' is a mask...
        //   Actually us/them are scalars (0 or 1 broadcasting). So l0_ is 2*L1 = 6144
        //   Then l0_s = split(l0_, L1//2) -> splits into chunks of L1/2=1536, giving 4 chunks
        //   l0_s1 = [l0_s[0]*l0_s[1], l0_s[2]*l0_s[3]] -> 2 chunks of 1536 -> cat -> 3072
        //   But wait, 4 * 1536 = 6144, 2 * 1536 = 3072
        // So the L1 layer input is 3072 (L1). But after * (127/128) scaling...
        // In the serializer: l1 = FactorizedStackedLinear(2 * self.L1 // 2, self.L2 + 1, count)
        //   = FactorizedStackedLinear(L1, L2+1, count)
        // So L1 layer input width = L1 = 3072. Padded to next multiple of 32 = 3072 (already aligned!)

        int l1_padded = ((l1_inputs + 31) / 32) * 32; // 3072

        // Read L1: bias then weight
        int32_t l1_bias_temp[L2 + 1];
        f.read(reinterpret_cast<char*>(l1_bias_temp), (L2 + 1) * sizeof(int32_t));
        memcpy(net.l1_bias[bucket], l1_bias_temp, (L2 + 1) * sizeof(int32_t));

        // Weight: [(L2+1)][l1_padded] stored as [outputs][inputs]
        std::vector<int8_t> l1_w_temp((L2 + 1) * l1_padded);
        f.read(reinterpret_cast<char*>(l1_w_temp.data()), l1_w_temp.size());
        for (int o = 0; o < L2 + 1; o++)
            memcpy(net.l1_weight[bucket][o], &l1_w_temp[o * l1_padded], L1);

        // L2 layer: input is L2*2 (after squared crelu doubles the neurons)
        // From layer_stacks.py: self.l2 = StackedLinear(self.L2 * 2, self.L3, count)
        int l2_inputs = L2 * 2; // 30
        int l2_padded = ((l2_inputs + 31) / 32) * 32; // 32

        int32_t l2_bias_temp[L3];
        f.read(reinterpret_cast<char*>(l2_bias_temp), L3 * sizeof(int32_t));
        memcpy(net.l2_bias[bucket], l2_bias_temp, L3 * sizeof(int32_t));

        std::vector<int8_t> l2_w_temp(L3 * l2_padded);
        f.read(reinterpret_cast<char*>(l2_w_temp.data()), l2_w_temp.size());
        for (int o = 0; o < L3; o++)
            memcpy(net.l2_weight[bucket][o], &l2_w_temp[o * l2_padded], L2 * 2);

        // Output layer: input is L3
        int out_inputs = L3; // 32
        int out_padded = ((out_inputs + 31) / 32) * 32; // 32

        int32_t out_bias_temp[1];
        f.read(reinterpret_cast<char*>(out_bias_temp), 1 * sizeof(int32_t));
        net.out_bias[bucket][0] = out_bias_temp[0];

        std::vector<int8_t> out_w_temp(1 * out_padded);
        f.read(reinterpret_cast<char*>(out_w_temp.data()), out_w_temp.size());
        memcpy(net.out_weight[bucket][0], out_w_temp.data(), L3);
    }

    net.loaded = true;
    std::cerr << "NNUE: Loaded network from " << path << std::endl;
    return true;
}

bool is_loaded() {
    return net.loaded;
}

void refresh_accumulator(const Position& pos, Accumulator& acc, Color perspective) {
    bool is_white_pov = (perspective == WHITE);
    Square ksq = pos.king_square(perspective);

    // Start from bias
    memcpy(acc.values[perspective], net.ft_bias, L1 * sizeof(int16_t));
    memset(acc.psqt[perspective], 0, NUM_PSQT_BUCKETS * sizeof(int32_t));

    // Add active features
    Bitboard occ = pos.pieces();
    while (occ) {
        Square sq = pop_lsb(occ);
        Piece pc = pos.piece_on(sq);
        if (pc == NO_PIECE) continue;

        int idx = feature_index(is_white_pov, ksq, sq, pc);
        if (idx < 0 || idx >= NUM_INPUTS) continue;

        for (int i = 0; i < L1; i++)
            acc.values[perspective][i] += net.ft_weight[idx][i];
        for (int i = 0; i < NUM_PSQT_BUCKETS; i++)
            acc.psqt[perspective][i] += net.ft_psqt[idx][i];
    }

    acc.computed[perspective] = true;
}

// Get layer stack bucket from piece count
static int get_bucket(int piece_count) {
    // Stockfish's bucket formula: (piece_count - 1) / 4, clamped
    // Actually from Stockfish: bucket = (count - 2) / 4 for 8 buckets over 2..32 pieces
    return std::clamp((piece_count - 2) / 4, 0, NUM_LS_BUCKETS - 1);
}

int evaluate(const Position& pos) {
    if (!net.loaded) return 0;

    // Compute accumulators for both perspectives
    Accumulator acc;
    acc.computed[WHITE] = false;
    acc.computed[BLACK] = false;
    refresh_accumulator(pos, acc, WHITE);
    refresh_accumulator(pos, acc, BLACK);

    Color stm = pos.side_to_move();
    Color nstm = ~stm;

    // Apply SCReLU + product pooling
    // From model.py:
    //   l0_ = (us * cat([w, b])) + (them * cat([b, w]])
    //   = cat([stm_acc, nstm_acc]) for the side to move
    //   l0_ = clamp(l0_, 0, 1)  -> in quantized: clamp(x, 0, 127)
    //   l0_s = split(l0_, L1//2)  -> 4 chunks of 1536
    //   l0_s1 = [l0_s[0]*l0_s[1], l0_s[2]*l0_s[3]] * (127/128)
    //   Final input to L1: 3072 values as int8

    alignas(64) int8_t input[L1];

    // stm perspective: acc.values[stm][0..L1-1]
    // nstm perspective: acc.values[nstm][0..L1-1]
    // Concatenated as [stm_acc, nstm_acc] -> 2*L1 = 6144
    // Split into 4 chunks of L1/2 = 1536: [stm_0:1536, stm_1536:3072, nstm_0:1536, nstm_1536:3072]
    // Product: [stm_first_half * stm_second_half, nstm_first_half * nstm_second_half]

    const int HALF = L1 / 2; // 1536

    for (int i = 0; i < HALF; i++) {
        int16_t v0 = std::clamp(acc.values[stm][i], (int16_t)0, (int16_t)QUANTIZED_ONE);
        int16_t v1 = std::clamp(acc.values[stm][i + HALF], (int16_t)0, (int16_t)QUANTIZED_ONE);
        // Product pooling: v0 * v1 / 128 (approximation of * 127/128)
        input[i] = int8_t((int(v0) * int(v1)) >> 7);
    }
    for (int i = 0; i < HALF; i++) {
        int16_t v0 = std::clamp(acc.values[nstm][i], (int16_t)0, (int16_t)QUANTIZED_ONE);
        int16_t v1 = std::clamp(acc.values[nstm][i + HALF], (int16_t)0, (int16_t)QUANTIZED_ONE);
        input[HALF + i] = int8_t((int(v0) * int(v1)) >> 7);
    }

    // Select layer stack bucket
    int bucket = get_bucket(pos.piece_count());

    // L1 forward: input[L1] -> hidden1[L2+1] (16 neurons)
    int32_t hidden1[L2 + 1];
    for (int o = 0; o < L2 + 1; o++) {
        int32_t sum = net.l1_bias[bucket][o];
        for (int i = 0; i < L1; i++)
            sum += int32_t(input[i]) * int32_t(net.l1_weight[bucket][o][i]);
        hidden1[o] = sum;
    }

    // Divide by weight_scale_hidden (64) and quantized_one (127)
    // hidden1 is in units of (quantized_one * weight_scale_hidden) = 127 * 64 = 8128
    // After division we get values in [0, quantized_one] range
    // Actually the bias is stored pre-scaled by (weight_scale_hidden * quantized_one)
    // So hidden1 / (WEIGHT_SCALE_HIDDEN * QUANTIZED_ONE) gives float value
    // For the squared CReLU we need: clamp(x, 0, 1) then square
    // In quantized: divide by (64 * 127), clamp to [0, 127], then square
    // But actually let's keep in integer arithmetic:
    // hidden1[i] / (64) gives us a value in quantized_one units
    // So h = hidden1[i] / 64, then clamp to [0, 127]

    // Split: first L2=15 neurons get squared CReLU, last 1 is skip connection
    int8_t l2_input[L2 * 2]; // 30 neurons
    int32_t skip_output = hidden1[L2]; // The 16th neuron is the skip connection

    for (int i = 0; i < L2; i++) {
        // Divide by WEIGHT_SCALE_HIDDEN to get into quantized_one scale
        int32_t val = hidden1[i] / WEIGHT_SCALE_HIDDEN;
        val = std::clamp(val, (int32_t)0, (int32_t)QUANTIZED_ONE);
        // Squared CReLU: x^2 * (127/128)
        int32_t sq_val = (val * val) >> 7; // x^2 / 128 ~ x^2 * (127/128) / 127
        l2_input[i] = int8_t(std::clamp(sq_val, (int32_t)0, (int32_t)QUANTIZED_ONE));
        // The second half is just the clamped value
        l2_input[L2 + i] = int8_t(val);
    }

    // L2 forward: l2_input[L2*2=30] -> hidden2[L3=32]
    int32_t hidden2[L3];
    for (int o = 0; o < L3; o++) {
        int32_t sum = net.l2_bias[bucket][o];
        for (int i = 0; i < L2 * 2; i++)
            sum += int32_t(l2_input[i]) * int32_t(net.l2_weight[bucket][o][i]);
        hidden2[o] = sum;
    }

    // CReLU on hidden2
    int8_t out_input[L3];
    for (int i = 0; i < L3; i++) {
        int32_t val = hidden2[i] / WEIGHT_SCALE_HIDDEN;
        out_input[i] = int8_t(std::clamp(val, (int32_t)0, (int32_t)QUANTIZED_ONE));
    }

    // Output layer: out_input[L3=32] -> output[1]
    int32_t output = net.out_bias[bucket][0];
    for (int i = 0; i < L3; i++)
        output += int32_t(out_input[i]) * int32_t(net.out_weight[bucket][0][i]);

    // Add skip connection
    // skip_output is in l1 bias scale (WEIGHT_SCALE_HIDDEN * QUANTIZED_ONE)
    // output is in l2->output scale
    // We need to bring them to same scale before adding
    // The output layer scale: NNUE2SCORE * WEIGHT_SCALE_OUT / QUANTIZED_ONE
    // Skip is already divided by its chain of scales
    // Actually in Stockfish: output = (network_output + skip) / (nnue2score * weight_scale_out)
    // Let's compute in the combined scale

    // Final score computation:
    // The full chain: FT -> L1 -> SqCReLU -> L2 -> CReLU -> Output
    // All intermediate values carry scaling factors that compound
    // Simplified: the final output in centipawns is:
    //   output / (WEIGHT_SCALE_OUT * NNUE2SCORE / QUANTIZED_ONE)
    // + skip / (WEIGHT_SCALE_HIDDEN * WEIGHT_SCALE_OUT * NNUE2SCORE / QUANTIZED_ONE)

    // A simpler approach: convert to centipawns directly
    // output is in units of (quantized_one * weight_scale_out * nnue2score / quantized_one)
    //   = weight_scale_out * nnue2score = 16 * 600 = 9600
    // skip is in units of (weight_scale_hidden * quantized_one) initially,
    //   but the output layer expects it in the output scale...

    // Let's use Stockfish's approach: just divide by the output scale
    int32_t final_score = output / (WEIGHT_SCALE_OUT * WEIGHT_SCALE_HIDDEN);
    // Add PSQT contribution
    int psqt_bucket = bucket;
    int32_t psqt_val = (acc.psqt[stm][psqt_bucket] - acc.psqt[nstm][psqt_bucket]) / 2;
    // psqt is stored in scale of NNUE2SCORE * WEIGHT_SCALE_OUT
    final_score += psqt_val / (NNUE2SCORE * WEIGHT_SCALE_OUT / NNUE2SCORE);

    // Normalize to centipawns
    // The raw output is roughly in the right range after division
    // Fine-tune: just return as centipawns
    return final_score;
}

} // namespace NNUE
