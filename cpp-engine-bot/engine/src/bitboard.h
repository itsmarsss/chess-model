#pragma once

#include "types.h"
#include <cstdint>

using Bitboard = uint64_t;

constexpr Bitboard FileABB = 0x0101010101010101ULL;
constexpr Bitboard FileBBB = FileABB << 1;
constexpr Bitboard FileCBB = FileABB << 2;
constexpr Bitboard FileDBB = FileABB << 3;
constexpr Bitboard FileEBB = FileABB << 4;
constexpr Bitboard FileFBB = FileABB << 5;
constexpr Bitboard FileGBB = FileABB << 6;
constexpr Bitboard FileHBB = FileABB << 7;

constexpr Bitboard Rank1BB = 0xFFULL;
constexpr Bitboard Rank2BB = Rank1BB << 8;
constexpr Bitboard Rank3BB = Rank1BB << 16;
constexpr Bitboard Rank4BB = Rank1BB << 24;
constexpr Bitboard Rank5BB = Rank1BB << 32;
constexpr Bitboard Rank6BB = Rank1BB << 40;
constexpr Bitboard Rank7BB = Rank1BB << 48;
constexpr Bitboard Rank8BB = Rank1BB << 56;

constexpr Bitboard square_bb(Square s) { return 1ULL << s; }

inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline Square lsb(Bitboard b) { return Square(__builtin_ctzll(b)); }
inline Square msb(Bitboard b) { return Square(63 - __builtin_clzll(b)); }
inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

constexpr Bitboard shift_north(Bitboard b) { return b << 8; }
constexpr Bitboard shift_south(Bitboard b) { return b >> 8; }
constexpr Bitboard shift_east(Bitboard b) { return (b & ~FileHBB) << 1; }
constexpr Bitboard shift_west(Bitboard b) { return (b & ~FileABB) >> 1; }
constexpr Bitboard shift_ne(Bitboard b) { return (b & ~FileHBB) << 9; }
constexpr Bitboard shift_nw(Bitboard b) { return (b & ~FileABB) << 7; }
constexpr Bitboard shift_se(Bitboard b) { return (b & ~FileHBB) >> 7; }
constexpr Bitboard shift_sw(Bitboard b) { return (b & ~FileABB) >> 9; }

// Attack tables
extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
extern Bitboard KnightAttacks[SQUARE_NB];
extern Bitboard KingAttacks[SQUARE_NB];
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];

// Magic bitboard structures
struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard* attacks;
    unsigned shift;

    unsigned index(Bitboard occupied) const {
        return unsigned(((occupied & mask) * magic) >> shift);
    }
};

extern Magic BishopMagics[SQUARE_NB];
extern Magic RookMagics[SQUARE_NB];

inline Bitboard bishop_attacks(Square s, Bitboard occupied) {
    return BishopMagics[s].attacks[BishopMagics[s].index(occupied)];
}

inline Bitboard rook_attacks(Square s, Bitboard occupied) {
    return RookMagics[s].attacks[RookMagics[s].index(occupied)];
}

inline Bitboard queen_attacks(Square s, Bitboard occupied) {
    return bishop_attacks(s, occupied) | rook_attacks(s, occupied);
}

inline Bitboard pawn_attacks(Color c, Square s) { return PawnAttacks[c][s]; }
inline Bitboard knight_attacks(Square s) { return KnightAttacks[s]; }
inline Bitboard king_attacks(Square s) { return KingAttacks[s]; }

void bitboard_init();
