#pragma once

#include <cstdint>
#include <string>
#include <cassert>

constexpr int MAX_PLY = 128;
constexpr int MAX_MOVES = 256;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };

constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int {
    NO_PIECE_TYPE = 0,
    PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6,
    PIECE_TYPE_NB = 7
};

enum Piece : int {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT = 2, W_BISHOP = 3, W_ROOK = 4, W_QUEEN = 5, W_KING = 6,
    B_PAWN = 9, B_KNIGHT = 10, B_BISHOP = 11, B_ROOK = 12, B_QUEEN = 13, B_KING = 14,
    PIECE_NB = 16
};

constexpr Piece make_piece(Color c, PieceType pt) {
    return Piece((c << 3) | pt);
}
constexpr Color color_of(Piece p) { return Color(p >> 3); }
constexpr PieceType type_of(Piece p) { return PieceType(p & 7); }

enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE = 64, SQUARE_NB = 64
};

constexpr int file_of(Square s) { return s & 7; }
constexpr int rank_of(Square s) { return s >> 3; }
constexpr Square make_square(int f, int r) { return Square((r << 3) | f); }
constexpr Square flip_rank(Square s) { return Square(s ^ 56); }
constexpr Square flip_file(Square s) { return Square(s ^ 7); }

enum Direction : int {
    NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1,
    NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9
};

constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
constexpr Square& operator+=(Square& s, Direction d) { return s = s + d; }

// Move encoding: 16 bits
// bits 0-5: from square
// bits 6-11: to square
// bits 12-13: promotion piece (0=knight, 1=bishop, 2=rook, 3=queen)
// bits 14-15: flags (0=normal, 1=promotion, 2=en passant, 3=castling)
enum Move : uint16_t { MOVE_NONE = 0, MOVE_NULL = 65 };

enum MoveFlags : uint16_t {
    FLAG_NORMAL = 0,
    FLAG_PROMOTION = 1 << 14,
    FLAG_EN_PASSANT = 2 << 14,
    FLAG_CASTLING = 3 << 14
};

constexpr Move make_move(Square from, Square to) {
    return Move((from) | (to << 6));
}
constexpr Move make_move(Square from, Square to, MoveFlags flags, PieceType promo = KNIGHT) {
    return Move((from) | (to << 6) | flags | ((promo - KNIGHT) << 12));
}

constexpr Square from_sq(Move m) { return Square(m & 0x3F); }
constexpr Square to_sq(Move m) { return Square((m >> 6) & 0x3F); }
constexpr MoveFlags flags_of(Move m) { return MoveFlags(m & 0xC000); }
constexpr PieceType promo_type(Move m) { return PieceType(((m >> 12) & 3) + KNIGHT); }

constexpr bool is_promotion(Move m) { return flags_of(m) == FLAG_PROMOTION; }
constexpr bool is_en_passant(Move m) { return flags_of(m) == FLAG_EN_PASSANT; }
constexpr bool is_castling(Move m) { return flags_of(m) == FLAG_CASTLING; }

// Castling rights
enum CastlingRight : int {
    NO_CASTLING = 0,
    WHITE_OO = 1, WHITE_OOO = 2,
    BLACK_OO = 4, BLACK_OOO = 8,
    ALL_CASTLING = 15
};

constexpr CastlingRight operator|(CastlingRight a, CastlingRight b) { return CastlingRight(int(a) | int(b)); }
constexpr CastlingRight operator&(CastlingRight a, CastlingRight b) { return CastlingRight(int(a) & int(b)); }
constexpr CastlingRight operator~(CastlingRight a) { return CastlingRight(~int(a) & 15); }
constexpr CastlingRight& operator|=(CastlingRight& a, CastlingRight b) { return a = a | b; }
constexpr CastlingRight& operator&=(CastlingRight& a, CastlingRight b) { return a = a & b; }

// Score constants
constexpr int VALUE_INFINITE = 32000;
constexpr int VALUE_MATE = 31000;
constexpr int VALUE_NONE = 32001;
constexpr int VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;

inline std::string square_to_string(Square s) {
    return std::string(1, 'a' + file_of(s)) + std::string(1, '1' + rank_of(s));
}

inline std::string move_to_uci(Move m) {
    if (m == MOVE_NONE) return "0000";
    std::string s = square_to_string(from_sq(m)) + square_to_string(to_sq(m));
    if (is_promotion(m)) {
        const char promo_chars[] = "nbrq";
        s += promo_chars[promo_type(m) - KNIGHT];
    }
    return s;
}
