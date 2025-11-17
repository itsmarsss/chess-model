#pragma once

#include "types.h"
#include "bitboard.h"
#include <string>
#include <cstring>

struct StateInfo {
    uint64_t key;
    CastlingRight castling;
    Square en_passant;
    int halfmove_clock;
    Piece captured;
    int plies_from_null;

    // NNUE dirty piece info
    struct DirtyPiece {
        int num;
        Piece piece[3];
        Square from[3];
        Square to[3];
    } dirty;

    StateInfo* previous;
};

class Position {
public:
    Position();

    void set(const std::string& fen, StateInfo& si);
    void set_startpos(StateInfo& si);
    std::string fen() const;

    // Board access
    Piece piece_on(Square s) const { return board[s]; }
    Bitboard pieces() const { return by_color[WHITE] | by_color[BLACK]; }
    Bitboard pieces(Color c) const { return by_color[c]; }
    Bitboard pieces(PieceType pt) const { return by_type[pt]; }
    Bitboard pieces(Color c, PieceType pt) const { return by_color[c] & by_type[pt]; }
    Bitboard pieces(PieceType pt1, PieceType pt2) const { return by_type[pt1] | by_type[pt2]; }
    Bitboard pieces(Color c, PieceType pt1, PieceType pt2) const { return by_color[c] & (by_type[pt1] | by_type[pt2]); }
    Square king_square(Color c) const { return lsb(pieces(c, KING)); }

    Color side_to_move() const { return side; }
    CastlingRight castling_rights() const { return st->castling; }
    Square ep_square() const { return st->en_passant; }
    int halfmove_clock() const { return st->halfmove_clock; }
    uint64_t key() const { return st->key; }
    int game_ply() const { return game_ply_; }

    bool is_empty(Square s) const { return board[s] == NO_PIECE; }

    // Move execution
    void do_move(Move m, StateInfo& new_st);
    void undo_move(Move m);
    void do_null_move(StateInfo& new_st);
    void undo_null_move();

    // Attack detection
    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;
    bool is_attacked_by(Color c, Square s) const;
    bool in_check() const;
    Bitboard checkers() const;

    // Move validation
    bool is_legal(Move m) const;
    bool is_pseudo_legal(Move m) const;
    bool gives_check(Move m) const;

    // Position properties
    bool has_non_pawn_material(Color c) const;
    int piece_count() const { return popcount(pieces()); }
    int piece_count(Color c, PieceType pt) const { return popcount(pieces(c, pt)); }

    // Repetition
    bool is_draw() const;

    // State
    StateInfo* state() const { return st; }

private:
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);

    Piece board[SQUARE_NB];
    Bitboard by_type[PIECE_TYPE_NB];
    Bitboard by_color[COLOR_NB];
    Color side;
    int game_ply_;
    StateInfo* st;
};

// Zobrist key tables
namespace Zobrist {
    extern uint64_t psq[PIECE_NB][SQUARE_NB];
    extern uint64_t castling[16];
    extern uint64_t enpassant[8];
    extern uint64_t side;
    void init();
}
