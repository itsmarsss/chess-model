#include "eval.h"
#include "bitboard.h"
#include "nnue.h"

namespace Eval {

int evaluate(const Position& pos) {
    if (NNUE::is_loaded()) {
        return NNUE::evaluate(pos);
    }
    return hce(pos);
}

// Material values in centipawns
constexpr int PieceValue[PIECE_TYPE_NB] = { 0, 100, 320, 330, 500, 900, 20000 };

// Piece-square tables (from white's perspective, rank 1 at index 0)
// Middlegame PST
constexpr int PawnPST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

constexpr int KnightPST[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

constexpr int BishopPST[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10, 10,  5, 10, 10,  5, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

constexpr int RookPST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};

constexpr int QueenPST[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

constexpr int KingMGPST[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

constexpr int KingEGPST[64] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
};

const int* PST[PIECE_TYPE_NB] = {
    nullptr, PawnPST, KnightPST, BishopPST, RookPST, QueenPST, KingMGPST
};

inline int pst_index(Color c, Square s) {
    // PST is stored from white's perspective (rank 1 at bottom = index 56-63 in table)
    // For white: flip rank so rank 1 is at bottom of table
    // For black: no flip (already reversed)
    return (c == WHITE) ? (Square((7 - rank_of(s)) * 8 + file_of(s))) : (Square(rank_of(s) * 8 + file_of(s)));
}

int hce(const Position& pos) {
    int mg_score = 0;
    int eg_score = 0;
    int game_phase = 0;

    // Phase weights
    constexpr int PhaseWeight[PIECE_TYPE_NB] = { 0, 0, 1, 1, 2, 4, 0 };
    constexpr int TotalPhase = 24; // 4*1 + 4*1 + 4*2 + 2*4

    for (Color c : {WHITE, BLACK}) {
        int sign = (c == WHITE) ? 1 : -1;

        for (PieceType pt = PAWN; pt <= KING; pt = PieceType(pt + 1)) {
            Bitboard bb = pos.pieces(c, pt);
            while (bb) {
                Square s = pop_lsb(bb);
                int idx = pst_index(c, s);

                // Material
                mg_score += sign * PieceValue[pt];
                eg_score += sign * PieceValue[pt];

                // PST
                mg_score += sign * PST[pt][idx];
                if (pt == KING)
                    eg_score += sign * KingEGPST[idx];
                else
                    eg_score += sign * PST[pt][idx];

                game_phase += PhaseWeight[pt];
            }
        }

        // Bishop pair bonus
        if (popcount(pos.pieces(c, BISHOP)) >= 2) {
            mg_score += sign * 30;
            eg_score += sign * 50;
        }

        // Rook on open/semi-open file
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square s = pop_lsb(rooks);
            Bitboard file_bb = FileABB << file_of(s);
            if (!(pos.pieces(c, PAWN) & file_bb)) {
                if (!(pos.pieces(~c, PAWN) & file_bb))
                    mg_score += sign * 15; // Open file
                else
                    mg_score += sign * 10; // Semi-open file
            }
        }

        // Pawn structure
        Bitboard pawns = pos.pieces(c, PAWN);
        Bitboard temp_pawns = pawns;
        while (temp_pawns) {
            Square s = pop_lsb(temp_pawns);
            Bitboard file_bb = FileABB << file_of(s);

            // Doubled pawns
            if (popcount(pawns & file_bb) > 1) {
                mg_score -= sign * 10;
                eg_score -= sign * 15;
            }

            // Isolated pawns
            Bitboard adjacent_files = 0;
            if (file_of(s) > 0) adjacent_files |= FileABB << (file_of(s) - 1);
            if (file_of(s) < 7) adjacent_files |= FileABB << (file_of(s) + 1);
            if (!(pawns & adjacent_files)) {
                mg_score -= sign * 15;
                eg_score -= sign * 20;
            }

            // Passed pawns
            Bitboard front_span;
            if (c == WHITE) {
                front_span = file_bb | adjacent_files;
                // Mask to ranks above this pawn
                for (int r = 0; r <= rank_of(s); r++)
                    front_span &= ~(Rank1BB << (r * 8));
            } else {
                front_span = file_bb | adjacent_files;
                for (int r = rank_of(s); r < 8; r++)
                    front_span &= ~(Rank1BB << (r * 8));
            }
            if (!(pos.pieces(~c, PAWN) & front_span)) {
                int rank_bonus = (c == WHITE) ? rank_of(s) : (7 - rank_of(s));
                mg_score += sign * (10 + rank_bonus * 10);
                eg_score += sign * (20 + rank_bonus * 15);
            }
        }

        // Mobility (simplified)
        Bitboard occ = pos.pieces();
        Bitboard mobility_area = ~pos.pieces(c, PAWN) & ~pos.pieces(c, KING);

        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            Square s = pop_lsb(knights);
            int mob = popcount(knight_attacks(s) & ~pos.pieces(c));
            mg_score += sign * (mob - 4) * 4;
            eg_score += sign * (mob - 4) * 4;
        }

        Bitboard bishops = pos.pieces(c, BISHOP);
        while (bishops) {
            Square s = pop_lsb(bishops);
            int mob = popcount(bishop_attacks(s, occ) & ~pos.pieces(c));
            mg_score += sign * (mob - 6) * 5;
            eg_score += sign * (mob - 6) * 5;
        }

        Bitboard rooks2 = pos.pieces(c, ROOK);
        while (rooks2) {
            Square s = pop_lsb(rooks2);
            int mob = popcount(rook_attacks(s, occ) & ~pos.pieces(c));
            mg_score += sign * (mob - 7) * 3;
            eg_score += sign * (mob - 7) * 3;
        }
    }

    // Tapered evaluation
    game_phase = std::min(game_phase, TotalPhase);
    int score = (mg_score * game_phase + eg_score * (TotalPhase - game_phase)) / TotalPhase;

    // Return from side-to-move perspective
    return (pos.side_to_move() == WHITE) ? score : -score;
}

} // namespace Eval
