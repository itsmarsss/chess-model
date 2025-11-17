#include "movepick.h"

// MVV-LVA: Most Valuable Victim - Least Valuable Attacker
static constexpr int VictimScore[PIECE_TYPE_NB] = { 0, 100, 300, 310, 500, 900, 10000 };
static constexpr int AttackerScore[PIECE_TYPE_NB] = { 0, 5, 4, 3, 2, 1, 0 };

int mvv_lva_score(const Position& pos, Move m) {
    Piece victim = pos.piece_on(to_sq(m));
    Piece attacker = pos.piece_on(from_sq(m));
    if (is_en_passant(m)) victim = make_piece(~pos.side_to_move(), PAWN);
    if (victim == NO_PIECE) return 0;
    return VictimScore[type_of(victim)] * 10 + AttackerScore[type_of(attacker)];
}

// Move scores stored alongside moves for sorting
static int move_scores[MAX_MOVES];

void score_moves(const Position& pos, MoveList& list, Move tt_move,
                 const SearchStack* ss, const SearchHistory& history) {
    for (int i = 0; i < list.count; i++) {
        Move m = list.moves[i];

        if (m == tt_move) {
            move_scores[i] = 1000000;
        } else if (pos.piece_on(to_sq(m)) != NO_PIECE || is_en_passant(m)) {
            move_scores[i] = 500000 + mvv_lva_score(pos, m);
        } else if (is_promotion(m)) {
            move_scores[i] = 400000 + (promo_type(m) == QUEEN ? 1000 : 0);
        } else if (ss && m == ss->killers[0]) {
            move_scores[i] = 200000;
        } else if (ss && m == ss->killers[1]) {
            move_scores[i] = 190000;
        } else {
            move_scores[i] = history.get(pos.side_to_move(), m);
        }
    }
}

void pick_move(MoveList& list, int start) {
    int best_idx = start;
    int best_score = move_scores[start];

    for (int i = start + 1; i < list.count; i++) {
        if (move_scores[i] > best_score) {
            best_score = move_scores[i];
            best_idx = i;
        }
    }

    if (best_idx != start) {
        std::swap(list.moves[start], list.moves[best_idx]);
        std::swap(move_scores[start], move_scores[best_idx]);
    }
}
