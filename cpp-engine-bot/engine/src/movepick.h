#pragma once

#include "types.h"
#include "movegen.h"

struct SearchStack {
    Move killers[2];
};

struct SearchHistory {
    int table[COLOR_NB][SQUARE_NB][SQUARE_NB];

    void clear() {
        for (int c = 0; c < COLOR_NB; c++)
            for (int f = 0; f < SQUARE_NB; f++)
                for (int t = 0; t < SQUARE_NB; t++)
                    table[c][f][t] = 0;
    }

    void update(Color c, Move m, int bonus) {
        int& entry = table[c][from_sq(m)][to_sq(m)];
        entry += bonus - entry * abs(bonus) / 16384;
    }

    int get(Color c, Move m) const {
        return table[c][from_sq(m)][to_sq(m)];
    }
};

// MVV-LVA score for move ordering
int mvv_lva_score(const Position& pos, Move m);

// Score and sort moves
void score_moves(const Position& pos, MoveList& list, Move tt_move,
                 const SearchStack* ss, const SearchHistory& history);

void pick_move(MoveList& list, int start);
