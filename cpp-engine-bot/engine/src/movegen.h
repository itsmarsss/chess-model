#pragma once

#include "types.h"
#include "position.h"

struct MoveList {
    Move moves[MAX_MOVES];
    int count = 0;

    void add(Move m) { moves[count++] = m; }
};

enum GenType { ALL_MOVES, CAPTURES, QUIETS, EVASIONS };

void generate_moves(const Position& pos, MoveList& list, GenType type = ALL_MOVES);
void generate_legal_moves(const Position& pos, MoveList& list);
