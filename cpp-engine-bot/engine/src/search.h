#pragma once

#include "position.h"
#include "timeman.h"
#include "movepick.h"
#include <atomic>
#include <string>

struct SearchLimits {
    int depth = 0;
    int movetime = 0;
    int wtime = 0, btime = 0;
    int winc = 0, binc = 0;
    int movestogo = 0;
    bool infinite = false;
};

struct SearchInfo {
    uint64_t nodes;
    int seldepth;
    TimeManager tm;
    bool stopped;

    SearchStack stack[MAX_PLY + 4];
    SearchHistory history;

    Move pv[MAX_PLY][MAX_PLY];
    int pv_length[MAX_PLY];
};

namespace Search {
    extern std::atomic<bool> stop_signal;

    void init();
    Move search(Position& pos, SearchLimits& limits);
    std::string pv_string(SearchInfo& info, int ply);
}
