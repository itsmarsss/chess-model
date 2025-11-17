#pragma once

#include <chrono>
#include <cstdint>

struct TimeManager {
    int64_t start_time;
    int64_t allocated_time;   // Soft limit
    int64_t max_time;         // Hard limit
    bool infinite;

    void init(int wtime, int btime, int winc, int binc, int movestogo, int movetime, bool is_white) {
        start_time = now();
        infinite = false;

        if (movetime > 0) {
            allocated_time = movetime - 50; // buffer for communication
            max_time = movetime - 10;
            return;
        }

        int time_left = is_white ? wtime : btime;
        int inc = is_white ? winc : binc;

        if (time_left <= 0) {
            allocated_time = 100;
            max_time = 200;
            return;
        }

        if (movestogo > 0) {
            allocated_time = time_left / movestogo + inc - 50;
            max_time = std::min(int64_t(time_left / 2), allocated_time * 5);
        } else {
            allocated_time = time_left / 20 + inc * 3 / 4;
            max_time = std::min(int64_t(time_left / 5), allocated_time * 5);
        }

        allocated_time = std::max(int64_t(50), allocated_time);
        max_time = std::max(int64_t(100), max_time);
    }

    void set_infinite() {
        start_time = now();
        infinite = true;
        allocated_time = INT64_MAX / 2;
        max_time = INT64_MAX / 2;
    }

    int64_t elapsed() const {
        return now() - start_time;
    }

    bool time_up() const {
        if (infinite) return false;
        return elapsed() >= allocated_time;
    }

    bool hard_limit() const {
        if (infinite) return false;
        return elapsed() >= max_time;
    }

    static int64_t now() {
        auto tp = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(tp).count();
    }
};
