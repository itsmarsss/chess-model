#include <iostream>
#include <string>
#include <cstring>
#include "Stockfish/src/position.h"
#include "Stockfish/src/bitboard.h"
#include "Stockfish/src/evaluate.h"
#include "Stockfish/src/nnue/network.h"
#include "Stockfish/src/types.h"
#include "Stockfish/src/uci.h"
#include "Stockfish/src/misc.h"

using namespace Stockfish;

constexpr auto StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Global position and network
Position* g_pos = nullptr;
StateInfo* g_si = nullptr;
Eval::NNUE::Networks* g_networks = nullptr;
Eval::NNUE::AccumulatorCaches* g_caches = nullptr;

extern "C" {

int nnue_init(const char* model_path) {
    try {
        // Initialize Stockfish components
        Bitboards::init();
        Position::init();
        
        // Create and load NNUE network
        std::string path(model_path);
        g_networks = new Eval::NNUE::Networks();
        g_networks->big.load(path);
        
        // Initialize caches
        g_caches = new Eval::NNUE::AccumulatorCaches(Eval::NNUE::Networks::Big);
        
        // Create position
        g_si = new StateInfo();
        g_pos = new Position();
        g_pos->set(StartFEN, false, g_si);
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "NNUE init error: " << e.what() << std::endl;
        return 1;
    }
}

int nnue_evaluate_fen(const char* fen) {
    if (!g_pos || !g_si || !g_networks || !g_caches) {
        return 0;
    }
    
    try {
        // Set position from FEN
        StateInfo si;
        g_pos->set(fen, false, &si);
        
        // Create accumulator stack
        Eval::NNUE::AccumulatorStack accumulators(*g_pos);
        
        // Evaluate
        Value eval = Eval::evaluate(*g_networks, *g_pos, accumulators, *g_caches, 0);
        
        return int(eval);
    } catch (const std::exception& e) {
        std::cerr << "NNUE eval error: " << e.what() << std::endl;
        return 0;
    }
}

void nnue_cleanup() {
    delete g_pos;
    delete g_si;
    delete g_networks;
    delete g_caches;
    g_pos = nullptr;
    g_si = nullptr;
    g_networks = nullptr;
    g_caches = nullptr;
}

}
