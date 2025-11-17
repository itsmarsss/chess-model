#include "uci.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include "movegen.h"
#include "nnue.h"
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace UCI {

static Position pos;
static StateInfo states[1024];
static int state_idx = 0;

static std::string nnue_path = "";

static Move parse_move(const Position& pos, const std::string& str) {
    if (str.length() < 4) return MOVE_NONE;

    Square from = make_square(str[0] - 'a', str[1] - '1');
    Square to = make_square(str[2] - 'a', str[3] - '1');

    MoveList moves;
    generate_legal_moves(pos, moves);

    for (int i = 0; i < moves.count; i++) {
        Move m = moves.moves[i];
        if (from_sq(m) == from && to_sq(m) == to) {
            if (str.length() == 5) {
                PieceType promo = NO_PIECE_TYPE;
                switch (str[4]) {
                    case 'n': promo = KNIGHT; break;
                    case 'b': promo = BISHOP; break;
                    case 'r': promo = ROOK; break;
                    case 'q': promo = QUEEN; break;
                }
                if (is_promotion(m) && promo_type(m) == promo) return m;
            } else {
                if (!is_promotion(m)) return m;
            }
        }
    }
    return MOVE_NONE;
}

static void position(std::istringstream& is) {
    std::string token;
    is >> token;

    state_idx = 0;

    if (token == "startpos") {
        pos.set_startpos(states[state_idx]);
        is >> token; // consume "moves" if present
    } else if (token == "fen") {
        std::string fen;
        while (is >> token && token != "moves")
            fen += token + " ";
        pos.set(fen, states[state_idx]);
    }

    // Apply moves
    while (is >> token) {
        Move m = parse_move(pos, token);
        if (m != MOVE_NONE) {
            state_idx++;
            pos.do_move(m, states[state_idx]);
        }
    }
}

static void go(std::istringstream& is) {
    SearchLimits limits;
    std::string token;

    while (is >> token) {
        if (token == "depth") is >> limits.depth;
        else if (token == "movetime") is >> limits.movetime;
        else if (token == "wtime") is >> limits.wtime;
        else if (token == "btime") is >> limits.btime;
        else if (token == "winc") is >> limits.winc;
        else if (token == "binc") is >> limits.binc;
        else if (token == "movestogo") is >> limits.movestogo;
        else if (token == "infinite") limits.infinite = true;
    }

    Move best = Search::search(pos, limits);
    std::cout << "bestmove " << move_to_uci(best) << std::endl;
}

static void setoption(std::istringstream& is) {
    std::string token, name, value;
    is >> token; // "name"

    while (is >> token && token != "value")
        name += (name.empty() ? "" : " ") + token;

    while (is >> token)
        value += (value.empty() ? "" : " ") + token;

    if (name == "Hash") {
        int mb = std::stoi(value);
        TT.resize(mb);
    } else if (name == "EvalFile") {
        nnue_path = value;
    }
}

static void perft_helper(Position& pos, int depth, uint64_t& nodes) {
    if (depth == 0) { nodes++; return; }

    MoveList moves;
    generate_legal_moves(pos, moves);

    StateInfo st;
    for (int i = 0; i < moves.count; i++) {
        pos.do_move(moves.moves[i], st);
        perft_helper(pos, depth - 1, nodes);
        pos.undo_move(moves.moves[i]);
    }
}

static void perft(int depth) {
    uint64_t total = 0;
    MoveList moves;
    generate_legal_moves(pos, moves);

    StateInfo st;
    for (int i = 0; i < moves.count; i++) {
        pos.do_move(moves.moves[i], st);
        uint64_t count = 0;
        perft_helper(pos, depth - 1, count);
        pos.undo_move(moves.moves[i]);
        std::cout << move_to_uci(moves.moves[i]) << ": " << count << std::endl;
        total += count;
    }
    std::cout << "\nTotal: " << total << std::endl;
}

void loop() {
    // Initialize with default hash size
    TT.resize(64);
    pos.set_startpos(states[0]);

    std::string line, token;

    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        is >> token;

        if (token == "uci") {
            std::cout << "id name ChessHacks-NNUE 1.0" << std::endl;
            std::cout << "id author Aadi" << std::endl;
            std::cout << "option name Hash type spin default 64 min 1 max 1024" << std::endl;
            std::cout << "option name EvalFile type string default <internal>" << std::endl;
            std::cout << "uciok" << std::endl;
        }
        else if (token == "isready") {
            if (!nnue_path.empty() && !NNUE::is_loaded()) {
                NNUE::init(nnue_path);
            }
            std::cout << "readyok" << std::endl;
        }
        else if (token == "ucinewgame") {
            TT.clear();
            state_idx = 0;
            pos.set_startpos(states[0]);
        }
        else if (token == "position") {
            position(is);
        }
        else if (token == "go") {
            // Check if it's a perft command
            std::string next;
            std::streampos pos_before = is.tellg();
            if (is >> next && next == "perft") {
                int depth;
                is >> depth;
                perft(depth);
            } else {
                is.seekg(pos_before);
                go(is);
            }
        }
        else if (token == "stop") {
            Search::stop_signal.store(true);
        }
        else if (token == "setoption") {
            setoption(is);
        }
        else if (token == "quit") {
            break;
        }
        else if (token == "d") {
            std::cout << pos.fen() << std::endl;
        }
    }
}

} // namespace UCI
