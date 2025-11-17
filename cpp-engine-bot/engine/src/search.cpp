#include "search.h"
#include "eval.h"
#include "tt.h"
#include "movegen.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace Search {

std::atomic<bool> stop_signal{false};

// LMR reduction table
static int LMR[64][64];

void init() {
    for (int d = 0; d < 64; d++)
        for (int m = 0; m < 64; m++)
            LMR[d][m] = int(0.75 + std::log(d) * std::log(m) / 2.25);
}

static bool check_time(SearchInfo& info) {
    if (info.nodes % 2048 != 0) return false;
    if (stop_signal.load(std::memory_order_relaxed)) {
        info.stopped = true;
        return true;
    }
    if (info.tm.hard_limit()) {
        info.stopped = true;
        return true;
    }
    return false;
}

static int quiescence(Position& pos, int alpha, int beta, int ply, SearchInfo& info) {
    if (check_time(info)) return 0;

    info.nodes++;
    if (ply > info.seldepth) info.seldepth = ply;

    int stand_pat = Eval::evaluate(pos);

    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    MoveList captures;
    generate_moves(pos, captures, CAPTURES);

    // Score captures by MVV-LVA
    score_moves(pos, captures, MOVE_NONE, nullptr, info.history);

    StateInfo st;
    for (int i = 0; i < captures.count; i++) {
        pick_move(captures, i);
        Move m = captures.moves[i];

        // Delta pruning: skip captures that can't improve alpha
        if (stand_pat + 900 < alpha && !is_promotion(m)) continue;

        pos.do_move(m, st);

        // Skip if leaves us in check (illegal)
        if (pos.is_attacked_by(pos.side_to_move(), pos.king_square(~pos.side_to_move()))) {
            pos.undo_move(m);
            continue;
        }

        int score = -quiescence(pos, -beta, -alpha, ply + 1, info);
        pos.undo_move(m);

        if (info.stopped) return 0;

        if (score > alpha) {
            alpha = score;
            if (score >= beta) return beta;
        }
    }

    return alpha;
}

static int negamax(Position& pos, int depth, int alpha, int beta, int ply, SearchInfo& info, bool do_null) {
    if (check_time(info)) return 0;

    info.pv_length[ply] = ply;

    if (ply > 0 && pos.is_draw()) return 0;

    bool in_check = pos.in_check();

    // Check extension
    if (in_check) depth++;

    if (depth <= 0) return quiescence(pos, alpha, beta, ply, info);

    info.nodes++;
    if (ply > info.seldepth) info.seldepth = ply;
    if (ply >= MAX_PLY - 1) return Eval::evaluate(pos);

    bool pv_node = (beta - alpha > 1);
    int orig_alpha = alpha;

    // TT probe
    bool tt_hit;
    TTEntry* tte = TT.probe(pos.key(), tt_hit);
    Move tt_move = tt_hit ? tte->best_move : MOVE_NONE;
    int tt_score = tt_hit ? int(tte->score) : VALUE_NONE;

    if (tt_hit && !pv_node && tte->depth >= depth) {
        if (tte->flag() == TT_EXACT) return tt_score;
        if (tte->flag() == TT_BETA && tt_score >= beta) return tt_score;
        if (tte->flag() == TT_ALPHA && tt_score <= alpha) return tt_score;
    }

    int static_eval = in_check ? -VALUE_INFINITE : Eval::evaluate(pos);

    // Null move pruning
    if (do_null && !in_check && !pv_node && depth >= 3 &&
        static_eval >= beta && pos.has_non_pawn_material(pos.side_to_move())) {
        int R = 3 + depth / 6;
        StateInfo st;
        pos.do_null_move(st);
        int null_score = -negamax(pos, depth - 1 - R, -beta, -beta + 1, ply + 1, info, false);
        pos.undo_null_move();

        if (info.stopped) return 0;
        if (null_score >= beta) {
            if (null_score >= VALUE_MATE_IN_MAX_PLY) null_score = beta;
            return null_score;
        }
    }

    // Futility pruning
    bool futility_pruning = false;
    if (!pv_node && !in_check && depth <= 3 && static_eval + 200 * depth <= alpha) {
        futility_pruning = true;
    }

    MoveList moves;
    generate_moves(pos, moves, ALL_MOVES);
    score_moves(pos, moves, tt_move, &info.stack[ply], info.history);

    int best_score = -VALUE_INFINITE;
    Move best_move = MOVE_NONE;
    int moves_searched = 0;

    StateInfo st;
    for (int i = 0; i < moves.count; i++) {
        pick_move(moves, i);
        Move m = moves.moves[i];

        pos.do_move(m, st);

        // Legality check
        if (pos.is_attacked_by(pos.side_to_move(), pos.king_square(~pos.side_to_move()))) {
            pos.undo_move(m);
            continue;
        }

        moves_searched++;
        bool is_capture = pos.state()->captured != NO_PIECE;
        bool is_quiet = !is_capture && !is_promotion(m);

        // Futility pruning for quiet moves
        if (futility_pruning && moves_searched > 1 && is_quiet && !pos.in_check()) {
            pos.undo_move(m);
            continue;
        }

        // Late move pruning
        if (!pv_node && !in_check && depth <= 3 && moves_searched > 3 + depth * 3 && is_quiet) {
            pos.undo_move(m);
            continue;
        }

        int score;

        // Late move reductions
        if (moves_searched > 3 && depth >= 3 && is_quiet && !in_check) {
            int reduction = LMR[std::min(depth, 63)][std::min(moves_searched, 63)];
            if (!pv_node) reduction++;
            reduction = std::max(0, std::min(reduction, depth - 2));

            score = -negamax(pos, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, info, true);

            if (score > alpha && reduction > 0) {
                score = -negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1, info, true);
            }
        } else if (!pv_node || moves_searched > 1) {
            score = -negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1, info, true);
        } else {
            score = alpha + 1; // Ensure full window search for first PV move
        }

        // Full window re-search for PV nodes
        if (pv_node && (moves_searched == 1 || score > alpha)) {
            score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, info, true);
        }

        pos.undo_move(m);

        if (info.stopped) return 0;

        if (score > best_score) {
            best_score = score;
            best_move = m;

            if (score > alpha) {
                alpha = score;

                // Update PV
                info.pv[ply][ply] = m;
                for (int j = ply + 1; j < info.pv_length[ply + 1]; j++)
                    info.pv[ply][j] = info.pv[ply + 1][j];
                info.pv_length[ply] = info.pv_length[ply + 1];

                if (score >= beta) {
                    // Update killer moves and history for quiet moves
                    if (is_quiet) {
                        info.stack[ply].killers[1] = info.stack[ply].killers[0];
                        info.stack[ply].killers[0] = m;
                        info.history.update(pos.side_to_move(), m, depth * depth);
                    }
                    break;
                }
            }
        }
    }

    if (moves_searched == 0) {
        return in_check ? -VALUE_MATE + ply : 0;
    }

    // Store in TT
    TTFlag flag;
    if (best_score >= beta) flag = TT_BETA;
    else if (best_score > orig_alpha) flag = TT_EXACT;
    else flag = TT_ALPHA;

    TT.store(pos.key(), best_score, static_eval, best_move, depth, flag);

    return best_score;
}

Move search(Position& pos, SearchLimits& limits) {
    SearchInfo info;
    memset(&info, 0, sizeof(info));
    info.nodes = 0;
    info.stopped = false;
    info.history.clear();
    stop_signal.store(false);

    // Initialize time management
    if (limits.infinite) {
        info.tm.set_infinite();
    } else if (limits.movetime > 0) {
        info.tm.init(0, 0, 0, 0, 0, limits.movetime, pos.side_to_move() == WHITE);
    } else {
        info.tm.init(limits.wtime, limits.btime, limits.winc, limits.binc,
                     limits.movestogo, 0, pos.side_to_move() == WHITE);
    }

    int max_depth = limits.depth > 0 ? limits.depth : MAX_PLY - 1;
    Move best_move = MOVE_NONE;
    int best_score = -VALUE_INFINITE;

    TT.new_search();

    // Iterative deepening
    for (int depth = 1; depth <= max_depth; depth++) {
        info.seldepth = 0;

        int score;
        int alpha = -VALUE_INFINITE;
        int beta = VALUE_INFINITE;

        // Aspiration windows (after depth 4)
        if (depth >= 4) {
            int window = 25;
            alpha = best_score - window;
            beta = best_score + window;

            score = negamax(pos, depth, alpha, beta, 0, info, true);

            // Widen window if we failed
            if (!info.stopped && (score <= alpha || score >= beta)) {
                alpha = -VALUE_INFINITE;
                beta = VALUE_INFINITE;
                score = negamax(pos, depth, alpha, beta, 0, info, true);
            }
        } else {
            score = negamax(pos, depth, alpha, beta, 0, info, true);
        }

        if (info.stopped && depth > 1) break;

        best_score = score;
        if (info.pv_length[0] > 0)
            best_move = info.pv[0][0];

        // Print info
        int64_t elapsed = info.tm.elapsed();
        uint64_t nps = elapsed > 0 ? info.nodes * 1000 / elapsed : 0;

        std::cout << "info depth " << depth
                  << " seldepth " << info.seldepth
                  << " score ";
        if (abs(score) >= VALUE_MATE_IN_MAX_PLY)
            std::cout << "mate " << (score > 0 ? (VALUE_MATE - score + 1) / 2 : -(VALUE_MATE + score) / 2);
        else
            std::cout << "cp " << score;
        std::cout << " nodes " << info.nodes
                  << " time " << elapsed
                  << " nps " << nps
                  << " pv";

        for (int i = 0; i < info.pv_length[0]; i++)
            std::cout << " " << move_to_uci(info.pv[0][i]);
        std::cout << std::endl;

        // Check soft time limit
        if (!limits.infinite && !limits.depth && info.tm.time_up()) break;
    }

    return best_move;
}

} // namespace Search
