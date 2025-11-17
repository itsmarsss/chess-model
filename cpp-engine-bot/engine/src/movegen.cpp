#include "movegen.h"

namespace {

void generate_pawn_moves(const Position& pos, MoveList& list, GenType type) {
    Color us = pos.side_to_move();
    Color them = ~us;
    Direction up = (us == WHITE) ? NORTH : SOUTH;
    Direction up_left = (us == WHITE) ? NORTH_WEST : SOUTH_WEST;
    Direction up_right = (us == WHITE) ? NORTH_EAST : SOUTH_EAST;
    Bitboard rank7 = (us == WHITE) ? Rank7BB : Rank2BB;
    Bitboard rank3 = (us == WHITE) ? Rank3BB : Rank6BB;
    Bitboard enemies = pos.pieces(them);
    Bitboard empty = ~pos.pieces();
    Bitboard pawns = pos.pieces(us, PAWN);

    // Promotions (from rank 7)
    Bitboard promo_pawns = pawns & rank7;
    Bitboard non_promo_pawns = pawns & ~rank7;

    if (promo_pawns) {
        // Promotion pushes
        if (type != CAPTURES) {
            Bitboard targets = (us == WHITE) ? shift_north(promo_pawns) : shift_south(promo_pawns);
            targets &= empty;
            while (targets) {
                Square to = pop_lsb(targets);
                Square from = to - up;
                list.add(make_move(from, to, FLAG_PROMOTION, QUEEN));
                list.add(make_move(from, to, FLAG_PROMOTION, ROOK));
                list.add(make_move(from, to, FLAG_PROMOTION, BISHOP));
                list.add(make_move(from, to, FLAG_PROMOTION, KNIGHT));
            }
        }
        // Promotion captures
        Bitboard cap_left = (us == WHITE) ? shift_nw(promo_pawns) : shift_sw(promo_pawns);
        Bitboard cap_right = (us == WHITE) ? shift_ne(promo_pawns) : shift_se(promo_pawns);
        cap_left &= enemies;
        cap_right &= enemies;
        while (cap_left) {
            Square to = pop_lsb(cap_left);
            Square from = to - up_left;
            list.add(make_move(from, to, FLAG_PROMOTION, QUEEN));
            list.add(make_move(from, to, FLAG_PROMOTION, ROOK));
            list.add(make_move(from, to, FLAG_PROMOTION, BISHOP));
            list.add(make_move(from, to, FLAG_PROMOTION, KNIGHT));
        }
        while (cap_right) {
            Square to = pop_lsb(cap_right);
            Square from = to - up_right;
            list.add(make_move(from, to, FLAG_PROMOTION, QUEEN));
            list.add(make_move(from, to, FLAG_PROMOTION, ROOK));
            list.add(make_move(from, to, FLAG_PROMOTION, BISHOP));
            list.add(make_move(from, to, FLAG_PROMOTION, KNIGHT));
        }
    }

    // Non-promotion moves
    if (type != CAPTURES) {
        // Single push
        Bitboard push1 = (us == WHITE) ? shift_north(non_promo_pawns) : shift_south(non_promo_pawns);
        push1 &= empty;
        // Double push
        Bitboard push2 = (us == WHITE) ? shift_north(push1 & rank3) : shift_south(push1 & rank3);
        push2 &= empty;

        while (push1) {
            Square to = pop_lsb(push1);
            list.add(make_move(to - up, to));
        }
        while (push2) {
            Square to = pop_lsb(push2);
            list.add(make_move(to - up - up, to));
        }
    }

    if (type != QUIETS) {
        // Captures
        Bitboard cap_left = (us == WHITE) ? shift_nw(non_promo_pawns) : shift_sw(non_promo_pawns);
        Bitboard cap_right = (us == WHITE) ? shift_ne(non_promo_pawns) : shift_se(non_promo_pawns);
        cap_left &= enemies;
        cap_right &= enemies;

        while (cap_left) {
            Square to = pop_lsb(cap_left);
            list.add(make_move(to - up_left, to));
        }
        while (cap_right) {
            Square to = pop_lsb(cap_right);
            list.add(make_move(to - up_right, to));
        }

        // En passant
        Square ep = pos.ep_square();
        if (ep != SQ_NONE) {
            Bitboard ep_pawns = pawn_attacks(them, ep) & non_promo_pawns;
            while (ep_pawns) {
                Square from = pop_lsb(ep_pawns);
                list.add(make_move(from, ep, FLAG_EN_PASSANT));
            }
        }
    }
}

void generate_piece_moves(const Position& pos, MoveList& list, PieceType pt, GenType type) {
    Color us = pos.side_to_move();
    Bitboard pieces_bb = pos.pieces(us, pt);
    Bitboard targets;

    if (type == CAPTURES)
        targets = pos.pieces(~us);
    else if (type == QUIETS)
        targets = ~pos.pieces();
    else
        targets = ~pos.pieces(us);

    while (pieces_bb) {
        Square from = pop_lsb(pieces_bb);
        Bitboard attacks;

        switch (pt) {
            case KNIGHT: attacks = knight_attacks(from); break;
            case BISHOP: attacks = bishop_attacks(from, pos.pieces()); break;
            case ROOK:   attacks = rook_attacks(from, pos.pieces()); break;
            case QUEEN:  attacks = queen_attacks(from, pos.pieces()); break;
            default: attacks = 0; break;
        }

        attacks &= targets;
        while (attacks) {
            Square to = pop_lsb(attacks);
            list.add(make_move(from, to));
        }
    }
}

void generate_king_moves(const Position& pos, MoveList& list, GenType type) {
    Color us = pos.side_to_move();
    Square ksq = pos.king_square(us);
    Bitboard targets;

    if (type == CAPTURES)
        targets = pos.pieces(~us);
    else if (type == QUIETS)
        targets = ~pos.pieces();
    else
        targets = ~pos.pieces(us);

    Bitboard attacks = king_attacks(ksq) & targets;
    while (attacks) {
        Square to = pop_lsb(attacks);
        list.add(make_move(ksq, to));
    }

    // Castling (only for ALL_MOVES and QUIETS)
    if (type == CAPTURES) return;
    if (pos.in_check()) return;

    Bitboard occ = pos.pieces();
    CastlingRight cr = pos.castling_rights();

    if (us == WHITE) {
        if ((cr & WHITE_OO) && !(occ & (square_bb(SQ_F1) | square_bb(SQ_G1)))) {
            if (!pos.is_attacked_by(~us, SQ_F1) && !pos.is_attacked_by(~us, SQ_G1)) {
                list.add(make_move(SQ_E1, SQ_G1, FLAG_CASTLING));
            }
        }
        if ((cr & WHITE_OOO) && !(occ & (square_bb(SQ_D1) | square_bb(SQ_C1) | square_bb(SQ_B1)))) {
            if (!pos.is_attacked_by(~us, SQ_D1) && !pos.is_attacked_by(~us, SQ_C1)) {
                list.add(make_move(SQ_E1, SQ_C1, FLAG_CASTLING));
            }
        }
    } else {
        if ((cr & BLACK_OO) && !(occ & (square_bb(SQ_F8) | square_bb(SQ_G8)))) {
            if (!pos.is_attacked_by(~us, SQ_F8) && !pos.is_attacked_by(~us, SQ_G8)) {
                list.add(make_move(SQ_E8, SQ_G8, FLAG_CASTLING));
            }
        }
        if ((cr & BLACK_OOO) && !(occ & (square_bb(SQ_D8) | square_bb(SQ_C8) | square_bb(SQ_B8)))) {
            if (!pos.is_attacked_by(~us, SQ_D8) && !pos.is_attacked_by(~us, SQ_C8)) {
                list.add(make_move(SQ_E8, SQ_C8, FLAG_CASTLING));
            }
        }
    }
}

} // anonymous namespace

void generate_moves(const Position& pos, MoveList& list, GenType type) {
    list.count = 0;
    generate_pawn_moves(pos, list, type);
    generate_piece_moves(pos, list, KNIGHT, type);
    generate_piece_moves(pos, list, BISHOP, type);
    generate_piece_moves(pos, list, ROOK, type);
    generate_piece_moves(pos, list, QUEEN, type);
    generate_king_moves(pos, list, type);
}

void generate_legal_moves(const Position& pos, MoveList& list) {
    MoveList pseudo;
    generate_moves(pos, pseudo, ALL_MOVES);

    list.count = 0;
    Position& pos_nc = const_cast<Position&>(pos);
    StateInfo st;

    for (int i = 0; i < pseudo.count; i++) {
        Move m = pseudo.moves[i];
        pos_nc.do_move(m, st);
        if (!pos_nc.is_attacked_by(pos_nc.side_to_move(), pos_nc.king_square(~pos_nc.side_to_move()))) {
            list.add(m);
        }
        pos_nc.undo_move(m);
    }
}
