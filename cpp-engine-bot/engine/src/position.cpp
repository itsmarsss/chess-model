#include "position.h"
#include <sstream>
#include <algorithm>

namespace Zobrist {
    uint64_t psq[PIECE_NB][SQUARE_NB];
    uint64_t castling[16];
    uint64_t enpassant[8];
    uint64_t side;

    static uint64_t rand64_state = 1070372ull;

    static uint64_t rand64() {
        rand64_state ^= rand64_state >> 12;
        rand64_state ^= rand64_state << 25;
        rand64_state ^= rand64_state >> 27;
        return rand64_state * 0x2545F4914F6CDD1DULL;
    }

    void init() {
        for (int p = 0; p < PIECE_NB; p++)
            for (int s = 0; s < SQUARE_NB; s++)
                psq[p][s] = rand64();

        for (int i = 0; i < 16; i++)
            castling[i] = rand64();

        for (int i = 0; i < 8; i++)
            enpassant[i] = rand64();

        side = rand64();
    }
}

Position::Position() {
    memset(board, 0, sizeof(board));
    memset(by_type, 0, sizeof(by_type));
    memset(by_color, 0, sizeof(by_color));
    side = WHITE;
    game_ply_ = 0;
    st = nullptr;
}

void Position::put_piece(Piece p, Square s) {
    board[s] = p;
    by_type[type_of(p)] |= square_bb(s);
    by_color[color_of(p)] |= square_bb(s);
}

void Position::remove_piece(Square s) {
    Piece p = board[s];
    by_type[type_of(p)] ^= square_bb(s);
    by_color[color_of(p)] ^= square_bb(s);
    board[s] = NO_PIECE;
}

void Position::move_piece(Square from, Square to) {
    Piece p = board[from];
    Bitboard move_bb = square_bb(from) | square_bb(to);
    by_type[type_of(p)] ^= move_bb;
    by_color[color_of(p)] ^= move_bb;
    board[from] = NO_PIECE;
    board[to] = p;
}

void Position::set_startpos(StateInfo& si) {
    set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", si);
}

void Position::set(const std::string& fen, StateInfo& si) {
    memset(board, 0, sizeof(board));
    memset(by_type, 0, sizeof(by_type));
    memset(by_color, 0, sizeof(by_color));
    st = &si;
    memset(st, 0, sizeof(StateInfo));
    st->en_passant = SQ_NONE;

    std::istringstream ss(fen);
    std::string token;

    // Piece placement
    ss >> token;
    int sq = 56;
    for (char c : token) {
        if (c == '/') {
            sq -= 16;
        } else if (c >= '1' && c <= '8') {
            sq += c - '0';
        } else {
            Color color = (c >= 'a') ? BLACK : WHITE;
            PieceType pt;
            char lower = c | 0x20;
            switch (lower) {
                case 'p': pt = PAWN; break;
                case 'n': pt = KNIGHT; break;
                case 'b': pt = BISHOP; break;
                case 'r': pt = ROOK; break;
                case 'q': pt = QUEEN; break;
                case 'k': pt = KING; break;
                default: pt = NO_PIECE_TYPE; break;
            }
            if (pt != NO_PIECE_TYPE) {
                put_piece(make_piece(color, pt), Square(sq));
            }
            sq++;
        }
    }

    // Side to move
    ss >> token;
    side = (token == "w") ? WHITE : BLACK;

    // Castling
    ss >> token;
    st->castling = NO_CASTLING;
    for (char c : token) {
        switch (c) {
            case 'K': st->castling |= WHITE_OO; break;
            case 'Q': st->castling |= WHITE_OOO; break;
            case 'k': st->castling |= BLACK_OO; break;
            case 'q': st->castling |= BLACK_OOO; break;
            default: break;
        }
    }

    // En passant
    ss >> token;
    if (token != "-" && token.length() == 2) {
        int f = token[0] - 'a';
        int r = token[1] - '1';
        st->en_passant = make_square(f, r);
    }

    // Halfmove clock and fullmove number
    int halfmove = 0, fullmove = 1;
    if (ss >> halfmove) st->halfmove_clock = halfmove;
    if (ss >> fullmove) game_ply_ = 2 * (fullmove - 1) + (side == BLACK ? 1 : 0);

    // Compute Zobrist key
    st->key = 0;
    for (int s = 0; s < 64; s++) {
        if (board[s] != NO_PIECE)
            st->key ^= Zobrist::psq[board[s]][s];
    }
    if (side == BLACK) st->key ^= Zobrist::side;
    st->key ^= Zobrist::castling[st->castling];
    if (st->en_passant != SQ_NONE)
        st->key ^= Zobrist::enpassant[file_of(st->en_passant)];

    st->previous = nullptr;
    st->plies_from_null = 0;
}

std::string Position::fen() const {
    std::string s;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            Piece p = board[make_square(f, r)];
            if (p == NO_PIECE) {
                empty++;
            } else {
                if (empty > 0) { s += char('0' + empty); empty = 0; }
                const char piece_chars[] = " PNBRQK  pnbrqk";
                s += piece_chars[p];
            }
        }
        if (empty > 0) s += char('0' + empty);
        if (r > 0) s += '/';
    }
    s += (side == WHITE) ? " w " : " b ";

    std::string castling_str;
    if (st->castling & WHITE_OO) castling_str += 'K';
    if (st->castling & WHITE_OOO) castling_str += 'Q';
    if (st->castling & BLACK_OO) castling_str += 'k';
    if (st->castling & BLACK_OOO) castling_str += 'q';
    s += castling_str.empty() ? "-" : castling_str;

    s += ' ';
    if (st->en_passant != SQ_NONE)
        s += square_to_string(st->en_passant);
    else
        s += '-';

    s += ' ' + std::to_string(st->halfmove_clock);
    s += ' ' + std::to_string(1 + (game_ply_ - (side == BLACK)) / 2);
    return s;
}

void Position::do_move(Move m, StateInfo& new_st) {
    memset(&new_st, 0, sizeof(StateInfo));
    new_st.previous = st;
    new_st.castling = st->castling;
    new_st.halfmove_clock = st->halfmove_clock + 1;
    new_st.en_passant = SQ_NONE;
    new_st.plies_from_null = st->plies_from_null + 1;
    new_st.dirty.num = 0;

    Square from = from_sq(m);
    Square to = to_sq(m);
    Piece pc = board[from];
    Piece captured = board[to];
    Color us = side;
    Color them = ~us;

    new_st.key = st->key ^ Zobrist::side;

    // Handle en passant capture
    if (is_en_passant(m)) {
        Square capsq = to + (us == WHITE ? SOUTH : NORTH);
        captured = board[capsq];
        new_st.dirty.piece[new_st.dirty.num] = captured;
        new_st.dirty.from[new_st.dirty.num] = capsq;
        new_st.dirty.to[new_st.dirty.num] = SQ_NONE;
        new_st.dirty.num++;
        new_st.key ^= Zobrist::psq[captured][capsq];
        remove_piece(capsq);
    }

    new_st.captured = captured;

    // Handle normal capture
    if (captured != NO_PIECE && !is_en_passant(m)) {
        new_st.dirty.piece[new_st.dirty.num] = captured;
        new_st.dirty.from[new_st.dirty.num] = to;
        new_st.dirty.to[new_st.dirty.num] = SQ_NONE;
        new_st.dirty.num++;
        new_st.key ^= Zobrist::psq[captured][to];
        remove_piece(to);
        new_st.halfmove_clock = 0;
    }

    // Update castling rights
    static const CastlingRight CastlingRightsMask[SQUARE_NB] = {
        CastlingRight(~WHITE_OOO), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(~(WHITE_OO | WHITE_OOO)), CastlingRight(15), CastlingRight(15), CastlingRight(~WHITE_OO),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(15), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(~BLACK_OOO), CastlingRight(15), CastlingRight(15), CastlingRight(15),
        CastlingRight(~(BLACK_OO | BLACK_OOO)), CastlingRight(15), CastlingRight(15), CastlingRight(~BLACK_OO)
    };

    new_st.key ^= Zobrist::castling[st->castling];
    new_st.castling &= CastlingRightsMask[from];
    new_st.castling &= CastlingRightsMask[to];
    new_st.key ^= Zobrist::castling[new_st.castling];

    // Move the piece
    new_st.key ^= Zobrist::psq[pc][from];

    if (is_castling(m)) {
        // Move king
        move_piece(from, to);
        new_st.key ^= Zobrist::psq[pc][to];

        // Move rook
        Square rook_from, rook_to;
        if (to > from) { // King-side
            rook_from = Square(to + 1); // h1 or h8
            rook_to = Square(to - 1);   // f1 or f8
        } else { // Queen-side
            rook_from = Square(to - 2); // a1 or a8
            rook_to = Square(to + 1);   // d1 or d8
        }
        Piece rook = board[rook_from];
        new_st.key ^= Zobrist::psq[rook][rook_from] ^ Zobrist::psq[rook][rook_to];
        move_piece(rook_from, rook_to);

        new_st.dirty.piece[new_st.dirty.num] = pc;
        new_st.dirty.from[new_st.dirty.num] = from;
        new_st.dirty.to[new_st.dirty.num] = to;
        new_st.dirty.num++;
        new_st.dirty.piece[new_st.dirty.num] = rook;
        new_st.dirty.from[new_st.dirty.num] = rook_from;
        new_st.dirty.to[new_st.dirty.num] = rook_to;
        new_st.dirty.num++;
    } else if (is_promotion(m)) {
        Piece promo_pc = make_piece(us, promo_type(m));
        remove_piece(from);
        put_piece(promo_pc, to);
        new_st.key ^= Zobrist::psq[promo_pc][to];

        new_st.dirty.piece[new_st.dirty.num] = pc;
        new_st.dirty.from[new_st.dirty.num] = from;
        new_st.dirty.to[new_st.dirty.num] = SQ_NONE;
        new_st.dirty.num++;
        new_st.dirty.piece[new_st.dirty.num] = promo_pc;
        new_st.dirty.from[new_st.dirty.num] = SQ_NONE;
        new_st.dirty.to[new_st.dirty.num] = to;
        new_st.dirty.num++;

        new_st.halfmove_clock = 0;
    } else {
        move_piece(from, to);
        new_st.key ^= Zobrist::psq[pc][to];

        new_st.dirty.piece[new_st.dirty.num] = pc;
        new_st.dirty.from[new_st.dirty.num] = from;
        new_st.dirty.to[new_st.dirty.num] = to;
        new_st.dirty.num++;
    }

    // Pawn specifics
    if (type_of(pc) == PAWN) {
        new_st.halfmove_clock = 0;
        // Double push: set en passant
        if (abs(rank_of(to) - rank_of(from)) == 2) {
            Square ep = Square((from + to) / 2);
            if (PawnAttacks[us][ep] & pieces(them, PAWN)) {
                new_st.en_passant = ep;
                new_st.key ^= Zobrist::enpassant[file_of(ep)];
            }
        }
    }

    // Remove old EP from key
    if (st->en_passant != SQ_NONE)
        new_st.key ^= Zobrist::enpassant[file_of(st->en_passant)];

    // Hmm, we need to be careful: the ep key was already toggled via new_st.key
    // Let me fix: we started with st->key ^ side, then removed old ep, added new ep
    // Actually let's recompute properly:
    // new_st.key was built incrementally above but let's fix the ep handling:
    // The key was XORed with old castling removed / new castling added
    // We need to also XOR out the old en_passant if it existed
    // This was done via the line above. The new en_passant was added inside the pawn block.

    side = them;
    game_ply_++;
    st = &new_st;
}

void Position::undo_move(Move m) {
    side = ~side;
    game_ply_--;

    Square from = from_sq(m);
    Square to = to_sq(m);
    Color us = side;

    if (is_castling(m)) {
        // Undo king move
        move_piece(to, from);
        // Undo rook move
        Square rook_from, rook_to;
        if (to > from) {
            rook_from = Square(to + 1);
            rook_to = Square(to - 1);
        } else {
            rook_from = Square(to - 2);
            rook_to = Square(to + 1);
        }
        move_piece(rook_to, rook_from);
    } else if (is_promotion(m)) {
        remove_piece(to);
        put_piece(make_piece(us, PAWN), from);
    } else {
        move_piece(to, from);
    }

    // Restore captured piece
    Piece captured = st->captured;
    if (captured != NO_PIECE) {
        Square capsq = to;
        if (is_en_passant(m))
            capsq = to + (us == WHITE ? SOUTH : NORTH);
        put_piece(captured, capsq);
    }

    st = st->previous;
}

void Position::do_null_move(StateInfo& new_st) {
    memset(&new_st, 0, sizeof(StateInfo));
    new_st.previous = st;
    new_st.key = st->key ^ Zobrist::side;
    new_st.castling = st->castling;
    new_st.halfmove_clock = st->halfmove_clock + 1;
    new_st.en_passant = SQ_NONE;
    new_st.plies_from_null = 0;
    new_st.captured = NO_PIECE;
    new_st.dirty.num = 0;

    if (st->en_passant != SQ_NONE)
        new_st.key ^= Zobrist::enpassant[file_of(st->en_passant)];

    side = ~side;
    game_ply_++;
    st = &new_st;
}

void Position::undo_null_move() {
    st = st->previous;
    side = ~side;
    game_ply_--;
}

Bitboard Position::attackers_to(Square s) const {
    return attackers_to(s, pieces());
}

Bitboard Position::attackers_to(Square s, Bitboard occupied) const {
    return (pawn_attacks(BLACK, s) & pieces(WHITE, PAWN))
         | (pawn_attacks(WHITE, s) & pieces(BLACK, PAWN))
         | (knight_attacks(s) & pieces(KNIGHT))
         | (bishop_attacks(s, occupied) & pieces(BISHOP, QUEEN))
         | (rook_attacks(s, occupied) & pieces(ROOK, QUEEN))
         | (king_attacks(s) & pieces(KING));
}

bool Position::is_attacked_by(Color c, Square s) const {
    Bitboard occ = pieces();
    if (pawn_attacks(~c, s) & pieces(c, PAWN)) return true;
    if (knight_attacks(s) & pieces(c, KNIGHT)) return true;
    if (bishop_attacks(s, occ) & pieces(c, BISHOP, QUEEN)) return true;
    if (rook_attacks(s, occ) & pieces(c, ROOK, QUEEN)) return true;
    if (king_attacks(s) & pieces(c, KING)) return true;
    return false;
}

bool Position::in_check() const {
    return is_attacked_by(~side, king_square(side));
}

Bitboard Position::checkers() const {
    return attackers_to(king_square(side)) & pieces(~side);
}

bool Position::is_legal(Move m) const {
    Color us = side;
    Square from = from_sq(m);
    Square ksq = king_square(us);

    if (is_en_passant(m)) {
        Square to = to_sq(m);
        Square capsq = to + (us == WHITE ? SOUTH : NORTH);
        Bitboard occupied = (pieces() ^ square_bb(from) ^ square_bb(capsq)) | square_bb(to);
        return !(bishop_attacks(ksq, occupied) & pieces(~us, BISHOP, QUEEN))
            && !(rook_attacks(ksq, occupied) & pieces(~us, ROOK, QUEEN));
    }

    if (is_castling(m)) {
        Square to = to_sq(m);
        Direction step = (to > from) ? EAST : WEST;
        for (Square s = from; s != to; s += step) {
            if (is_attacked_by(~us, s)) return false;
        }
        if (is_attacked_by(~us, to)) return false;
        return true;
    }

    if (type_of(board[from]) == KING) {
        return !is_attacked_by(~us, to_sq(m));
    }

    // If the piece is pinned, it can only move along the pin line
    Bitboard pinners = (bishop_attacks(ksq, 0) & pieces(~us, BISHOP, QUEEN))
                     | (rook_attacks(ksq, 0) & pieces(~us, ROOK, QUEEN));
    // simplified: not on a line with king, or moves along that line
    if (!(LineBB[ksq][from] & square_bb(from))) return true;
    return LineBB[ksq][from] & square_bb(to_sq(m));
}

bool Position::gives_check(Move m) const {
    Square to = to_sq(m);
    Square ksq = king_square(~side);
    Piece pc = board[from_sq(m)];
    PieceType pt = type_of(pc);

    if (is_promotion(m)) pt = promo_type(m);

    // Direct check
    Bitboard occ = pieces() ^ square_bb(from_sq(m));
    if (!is_castling(m)) occ |= square_bb(to);

    switch (pt) {
        case PAWN:   if (pawn_attacks(side, to) & square_bb(ksq)) return true; break;
        case KNIGHT: if (knight_attacks(to) & square_bb(ksq)) return true; break;
        case BISHOP: if (bishop_attacks(to, occ) & square_bb(ksq)) return true; break;
        case ROOK:   if (rook_attacks(to, occ) & square_bb(ksq)) return true; break;
        case QUEEN:  if (queen_attacks(to, occ) & square_bb(ksq)) return true; break;
        default: break;
    }

    // Discovered check
    if (LineBB[ksq][from_sq(m)] & square_bb(ksq)) {
        if (!(LineBB[ksq][from_sq(m)] & square_bb(to))) {
            if (bishop_attacks(ksq, occ) & pieces(side, BISHOP, QUEEN)) return true;
            if (rook_attacks(ksq, occ) & pieces(side, ROOK, QUEEN)) return true;
        }
    }

    return false;
}

bool Position::has_non_pawn_material(Color c) const {
    return pieces(c, KNIGHT) | pieces(c, BISHOP) | pieces(c, ROOK) | pieces(c, QUEEN);
}

bool Position::is_draw() const {
    if (st->halfmove_clock >= 100) return true;

    // Repetition detection
    int end = std::min(st->halfmove_clock, st->plies_from_null);
    StateInfo* stp = st->previous;
    int cnt = 0;
    for (int i = 2; i <= end; i += 2) {
        if (stp && stp->previous) {
            stp = stp->previous->previous;
            if (stp && stp->key == st->key) {
                cnt++;
                if (cnt >= 2) return true;
            }
        } else break;
    }
    return false;
}
