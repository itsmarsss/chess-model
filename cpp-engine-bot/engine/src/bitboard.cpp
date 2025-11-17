#include "bitboard.h"
#include <cstring>

Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard KnightAttacks[SQUARE_NB];
Bitboard KingAttacks[SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];

Magic BishopMagics[SQUARE_NB];
Magic RookMagics[SQUARE_NB];

static Bitboard BishopTable[0x1480];
static Bitboard RookTable[0x19000];

namespace {

Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occupied) {
    Bitboard attacks = 0;
    const Direction bishop_dirs[] = {NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST};
    const Direction rook_dirs[] = {NORTH, SOUTH, EAST, WEST};
    const Direction* dirs = (pt == BISHOP) ? bishop_dirs : rook_dirs;

    for (int i = 0; i < 4; i++) {
        Square s = sq;
        while (true) {
            int f = file_of(s), r = rank_of(s);
            s += dirs[i];
            int nf = file_of(s), nr = rank_of(s);
            if (int(s) < 0 || int(s) >= 64) break;
            if (abs(nf - f) > 1 || abs(nr - r) > 1) break;
            f = nf; r = nr;
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
        }
    }
    return attacks;
}

// Fancy magic numbers (pre-computed, known good)
// Using the same magics as many open-source engines
constexpr Bitboard RookMagicNumbers[SQUARE_NB] = {
    0x8a80104000800020ULL, 0x140002000100040ULL, 0x2801880a0017001ULL,
    0x100081001000420ULL, 0x200020010080420ULL, 0x3001c0002010008ULL,
    0x8480008002000100ULL, 0x2080088004402900ULL, 0x800098204000ULL,
    0x2024401000200040ULL, 0x100802000801000ULL, 0x120800800801000ULL,
    0x208808088000400ULL, 0x2802200800400ULL, 0x2200800100020080ULL,
    0x801000060821100ULL, 0x80044006422000ULL, 0x100808020004000ULL,
    0x12108a0010204200ULL, 0x140848010000802ULL, 0x481828014002800ULL,
    0x8094004002004100ULL, 0x4010040010010802ULL, 0x20008806104ULL,
    0x100400080208000ULL, 0x2040002120081000ULL, 0x21200680100081ULL,
    0x20100080080080ULL, 0x2000a00200410ULL, 0x20080800400ULL,
    0x80088400100102ULL, 0x80004600042881ULL, 0x4040008040800020ULL,
    0x440003000200801ULL, 0x4200011004500ULL, 0x188020010100100ULL,
    0x14800401802800ULL, 0x2080040080800200ULL, 0x124080204001001ULL,
    0x200046502000484ULL, 0x480400080088020ULL, 0x1000422010034000ULL,
    0x30200100110040ULL, 0x100021010009ULL, 0x2002080100110004ULL,
    0x202008004008002ULL, 0x20020004010100ULL, 0x2048440040820001ULL,
    0x101002200408200ULL, 0x40802000401080ULL, 0x4008142004410100ULL,
    0x2060820c0120200ULL, 0x1001004080100ULL, 0x20c020080040080ULL,
    0x2935610830022400ULL, 0x44440041009200ULL, 0x280001040802101ULL,
    0x2100190040002085ULL, 0x80c0084100102001ULL, 0x4024081001000421ULL,
    0x20030a0244872ULL, 0x12001008414402ULL, 0x2006104900a0804ULL,
    0x1004081002402ULL
};

constexpr Bitboard BishopMagicNumbers[SQUARE_NB] = {
    0x40040844404084ULL, 0x2004208a004208ULL, 0x10190041080202ULL,
    0x108060845042010ULL, 0x581104180800210ULL, 0x2112080446200010ULL,
    0x1080820820060210ULL, 0x3c0808410220200ULL, 0x4050404440404ULL,
    0x21001420088ULL, 0x24d0080801082102ULL, 0x1020a0a020400ULL,
    0x40308200402ULL, 0x4011002100800ULL, 0x401484104104005ULL,
    0x801010402020200ULL, 0x400210c3880100ULL, 0x404022024108200ULL,
    0x810018200204102ULL, 0x4002801a02003ULL, 0x85040820080400ULL,
    0x810102c808880400ULL, 0xe900410884800ULL, 0x8002020480840102ULL,
    0x220200865090201ULL, 0x2010100a02021202ULL, 0x152048408022401ULL,
    0x20080002081110ULL, 0x4001001021004000ULL, 0x800040400a011002ULL,
    0xe4004081011002ULL, 0x1c004001012080ULL, 0x8004200962a00220ULL,
    0x8422100208500202ULL, 0x2000402200300c08ULL, 0x8646020080080080ULL,
    0x80020a0200100808ULL, 0x2010004880111000ULL, 0x623000a080011400ULL,
    0x42008c0340209202ULL, 0x209188240001000ULL, 0x400408a884001800ULL,
    0x110400a6080400ULL, 0x1840060a44020800ULL, 0x90080104000041ULL,
    0x201011000808101ULL, 0x1a2208080504f080ULL, 0x8012020600211212ULL,
    0x500861011240000ULL, 0x180806108200800ULL, 0x4000020e01040044ULL,
    0x300000261044000aULL, 0x802241102020002ULL, 0x20906061210001ULL,
    0x5a84841004010310ULL, 0x4010801011c04ULL, 0xa010109502200ULL,
    0x4a02012000ULL, 0x500201010098b028ULL, 0x8040002811040900ULL,
    0x28000010020204ULL, 0x6000020202d0240ULL, 0x8918844842082200ULL,
    0x4010011029020020ULL
};

constexpr int RookShifts[SQUARE_NB] = {
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    52, 53, 53, 53, 53, 53, 53, 52
};

constexpr int BishopShifts[SQUARE_NB] = {
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
};

void init_magics(PieceType pt, Magic magics[], Bitboard table[],
                 const Bitboard magic_numbers[], const int shifts[]) {
    Bitboard* current = table;

    for (int sq = 0; sq < 64; sq++) {
        Magic& m = magics[sq];

        // Compute mask: relevant occupancy squares (not edges)
        Bitboard edges = ((Rank1BB | Rank8BB) & ~(Rank1BB << (rank_of(Square(sq)) * 8))) |
                         ((FileABB | FileHBB) & ~(FileABB << file_of(Square(sq))));
        m.mask = sliding_attack(pt, Square(sq), 0) & ~edges;
        m.magic = magic_numbers[sq];
        m.shift = shifts[sq];
        m.attacks = current;

        // Enumerate all subsets of the mask
        Bitboard occ = 0;
        int size = 0;
        do {
            m.attacks[m.index(occ)] = sliding_attack(pt, Square(sq), occ);
            size++;
            occ = (occ - m.mask) & m.mask;
        } while (occ);

        current += size;
    }
}

} // namespace

void bitboard_init() {
    // Pawn attacks
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b = square_bb(Square(sq));
        PawnAttacks[WHITE][sq] = shift_ne(b) | shift_nw(b);
        PawnAttacks[BLACK][sq] = shift_se(b) | shift_sw(b);
    }

    // Knight attacks
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b = square_bb(Square(sq));
        KnightAttacks[sq] = 0;
        // << shifts (moving to higher squares)
        KnightAttacks[sq] |= (b & ~FileHBB) << 17;               // N2E: rank+2, file+1
        KnightAttacks[sq] |= (b & ~FileABB) << 15;               // N2W: rank+2, file-1
        KnightAttacks[sq] |= (b & ~(FileGBB | FileHBB)) << 10;   // NE2: rank+1, file+2
        KnightAttacks[sq] |= (b & ~(FileABB | FileBBB)) << 6;    // NW2: rank+1, file-2
        // >> shifts (moving to lower squares)
        KnightAttacks[sq] |= (b & ~(FileGBB | FileHBB)) >> 6;    // SE2: rank-1, file+2
        KnightAttacks[sq] |= (b & ~(FileABB | FileBBB)) >> 10;   // SW2: rank-1, file-2
        KnightAttacks[sq] |= (b & ~FileHBB) >> 15;               // S2E: rank-2, file+1
        KnightAttacks[sq] |= (b & ~FileABB) >> 17;               // S2W: rank-2, file-1
    }

    // King attacks
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b = square_bb(Square(sq));
        KingAttacks[sq] = 0;
        KingAttacks[sq] |= shift_north(b) | shift_south(b);
        KingAttacks[sq] |= shift_east(b) | shift_west(b);
        KingAttacks[sq] |= shift_ne(b) | shift_nw(b);
        KingAttacks[sq] |= shift_se(b) | shift_sw(b);
    }

    // Magic bitboards
    init_magics(ROOK, RookMagics, RookTable, RookMagicNumbers, RookShifts);
    init_magics(BISHOP, BishopMagics, BishopTable, BishopMagicNumbers, BishopShifts);

    // Between and Line bitboards
    memset(BetweenBB, 0, sizeof(BetweenBB));
    memset(LineBB, 0, sizeof(LineBB));

    for (int s1 = 0; s1 < 64; s1++) {
        for (int s2 = 0; s2 < 64; s2++) {
            if (s1 == s2) continue;

            if (bishop_attacks(Square(s1), 0) & square_bb(Square(s2))) {
                LineBB[s1][s2] = (bishop_attacks(Square(s1), 0) & bishop_attacks(Square(s2), 0))
                                 | square_bb(Square(s1)) | square_bb(Square(s2));
                BetweenBB[s1][s2] = bishop_attacks(Square(s1), square_bb(Square(s2)))
                                    & bishop_attacks(Square(s2), square_bb(Square(s1)));
            }
            if (rook_attacks(Square(s1), 0) & square_bb(Square(s2))) {
                LineBB[s1][s2] = (rook_attacks(Square(s1), 0) & rook_attacks(Square(s2), 0))
                                 | square_bb(Square(s1)) | square_bb(Square(s2));
                BetweenBB[s1][s2] = rook_attacks(Square(s1), square_bb(Square(s2)))
                                    & rook_attacks(Square(s2), square_bb(Square(s1)));
            }
        }
    }
}
