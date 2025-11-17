#include "bitboard.h"
#include "position.h"
#include "search.h"
#include "uci.h"

int main() {
    bitboard_init();
    Zobrist::init();
    Search::init();
    UCI::loop();
    return 0;
}
