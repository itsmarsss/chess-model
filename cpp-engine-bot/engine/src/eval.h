#pragma once

#include "position.h"

namespace Eval {
    int evaluate(const Position& pos);
    int hce(const Position& pos); // Hand-crafted eval only
}
