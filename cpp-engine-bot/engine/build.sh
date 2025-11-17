#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

CXX="${CXX:-clang++}"
CXXFLAGS="-std=c++17 -O3 -march=native -DNDEBUG -flto"

# Detect if cmake is available
if command -v cmake &> /dev/null; then
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
else
    # Direct compilation
    SOURCES=(
        "$SCRIPT_DIR/src/main.cpp"
        "$SCRIPT_DIR/src/bitboard.cpp"
        "$SCRIPT_DIR/src/position.cpp"
        "$SCRIPT_DIR/src/movegen.cpp"
        "$SCRIPT_DIR/src/search.cpp"
        "$SCRIPT_DIR/src/eval.cpp"
        "$SCRIPT_DIR/src/tt.cpp"
        "$SCRIPT_DIR/src/movepick.cpp"
        "$SCRIPT_DIR/src/uci.cpp"
        "$SCRIPT_DIR/src/nnue.cpp"
    )
    echo "Compiling with $CXX..."
    $CXX $CXXFLAGS -I"$SCRIPT_DIR/src" "${SOURCES[@]}" -o "$BUILD_DIR/chess_engine"
fi

echo ""
echo "Engine built successfully: $BUILD_DIR/chess_engine"
