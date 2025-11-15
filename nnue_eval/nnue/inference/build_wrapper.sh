#!/bin/bash
# Build NNUE wrapper from Stockfish source

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFERENCE_DIR="$SCRIPT_DIR"
STOCKFISH_DIR="$INFERENCE_DIR/Stockfish"
WRAPPER_NAME="nnue_wrapper_stockfish"

echo "Building NNUE wrapper from Stockfish..."
echo "Stockfish directory: $STOCKFISH_DIR"

# Check if Stockfish directory exists
if [ ! -d "$STOCKFISH_DIR" ]; then
    echo "Error: Stockfish directory not found at $STOCKFISH_DIR"
    echo "Please clone Stockfish first:"
    echo "  cd $INFERENCE_DIR"
    echo "  git clone https://github.com/official-stockfish/Stockfish.git"
    exit 1
fi

# Create wrapper source file
cat > "$INFERENCE_DIR/nnue_wrapper.cpp" << 'EOF'
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
EOF

echo "Wrapper source created at $INFERENCE_DIR/nnue_wrapper.cpp"

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Building for: $OS on $ARCH"

# Build command
if [[ "$OS" == "Darwin" ]]; then
    # macOS
    EXT="dylib"
    EXTRA_FLAGS="-undefined dynamic_lookup"
else
    # Linux
    EXT="so"
    EXTRA_FLAGS=""
fi

# Compile
echo "Compiling NNUE wrapper..."

g++ -std=c++17 -O3 -fPIC -shared \
    -I"$STOCKFISH_DIR/src" \
    -DUSE_PTHREADS \
    -DNNUE_EMBEDDING_OFF \
    "$INFERENCE_DIR/nnue_wrapper.cpp" \
    "$STOCKFISH_DIR/src/bitboard.cpp" \
    "$STOCKFISH_DIR/src/position.cpp" \
    "$STOCKFISH_DIR/src/misc.cpp" \
    "$STOCKFISH_DIR/src/uci.cpp" \
    "$STOCKFISH_DIR/src/ucioption.cpp" \
    "$STOCKFISH_DIR/src/evaluate.cpp" \
    "$STOCKFISH_DIR/src/nnue/network.cpp" \
    "$STOCKFISH_DIR/src/nnue/nnue_misc.cpp" \
    "$STOCKFISH_DIR/src/nnue/nnue_accumulator.cpp" \
    "$STOCKFISH_DIR/src/nnue/features/half_ka_v2_hm.cpp" \
    "$STOCKFISH_DIR/src/thread.cpp" \
    "$STOCKFISH_DIR/src/tune.cpp" \
    -o "$INFERENCE_DIR/${WRAPPER_NAME}.${EXT}" \
    $EXTRA_FLAGS \
    -pthread

if [ $? -eq 0 ]; then
    echo "✓ Successfully built: $INFERENCE_DIR/${WRAPPER_NAME}.${EXT}"
    
    # Create symlink without _stockfish suffix for compatibility
    ln -sf "${WRAPPER_NAME}.${EXT}" "$INFERENCE_DIR/nnue_wrapper.${EXT}"
    echo "✓ Created symlink: $INFERENCE_DIR/nnue_wrapper.${EXT}"
    
    # Clean up wrapper source
    rm -f "$INFERENCE_DIR/nnue_wrapper.cpp"
    
    echo ""
    echo "NNUE wrapper built successfully!"
    echo "Library: $INFERENCE_DIR/${WRAPPER_NAME}.${EXT}"
else
    echo "✗ Build failed"
    exit 1
fi
