#!/bin/bash
# Script to port NNUE files from Stockfish to chesshacks project

# Create directories
mkdir -p nnue/features nnue/layers

# Copy foundation files (updating include paths)
echo "Porting foundation files..."

# Copy types.h (no changes needed for includes)
cp Stockfish/src/types.h types.h

# Copy misc.h (no changes needed for includes)  
cp Stockfish/src/misc.h misc.h

# Copy bitboard files
cp Stockfish/src/bitboard.h bitboard.h
cp Stockfish/src/bitboard.cpp bitboard.cpp

# Copy NNUE core files
echo "Porting NNUE core files..."
cp Stockfish/src/nnue/nnue_common.h nnue/nnue_common.h
cp Stockfish/src/nnue/nnue_architecture.h nnue/nnue_architecture.h
cp Stockfish/src/nnue/nnue_feature_transformer.h nnue/nnue_feature_transformer.h
cp Stockfish/src/nnue/nnue_accumulator.h nnue/nnue_accumulator.h
cp Stockfish/src/nnue/nnue_accumulator.cpp nnue/nnue_accumulator.cpp
cp Stockfish/src/nnue/network.h nnue/network.h
cp Stockfish/src/nnue/network.cpp nnue/network.cpp
cp Stockfish/src/nnue/nnue_misc.h nnue/nnue_misc.h
cp Stockfish/src/nnue/nnue_misc.cpp nnue/nnue_misc.cpp
cp Stockfish/src/nnue/simd.h nnue/simd.h

# Copy layer files
echo "Porting layer files..."
cp Stockfish/src/nnue/layers/*.h nnue/layers/

# Copy feature files
echo "Porting feature files..."
cp Stockfish/src/nnue/features/*.h nnue/features/
cp Stockfish/src/nnue/features/*.cpp nnue/features/

echo "Files copied. Now updating include paths..."

# Update include paths in all files
find nnue -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i '' 's|#include "../|#include "|g' {} \;
find nnue -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i '' 's|#include "../../|#include "../|g' {} \;
find nnue -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i '' 's|#include "\.\./bitboard\.h"|#include "bitboard.h"|g' {} \;
find nnue -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i '' 's|#include "\.\./misc\.h"|#include "misc.h"|g' {} \;
find nnue -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i '' 's|#include "\.\./types\.h"|#include "types.h"|g' {} \;
find nnue -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i '' 's|#include "\.\./position\.h"|#include "position.h"|g' {} \;
find nnue -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i '' 's|#include "\.\./evaluate\.h"|#include "evaluate.h"|g' {} \;
find nnue -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i '' 's|#include "\.\./incbin/|#include "incbin/|g' {} \;
find nnue -type f \( -name "*.h" -o -name "*.cpp" \) -exec sed -i '' 's|#include "\.\./uci\.h"|#include "uci.h"|g' {} \;

# Update includes in bitboard.cpp
sed -i '' 's|#include "misc\.h"|#include "misc.h"|g' bitboard.cpp

echo "Done! Files ported and include paths updated."

