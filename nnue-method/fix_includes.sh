#!/bin/bash
# Fix include paths in all NNUE files

cd "$(dirname "$0")"

# Fix files in nnue/ directory - need ../ for root files
for file in nnue/*.h nnue/*.cpp; do
    if [ -f "$file" ]; then
        sed -i '' 's|#include "misc\.h"|#include "../misc.h"|g' "$file"
        sed -i '' 's|#include "types\.h"|#include "../types.h"|g' "$file"
        sed -i '' 's|#include "bitboard\.h"|#include "../bitboard.h"|g' "$file"
        sed -i '' 's|#include "position\.h"|#include "../position.h"|g' "$file"
        sed -i '' 's|#include "evaluate\.h"|#include "../evaluate.h"|g' "$file"
        sed -i '' 's|#include "uci\.h"|#include "../uci.h"|g' "$file"
        # Fix nnue includes within nnue directory
        sed -i '' 's|#include "nnue_common\.h"|#include "nnue_common.h"|g' "$file"
        sed -i '' 's|#include "simd\.h"|#include "simd.h"|g' "$file"
    fi
done

# Fix files in nnue/features/ directory - need ../../ for root, ../ for nnue
for file in nnue/features/*.h nnue/features/*.cpp; do
    if [ -f "$file" ]; then
        sed -i '' 's|#include "misc\.h"|#include "../../misc.h"|g' "$file"
        sed -i '' 's|#include "types\.h"|#include "../../types.h"|g' "$file"
        sed -i '' 's|#include "bitboard\.h"|#include "../../bitboard.h"|g' "$file"
        sed -i '' 's|#include "position\.h"|#include "../../position.h"|g' "$file"
        sed -i '' 's|#include "nnue_common\.h"|#include "../nnue_common.h"|g' "$file"
        sed -i '' 's|#include "\.\./nnue_common\.h"|#include "../nnue_common.h"|g' "$file"
    fi
done

# Fix files in nnue/layers/ directory - need ../../ for root, ../ for nnue
for file in nnue/layers/*.h; do
    if [ -f "$file" ]; then
        sed -i '' 's|#include "bitboard\.h"|#include "../../bitboard.h"|g' "$file"
        sed -i '' 's|#include "nnue_common\.h"|#include "../nnue_common.h"|g' "$file"
        sed -i '' 's|#include "simd\.h"|#include "../simd.h"|g' "$file"
    fi
done

# Fix simd.h specifically
sed -i '' 's|#include "\.\./types\.h"|#include "../types.h"|g' nnue/simd.h
sed -i '' 's|#include "nnue_common\.h"|#include "nnue_common.h"|g' nnue/simd.h

# Fix nnue_architecture.h includes
sed -i '' 's|#include "features/|#include "features/|g' nnue/nnue_architecture.h
sed -i '' 's|#include "layers/|#include "layers/|g' nnue/nnue_architecture.h

# Fix nnue_feature_transformer.h
sed -i '' 's|#include "\.\./position\.h"|#include "../position.h"|g' nnue/nnue_feature_transformer.h
sed -i '' 's|#include "\.\./types\.h"|#include "../types.h"|g' nnue/nnue_feature_transformer.h

# Fix nnue_accumulator includes
sed -i '' 's|#include "\.\./bitboard\.h"|#include "../bitboard.h"|g' nnue/nnue_accumulator.cpp
sed -i '' 's|#include "\.\./misc\.h"|#include "../misc.h"|g' nnue/nnue_accumulator.cpp
sed -i '' 's|#include "\.\./position\.h"|#include "../position.h"|g' nnue/nnue_accumulator.cpp
sed -i '' 's|#include "\.\./types\.h"|#include "../types.h"|g' nnue/nnue_accumulator.cpp
sed -i '' 's|#include "features/|#include "features/|g' nnue/nnue_accumulator.cpp

# Fix network.cpp includes
sed -i '' 's|#include "\.\./incbin/|#include "incbin/|g' nnue/network.cpp
sed -i '' 's|#include "\.\./evaluate\.h"|#include "../evaluate.h"|g' nnue/network.cpp
sed -i '' 's|#include "\.\./misc\.h"|#include "../misc.h"|g' nnue/network.cpp
sed -i '' 's|#include "\.\./position\.h"|#include "../position.h"|g' nnue/network.cpp
sed -i '' 's|#include "\.\./types\.h"|#include "../types.h"|g' nnue/network.cpp

# Fix nnue_misc.cpp includes
sed -i '' 's|#include "\.\./position\.h"|#include "../position.h"|g' nnue/nnue_misc.cpp
sed -i '' 's|#include "\.\./types\.h"|#include "../types.h"|g' nnue/nnue_misc.cpp
sed -i '' 's|#include "\.\./uci\.h"|#include "../uci.h"|g' nnue/nnue_misc.cpp

# Fix feature .cpp files
sed -i '' 's|#include "\.\./\.\./bitboard\.h"|#include "../../bitboard.h"|g' nnue/features/*.cpp
sed -i '' 's|#include "\.\./\.\./misc\.h"|#include "../../misc.h"|g' nnue/features/*.cpp
sed -i '' 's|#include "\.\./\.\./position\.h"|#include "../../position.h"|g' nnue/features/*.cpp
sed -i '' 's|#include "\.\./\.\./types\.h"|#include "../../types.h"|g' nnue/features/*.cpp
sed -i '' 's|#include "\.\./nnue_common\.h"|#include "../nnue_common.h"|g' nnue/features/*.cpp

echo "Include paths fixed!"

