# NNUE Porting Guide - Files Required from Stockfish

This document identifies all the files needed to port Stockfish's NNUE (Efficiently Updatable Neural Networks) evaluation function to your chess hackathon project.

## Core NNUE Files (Required)

### 1. Core NNUE Headers and Implementation

**Location: `Stockfish/src/nnue/`**

#### Essential Core Files:
- `nnue_common.h` - Constants, types, and utility functions (LEB128 compression, endianness handling)
- `nnue_architecture.h` - Network architecture definition (layer structure, dimensions)
- `nnue_feature_transformer.h` - Feature transformation from board position to neural network input
- `nnue_accumulator.h` - Incremental accumulator for efficient evaluation updates
- `nnue_accumulator.cpp` - Implementation of accumulator operations
- `network.h` - Main network class definition
- `network.cpp` - Network implementation (loading, saving, evaluation)
- `nnue_misc.h` - Miscellaneous types and structures (EvalFile, NnueEvalTrace)
- `nnue_misc.cpp` - Implementation of misc functions
- `simd.h` - SIMD (vector instruction) abstractions for performance

### 2. Feature Sets

**Location: `Stockfish/src/nnue/features/`**

#### Required Feature Files:
- `half_ka_v2_hm.h` - HalfKAv2_hm feature set header (piece-square features)
- `half_ka_v2_hm.cpp` - HalfKAv2_hm feature set implementation
- `full_threats.h` - FullThreats feature set header (threat-based features)
- `full_threats.cpp` - FullThreats feature set implementation

### 3. Neural Network Layers

**Location: `Stockfish/src/nnue/layers/`**

#### Required Layer Files:
- `affine_transform.h` - Dense affine transformation layer
- `affine_transform_sparse_input.h` - Sparse input affine transformation layer
- `clipped_relu.h` - Clipped ReLU activation function
- `sqr_clipped_relu.h` - Squared clipped ReLU activation function

## Supporting Stockfish Files (Required Dependencies)

### 4. Core Stockfish Types and Utilities

**Location: `Stockfish/src/`**

#### Essential Dependencies:
- `types.h` - Core type definitions (Color, Piece, Square, Value, Move, etc.)
  - **Critical**: Defines `DirtyPiece`, `DirtyThreats`, `DirtyBoardData` used by NNUE
- `misc.h` - Utility functions (hash_combine, FixedString, ValueList, etc.)
- `bitboard.h` - Bitboard operations (required by features)
- `bitboard.cpp` - Bitboard implementation
- `position.h` - Position class interface (NNUE needs to query board state)
- `position.cpp` - Position class implementation

### 5. Evaluation Interface

**Location: `Stockfish/src/`**

- `evaluate.h` - Evaluation function interface
- `evaluate.cpp` - Evaluation function implementation (optional, but recommended for reference)

## File Dependency Graph

```
nnue_common.h
  └─> misc.h
  └─> types.h

nnue_architecture.h
  └─> nnue_common.h
  └─> features/half_ka_v2_hm.h
  └─> features/full_threats.h
  └─> layers/*.h

nnue_accumulator.h
  └─> types.h
  └─> nnue_architecture.h
  └─> nnue_common.h

nnue_feature_transformer.h
  └─> position.h
  └─> types.h
  └─> nnue_accumulator.h
  └─> nnue_architecture.h
  └─> nnue_common.h
  └─> simd.h

network.h
  └─> misc.h
  └─> types.h
  └─> nnue_accumulator.h
  └─> nnue_architecture.h
  └─> nnue_common.h
  └─> nnue_feature_transformer.h
  └─> nnue_misc.h

features/half_ka_v2_hm.h
  └─> misc.h
  └─> types.h
  └─> nnue_common.h

features/full_threats.h
  └─> misc.h
  └─> types.h
  └─> nnue_common.h
```

## Minimal Porting Checklist

### Phase 1: Core Infrastructure
- [ ] Port `types.h` - Ensure all chess types are defined
- [ ] Port `misc.h` - Utility functions (especially `hash_combine`, `ValueList`, `FixedString`)
- [ ] Port `bitboard.h` and `bitboard.cpp` - Bitboard operations
- [ ] Port `position.h` - Position class interface

### Phase 2: NNUE Core
- [ ] Port `nnue_common.h` - Constants and utilities
- [ ] Port `simd.h` - SIMD abstractions (or stub out if not using SIMD)
- [ ] Port all layer files (`layers/*.h`)
- [ ] Port `nnue_architecture.h`

### Phase 3: Features
- [ ] Port `features/half_ka_v2_hm.h` and `.cpp`
- [ ] Port `features/full_threats.h` and `.cpp`

### Phase 4: Accumulator and Transformer
- [ ] Port `nnue_accumulator.h` and `.cpp`
- [ ] Port `nnue_feature_transformer.h`

### Phase 5: Network
- [ ] Port `nnue_misc.h` and `.cpp`
- [ ] Port `network.h` and `.cpp`

## Key Integration Points

### 1. Position Class Requirements
Your `Position` class must provide:
- `piece_on(Square)` - Get piece at square
- `pieces()` - Get all pieces bitboard
- `pieces(Color)` - Get pieces for a color
- `side_to_move()` - Current player
- `count<PieceType>()` - Count pieces
- `non_pawn_material(Color)` - Material value
- Access to piece bitboards for feature extraction

### 2. Types Required from types.h
- `Color`, `Piece`, `PieceType`, `Square`
- `DirtyPiece`, `DirtyThreats`, `DirtyBoardData`
- `Value`, `Bitboard`, `Key`
- `Move` class

### 3. Evaluation Function Integration
The main evaluation function signature:
```cpp
Value evaluate(
    const NNUE::Networks& networks,
    const Position& pos,
    NNUE::AccumulatorStack& accumulators,
    NNUE::AccumulatorCaches& caches,
    int optimism
);
```

### 4. Move Making Integration
When making/undoing moves, you need to update the accumulator:
```cpp
// When making a move
accumulatorStack.push(dirtyBoardData);  // or accumulatorStack.push() and fill manually

// When undoing a move
accumulatorStack.pop();
```

## Optional but Recommended

- `incbin/incbin.h` - For embedding NNUE networks in binary (optional)
- `evaluate.cpp` - Reference implementation of evaluation blending

## Notes

1. **SIMD Support**: The `simd.h` file abstracts SIMD operations. If you're not using SIMD, you may need to stub out or provide scalar implementations.

2. **Network Files**: You'll need actual NNUE network weight files (`.nnue` files) to use the evaluation. These are separate from the code.

3. **Namespace**: Stockfish uses `Stockfish::Eval::NNUE` namespace. You may want to adapt this to your project's namespace.

4. **Position Interface**: The NNUE code expects a specific `Position` interface. You may need to create an adapter if your position representation differs.

5. **Thread Safety**: `AccumulatorStack` and `AccumulatorCaches` should be per-thread if you're doing parallel search.

## File Count Summary

- **Core NNUE files**: ~10 files
- **Feature files**: 4 files (2 headers + 2 implementations)
- **Layer files**: 4 header files
- **Supporting files**: 5-6 files (types, misc, bitboard, position)
- **Total**: ~23-24 files

## Next Steps

1. Start by porting the core types and utilities
2. Port the NNUE core files in dependency order
3. Integrate with your position representation
4. Test with a simple position evaluation
5. Integrate with your search algorithm

