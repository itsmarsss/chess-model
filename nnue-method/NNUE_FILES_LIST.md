# NNUE Files to Port - Quick Reference

## Core NNUE Directory (`src/nnue/`)

### Headers (Required)
- `nnue_common.h`
- `nnue_architecture.h`
- `nnue_feature_transformer.h`
- `nnue_accumulator.h`
- `network.h`
- `nnue_misc.h`
- `simd.h`

### Implementations (Required)
- `nnue_accumulator.cpp`
- `network.cpp`
- `nnue_misc.cpp`

## Features Directory (`src/nnue/features/`)

### HalfKAv2 Feature Set
- `half_ka_v2_hm.h`
- `half_ka_v2_hm.cpp`

### FullThreats Feature Set
- `full_threats.h`
- `full_threats.cpp`

## Layers Directory (`src/nnue/layers/`)

- `affine_transform.h`
- `affine_transform_sparse_input.h`
- `clipped_relu.h`
- `sqr_clipped_relu.h`

## Supporting Files (`src/`)

### Critical Dependencies
- `types.h` ⚠️ **REQUIRED** - Defines DirtyPiece, DirtyThreats, etc.
- `misc.h` ⚠️ **REQUIRED** - Utility functions
- `bitboard.h` ⚠️ **REQUIRED** - Bitboard operations
- `bitboard.cpp` ⚠️ **REQUIRED**
- `position.h` ⚠️ **REQUIRED** - Position interface
- `position.cpp` ⚠️ **REQUIRED**

### Optional Reference
- `evaluate.h` - Evaluation interface
- `evaluate.cpp` - Evaluation implementation

## Total File Count

- **NNUE Core**: 7 headers + 3 implementations = 10 files
- **Features**: 2 headers + 2 implementations = 4 files
- **Layers**: 4 headers = 4 files
- **Supporting**: 3 headers + 3 implementations = 6 files
- **Total**: **24 files minimum**

## Priority Order for Porting

1. **Foundation** (Must port first):
   - `types.h`
   - `misc.h`
   - `bitboard.h` + `bitboard.cpp`
   - `position.h` + `position.cpp`

2. **NNUE Base**:
   - `nnue_common.h`
   - `simd.h`
   - All `layers/*.h`

3. **Features**:
   - `features/half_ka_v2_hm.h` + `.cpp`
   - `features/full_threats.h` + `.cpp`

4. **Core NNUE**:
   - `nnue_architecture.h`
   - `nnue_accumulator.h` + `.cpp`
   - `nnue_feature_transformer.h`

5. **Network**:
   - `nnue_misc.h` + `.cpp`
   - `network.h` + `.cpp`

