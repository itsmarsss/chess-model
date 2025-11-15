# NNUE Integration Guide

## Quick Start

All NNUE files have been successfully ported from Stockfish. Here's how to integrate them into your project.

## File Structure

```
chesshacks/
├── types.h                    ✅ Ported
├── misc.h                     ✅ Ported
├── bitboard.h                 ✅ Ported
├── bitboard.cpp               ✅ Ported
├── position.h                 ⚠️  YOU NEED TO PROVIDE THIS
├── nnue/
│   ├── nnue_common.h          ✅ Ported
│   ├── nnue_architecture.h    ✅ Ported
│   ├── nnue_feature_transformer.h  ✅ Ported
│   ├── nnue_accumulator.h     ✅ Ported
│   ├── nnue_accumulator.cpp   ✅ Ported
│   ├── network.h              ✅ Ported
│   ├── network.cpp            ✅ Ported
│   ├── nnue_misc.h            ✅ Ported
│   ├── nnue_misc.cpp          ✅ Ported
│   ├── simd.h                 ✅ Ported
│   ├── features/
│   │   ├── half_ka_v2_hm.h    ✅ Ported
│   │   ├── half_ka_v2_hm.cpp  ✅ Ported
│   │   ├── full_threats.h     ✅ Ported
│   │   └── full_threats.cpp   ✅ Ported
│   └── layers/
│       ├── affine_transform.h ✅ Ported
│       ├── affine_transform_sparse_input.h ✅ Ported
│       ├── clipped_relu.h     ✅ Ported
│       └── sqr_clipped_relu.h ✅ Ported
```

## Step 1: Implement Position Interface

Your `position.h` must implement the interface documented in `position_interface_requirements.h`.

**Minimum Required Methods:**
- `piece_on(Square s) const`
- `pieces() const`
- `pieces(Color c) const`
- `pieces(Color c, PieceTypes... pts) const` (template)
- `side_to_move() const`
- `count<PieceType>() const` (template)
- `count<PieceType>(Color c) const` (template)
- `square<PieceType>(Color c) const` (template)
- `non_pawn_material(Color c) const`
- `rule50_count() const`

See `position_interface_requirements.h` for detailed documentation.

## Step 2: Initialize Required Components

```cpp
#include "bitboard.h"
#include "nnue/features/full_threats.h"
#include "nnue/network.h"

// At program startup:
void initialize_nnue() {
    // Initialize bitboard tables (REQUIRED)
    Stockfish::Bitboards::init();
    
    // Initialize threat feature offsets (REQUIRED for FullThreats)
    Stockfish::Eval::NNUE::Features::init_threat_offsets();
}
```

## Step 3: Load NNUE Network

```cpp
#include "nnue/network.h"

using namespace Stockfish::Eval::NNUE;

// Create network instances
auto networkBig = std::make_unique<NetworkBig>(
    EvalFile{FixedString<256>("nn-49c1193b131c.nnue")}, 
    EmbeddedNNUEType::BIG
);

auto networkSmall = std::make_unique<NetworkSmall>(
    EvalFile{FixedString<256>("nn-37f18f62d772.nnue")}, 
    EmbeddedNNUEType::SMALL
);

// Create Networks structure
Networks networks(std::move(networkBig), std::move(networkSmall));

// Load the network files
networks.big.load(".", "");    // Load from current directory
networks.small.load(".", "");
```

## Step 4: Use NNUE Evaluation

```cpp
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"

using namespace Stockfish::Eval::NNUE;

// Create per-thread accumulator structures
AccumulatorStack accumulatorStack;
AccumulatorCaches accumulatorCaches(networks);

// Evaluate a position
Value evaluate_position(const Position& pos, 
                        const Networks& networks,
                        AccumulatorStack& accumulators,
                        AccumulatorCaches& caches) {
    // Reset accumulators for new position
    accumulators.reset();
    
    // Evaluate using the networks
    auto [psqt, positional] = networks.big.evaluate(
        pos, accumulators, caches.big
    );
    
    // Combine PSQT and positional scores
    Value nnue = (125 * psqt + 131 * positional) / 128;
    
    return nnue;
}
```

## Step 5: Incremental Updates (Optional but Recommended)

For better performance, update accumulators incrementally when making moves:

```cpp
// When making a move
DirtyBoardData dirtyData;
// ... populate dirtyData with move information ...
accumulatorStack.push(dirtyData);

// When undoing a move
accumulatorStack.pop();
```

## Compilation

### Required Compiler Flags

```bash
# For SIMD support (choose based on your CPU)
-DUSE_AVX2      # For AVX2 support
-DUSE_SSE41     # For SSE4.1 support
-DUSE_SSSE3     # For SSSE3 support
-DUSE_SSE2      # For SSE2 support (minimum)
-DUSE_NEON      # For ARM NEON support

# Optional optimizations
-DUSE_POPCNT    # For popcount instruction
-DUSE_PEXT      # For pext instruction
-DNDEBUG        # Disable debug mode
```

### Example Compilation

```bash
g++ -std=c++17 -O3 -DUSE_AVX2 -DNDEBUG \
    -I. \
    bitboard.cpp \
    nnue/nnue_accumulator.cpp \
    nnue/network.cpp \
    nnue/nnue_misc.cpp \
    nnue/features/half_ka_v2_hm.cpp \
    nnue/features/full_threats.cpp \
    your_code.cpp \
    -o your_program
```

## Network Files

You'll need to download the NNUE network files:
- `nn-49c1193b131c.nnue` (big network)
- `nn-37f18f62d772.nnue` (small network)

These can be downloaded from the Stockfish repository or official releases.

## Troubleshooting

### Include Path Issues
All include paths have been updated. If you see include errors:
1. Ensure `position.h` is in the root directory
2. Check that all `nnue/` subdirectory includes use correct relative paths

### Missing Position Methods
If you get linker errors about missing Position methods, implement all methods listed in `position_interface_requirements.h`.

### SIMD Not Working
- Ensure you're using the correct `-DUSE_*` flag for your CPU
- Check that your compiler supports the SIMD instruction set
- You can compile without SIMD (slower but will work)

## Performance Tips

1. **Use Incremental Updates**: Update accumulators incrementally instead of full refresh
2. **Per-Thread Accumulators**: Use separate `AccumulatorStack` and `AccumulatorCaches` per thread
3. **SIMD Support**: Enable the highest SIMD level your CPU supports
4. **Network Selection**: Use small network for positions with large material imbalance

## Next Steps

1. Implement `position.h` with required interface
2. Initialize bitboards and threat offsets at startup
3. Load NNUE network files
4. Integrate evaluation into your search algorithm
5. Test with known positions

## Additional Resources

- See `PORTED_FILES_SUMMARY.md` for complete file list
- See `NNUE_PORTING_GUIDE.md` for detailed porting information
- See `position_interface_requirements.h` for Position interface details

