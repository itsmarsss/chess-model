# NNUE Evaluation Module

Complete NNUE (Efficiently Updatable Neural Networks) evaluation code ported from Stockfish, organized for easy integration into chess projects.

## Structure

```
nnue_eval/
├── Foundation Files
│   ├── types.h              # Core chess types
│   ├── misc.h               # Utility functions
│   ├── bitboard.h/cpp       # Bitboard operations
│   └── position.h           # Position interface (YOU NEED TO PROVIDE)
│
└── nnue/
    ├── evaluation/          # C++ Evaluation Code
    │   ├── network.h/cpp   # Network loading/evaluation
    │   ├── nnue_accumulator.h/cpp
    │   ├── nnue_feature_transformer.h
    │   ├── nnue_architecture.h
    │   ├── nnue_common.h
    │   ├── nnue_misc.h/cpp
    │   ├── simd.h
    │   ├── features/        # Feature extraction
    │   └── layers/          # Neural network layers
    │
    └── training/            # Python Training Code
        ├── train.py         # Main training script
        ├── serialize.py     # Convert to .nnue format
        └── ...              # Training infrastructure
```

## Quick Start

### 1. Provide Position Interface

Create `nnue_eval/position.h` implementing the Position class. See `INTEGRATION.md` for interface requirements.

### 2. Initialize at Startup

```cpp
#include "nnue_eval/bitboard.h"
#include "nnue_eval/nnue/evaluation/features/full_threats.h"

void initialize_nnue() {
    Stockfish::Bitboards::init();
    Stockfish::Eval::NNUE::Features::init_threat_offsets();
}
```

### 3. Load Networks

```cpp
#include "nnue_eval/nnue/evaluation/network.h"

using namespace Stockfish::Eval::NNUE;

auto networkBig = std::make_unique<NetworkBig>(...);
networkBig->load(".", "nn-49c1193b131c.nnue");
```

### 4. Evaluate Positions

```cpp
AccumulatorStack accumulatorStack;
AccumulatorCaches accumulatorCaches(networks);

auto [psqt, positional] = networkBig->evaluate(
    pos, accumulatorStack, caches.big
);
```

## Compilation

```bash
g++ -std=c++17 -O3 -DUSE_AVX2 -I. \
    nnue_eval/bitboard.cpp \
    nnue_eval/nnue/evaluation/nnue_accumulator.cpp \
    nnue_eval/nnue/evaluation/network.cpp \
    nnue_eval/nnue/evaluation/nnue_misc.cpp \
    nnue_eval/nnue/evaluation/features/half_ka_v2_hm.cpp \
    nnue_eval/nnue/evaluation/features/full_threats.cpp \
    your_code.cpp
```

## Include Paths

From outside `nnue_eval/`:
```cpp
#include "nnue_eval/types.h"
#include "nnue_eval/nnue/evaluation/network.h"
```

## What's Included

- ✅ **Evaluation Code**: 18 C++ files for fast position evaluation
- ✅ **Training Code**: 50+ Python files for training networks
- ✅ **Foundation**: Core types and utilities
- ✅ **Self-Contained**: All dependencies included

## Documentation

- **INTEGRATION.md** - Detailed integration guide
- **nnue/evaluation/README.md** - Evaluation code details
- **nnue/training/README.md** - Training code details

## Important Notes

- **Evaluation Only**: The evaluation code loads and uses pre-trained networks. It does NOT train networks.
- **Training**: Use `nnue/training/` for training new networks (separate Python codebase).
- **Position Interface**: You must provide `position.h` - see `INTEGRATION.md` for requirements.

## Benefits

✅ Single folder for all NNUE work  
✅ Self-contained with relative includes  
✅ Both evaluation and training included  
✅ Easy to integrate and manage
