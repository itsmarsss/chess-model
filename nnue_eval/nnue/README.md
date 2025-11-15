# NNUE Module Structure

This folder contains both evaluation and training code for NNUE networks.

## Folder Structure

```
nnue_eval/
├── types.h, misc.h, bitboard.h, bitboard.cpp  (Foundation files)
├── position.h  (YOU NEED TO PROVIDE)
│
└── nnue/
    ├── evaluation/          # Evaluation code (C++)
    │   ├── network.h/cpp    # Network loading and evaluation
    │   ├── nnue_accumulator.h/cpp
    │   ├── nnue_feature_transformer.h
    │   ├── nnue_architecture.h
    │   ├── nnue_common.h
    │   ├── nnue_misc.h/cpp
    │   ├── simd.h
    │   ├── features/        # Feature extraction
    │   └── layers/          # Neural network layers
    │
    └── training/            # Training code (Python)
        ├── train.py         # Main training script
        ├── run_games.py     # Game runner for testing
        ├── serialize.py     # Network serialization
        ├── model/           # PyTorch model definitions
        ├── data_loader/     # Training data loading
        ├── lib/             # C++ data loader library
        ├── scripts/         # Training scripts
        └── requirements.txt # Python dependencies
```

## Evaluation Code (`nnue/evaluation/`)

**Purpose**: Fast evaluation of chess positions using pre-trained networks

**Language**: C++

**Usage**: Load `.nnue` network files and evaluate positions

**Key Files**:
- `network.h/cpp` - Network loading and evaluation
- `nnue_accumulator.h/cpp` - Incremental evaluation updates
- `features/` - Feature extraction (HalfKAv2, FullThreats)
- `layers/` - Neural network layers (forward pass only)

## Training Code (`nnue/training/`)

**Purpose**: Train new NNUE networks from scratch

**Language**: Python (PyTorch)

**Usage**: Train networks using chess position datasets

**Key Files**:
- `train.py` - Main training script
- `model/` - PyTorch model architecture
- `data_loader/` - Training data loading
- `serialize.py` - Convert PyTorch models to `.nnue` format
- `run_games.py` - Test trained networks by playing games

## Workflow

1. **Training**: Use `nnue/training/train.py` to train networks
2. **Serialization**: Use `nnue/training/serialize.py` to convert to `.nnue` format
3. **Evaluation**: Use `nnue/evaluation/` code to load and use trained networks

## Include Paths

### From evaluation code:
- Root files: `#include "../../types.h"`
- Same directory: `#include "nnue_common.h"`
- Features: `#include "../nnue_common.h"`
- Layers: `#include "../nnue_common.h"`

### From training code:
- Python imports work normally: `import model`, `import data_loader`

## Quick Start

### Training
```bash
cd nnue_eval/nnue/training
pip install -r requirements.txt
python train.py --help
```

### Evaluation
```cpp
#include "nnue_eval/nnue/evaluation/network.h"
// ... use network for evaluation
```

See individual README files in each subfolder for detailed instructions.

