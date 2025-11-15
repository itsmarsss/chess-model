# NNUE Training Code

This folder contains the Python/PyTorch code for training NNUE networks.

## Purpose

Train new NNUE networks from scratch or fine-tune existing ones using chess position datasets.

## Structure

- `train.py` - Main training script
- `run_games.py` - Test trained networks by playing games
- `serialize.py` - Convert PyTorch models to `.nnue` format
- `model/` - PyTorch model architecture definitions
- `data_loader/` - Training data loading (Python + C++)
- `lib/` - C++ data loader library headers
- `scripts/` - Helper scripts for training
- `requirements.txt` - Python dependencies

## Setup

### Using Docker (Recommended)

```bash
./run_docker.sh
```

### Local Setup

```bash
pip install -r requirements.txt
```

## Training

### Easy Training
```bash
python scripts/easy_train.py
```

### Advanced Training
```bash
python train.py --help
```

## Serialization

Convert trained PyTorch models to `.nnue` format for use with evaluation code:

```bash
python serialize.py checkpoint.ckpt output.nnue
```

## Testing Networks

Run games to test trained networks:

```bash
python run_games.py --concurrency 16 run96
```

## Workflow

1. **Prepare Data**: Collect chess positions with evaluations
2. **Train**: Run `train.py` to train network
3. **Serialize**: Convert `.ckpt` to `.nnue` using `serialize.py`
4. **Use**: Load `.nnue` file in evaluation code

## Documentation

See the original README.md in this folder for detailed instructions.

## Note

This is the **training** code. The **evaluation** code is in `../evaluation/`.
