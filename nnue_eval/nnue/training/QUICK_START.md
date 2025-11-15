# Quick Start - Modal Training

## Default Behavior: 1 File Only

By default, the script downloads **only 1 file** for faster runs. This is perfect for testing and faster iteration.

## Usage Examples

### 1. Quick Test (1 minute, 1 file)
```bash
modal run modal_train.py --test-mode
```

### 2. Training with 1 File (Default - Fastest)
```bash
modal run modal_train.py
# Or explicitly:
modal run modal_train.py --max-files 1
```

### 3. Training with Specific File
```bash
modal run modal_train.py --specific-file test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst
```

### 4. Training with Multiple Files
```bash
modal run modal_train.py --max-files 3
```

### 5. Training with All Files (Full Dataset)
```bash
modal run modal_train.py --max-files 0
```

## File Sizes

Each `.binpack.zst` file is approximately:
- **Compressed**: 7-8 GB
- **Decompressed**: ~15-20 GB (estimated)

Downloading and decompressing 1 file takes:
- **Download**: ~5-10 minutes (depending on connection)
- **Decompress**: ~2-5 minutes

## Recommendations

- **For testing**: Use `--test-mode` (1 file, minimal training)
- **For development**: Use default (1 file, full training)
- **For production**: Use `--max-files 0` (all files, full training)

## Available Files in Dataset

The dataset contains files like:
- `test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst`
- `test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst`
- `test80-2024-03-mar-2tb7p.min-v2.v6.binpack.zst`
- ... and more

You can specify any of these with `--specific-file`.

