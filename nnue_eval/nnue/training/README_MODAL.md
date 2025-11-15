# Running NNUE Training on Modal

This guide explains how to run NNUE training on Modal using Hugging Face datasets.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Hugging Face Account**: For accessing datasets (optional if dataset is public)
3. **Modal CLI**: Install with `pip install modal`

## Setup

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal token set
```

Follow the prompts to authenticate.

### 3. (Optional) Create Hugging Face Secret

If using private datasets, create a Modal secret:

```bash
modal secret create huggingface HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

## Running Training

### Basic Usage

```bash
modal run nnue_eval/nnue/training/modal_train.py \
    --dataset-name linrock/test80-2024 \
    --batch-size 16384 \
    --max-epochs 10 \
    --features HalfKAv2_hm^
```

### Parameters

- `--dataset-name`: Hugging Face dataset name (default: `linrock/test80-2024`)
- `--batch-size`: Training batch size (default: 16384)
- `--max-epochs`: Maximum number of epochs (default: 10)
- `--features`: Feature set to use (default: `HalfKAv2_hm^`)
  - Options: `HalfKP`, `HalfKP^`, `HalfKA`, `HalfKA^`, `HalfKAv2`, `HalfKAv2^`, `HalfKAv2_hm`, `HalfKAv2_hm^`

### Example Commands

**Quick test run:**
```bash
modal run nnue_eval/nnue/training/modal_train.py \
    --dataset-name linrock/test80-2024 \
    --batch-size 8192 \
    --max-epochs 1
```

**Full training run:**
```bash
modal run nnue_eval/nnue/training/modal_train.py \
    --dataset-name linrock/test80-2024 \
    --batch-size 16384 \
    --max-epochs 50 \
    --features HalfKAv2_hm^
```

## Dataset Format

The script expects the Hugging Face dataset to contain `.binpack` files. The dataset `linrock/test80-2024` should have these files.

If your dataset is in a different format, you may need to:
1. Convert it to `.binpack` format
2. Upload it to Hugging Face
3. Modify the download logic in `modal_train.py`

## GPU Options

By default, the script uses `A10G` GPU. To change the GPU type, edit `modal_train.py`:

```python
@app.function(
    gpu="A100",  # Options: "A10G", "A100", "T4", "L4"
    ...
)
```

## Checkpoints

Checkpoints are automatically saved to a Modal volume named `nnue-checkpoints`. They persist between runs.

To download checkpoints:
```bash
modal volume download nnue-checkpoints /local/path
```

## Monitoring

View logs and monitor training:
```bash
modal app logs nnue-training
```

Or use the Modal dashboard at https://modal.com/apps

## Troubleshooting

### Dataset Not Found
- Verify the dataset name is correct
- Check if the dataset is public or if you need authentication
- Ensure you've created the Hugging Face secret if needed

### C++ Data Loader Issues
The C++ data loader may need to be compiled. The script will attempt this, but if it fails:
1. Compile locally and include the binary
2. Or modify the image to include build tools

### Out of Memory
- Reduce `batch-size`
- Reduce `epoch-size`
- Use a smaller GPU (T4 instead of A100)

## Cost Estimation

Modal charges based on:
- GPU time used
- Storage (volumes)
- Network egress

Approximate costs:
- A10G: ~$1.10/hour
- A100: ~$4.00/hour
- T4: ~$0.40/hour

A typical training run might take 10-50 hours depending on dataset size and epochs.

## Next Steps

After training completes:
1. Download checkpoints from the Modal volume
2. Convert to `.nnue` format using `serialize.py`:
   ```bash
   python serialize.py checkpoint.ckpt output.nnue --features HalfKAv2_hm^
   ```
3. Use the `.nnue` file with the evaluation code

