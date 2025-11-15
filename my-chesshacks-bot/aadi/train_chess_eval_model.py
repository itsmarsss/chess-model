# train_chess_eval_modal.py

import os
from dataclasses import dataclass

import modal

# ---------- Modal setup ----------
app = modal.App("chess-eval-cnn")

# Persistent storage for models
VOLUME_NAME = "chess-models"
CACHE_VOLUME_NAME = "chess-dataset-cache"


# Image with dependencies installed
# You can add versions if you want: .pip_install(["torch==2.2.0", ...])
image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "torch",
            "numpy",
            "datasets",
            "python-chess",
            "tqdm",
        ]
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


# ---------- Training Config ----------
from dataclasses import dataclass


@dataclass
class Config:
    model_out: str = "chess_eval_cnn.pt"
    max_positions: int = 1_000_000  # Use more data
    train_fraction: float = 0.9
    batch_size: int = 128  # Reduce for larger model
    epochs: int = 100  # More epochs for convergence
    lr: float = 5e-4  # Lower learning rate
    cp_scale: float = 1000.0
    num_workers: int = 4
    num_blocks: int = 6  # Reduced from 10 to 6
    num_channels: int = 128  # Reduced from 256 to 128
    value_hidden: int = 128  # Reduced FC layer size


CFG = Config()

# ---------- Model code moved into the remote worker ----------
# NOTE: heavy dependencies (numpy, torch, datasets, python-chess, etc.) are
# imported inside the remote function so that `modal run` executed locally
# doesn't require these packages to be installed on the developer machine.


# ---------- Remote training function ----------
@app.function(
    image=image,
    gpu="B200",
    volumes={
        "/models": volume,
        "/cache": cache_volume,  # Mount cache volume
    },
    timeout=60 * 60 * 14,
)
def train_remote(cfg_dict: dict):
    # Reconstruct config on the worker
    cfg = Config(**cfg_dict)
    # Import heavy dependencies inside the worker so the local process
    # (which executes `modal run` to register the function) doesn't need them.
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import chess
    from datasets import load_dataset
    from tqdm import tqdm

    PIECE_TO_INDEX = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }


    def fen_to_tensor(fen: str) -> torch.Tensor:
        planes = np.zeros((13, 8, 8), dtype=np.float32)
        board = chess.Board(fen)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            base = 0 if piece.color == chess.WHITE else 6
            p_index = PIECE_TO_INDEX[piece.piece_type] + base

            file = chess.square_file(square)
            rank = chess.square_rank(square)
            planes[p_index, 7 - rank, file] = 1.0

        stm_plane = 12
        planes[stm_plane, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

        return torch.from_numpy(planes)


    class ChessEvalDataset(Dataset):
        def __init__(self, hf_dataset, cp_scale: float):
            self.data = hf_dataset
            self.cp_scale = cp_scale
            # Set a very high value for mate (e.g., 10000 centipawns)
            self.mate_value = 10000.0

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data[idx]
            fen = row["fen"]
            cp = row["cp"]
            mate = row["mate"]
            
            x = fen_to_tensor(fen)
            
            # Handle mate vs centipawn evaluation
            if mate is not None:
                # Mate value: positive if white is winning, negative if black is winning
                # Mate in N moves -> use mate_value
                if mate > 0:
                    eval_cp = self.mate_value
                else:
                    eval_cp = -self.mate_value
            elif cp is not None:
                eval_cp = float(cp)
            else:
                # Skip positions with neither cp nor mate (shouldn't happen)
                # But as a fallback, use 0
                eval_cp = 0.0
            
            # Normalize by cp_scale
            y = eval_cp / self.cp_scale
            y = torch.tensor(y, dtype=torch.float32)
            
            return x, y


    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            
        def forward(self, x):
            residual = x
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = torch.relu(out)
            return out

    class ChessEvalCNN(nn.Module):
        def __init__(self, num_blocks=6, num_channels=128, value_hidden=128):
            super().__init__()
            # Initial convolution
            self.conv_input = nn.Sequential(
                nn.Conv2d(13, num_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_channels),
                nn.ReLU()
            )
            
            # Residual tower
            self.residual_blocks = nn.Sequential(
                *[ResidualBlock(num_channels) for _ in range(num_blocks)]
            )
            
            # Value head - more compact
            self.value_head = nn.Sequential(
                nn.Conv2d(num_channels, 16, kernel_size=1),  # Reduced from 32 to 16
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16 * 8 * 8, value_hidden),  # 1024 -> value_hidden
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(value_hidden, 1)  # value_hidden -> 1
            )

        def forward(self, x):
            x = self.conv_input(x)
            x = self.residual_blocks(x)
            x = self.value_head(x)
            return x.squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Worker] Using device: {device}")

    # Set cache directory for Hugging Face datasets
    import os
    os.environ["HF_DATASETS_CACHE"] = "/cache/huggingface"
    
    print("[Worker] Loading dataset from Hugging Face...")
    try:
        dset = load_dataset(
            "Lichess/chess-position-evaluations", 
            split="train",
            cache_dir="/cache/huggingface"  # Explicitly set cache
        )
        # Don't filter out positions - we'll handle mate in the dataset class
        print(f"[Worker] Loaded {len(dset)} positions")
    except Exception as e:
        print(f"[Worker] Error loading dataset: {e}")
        raise

    if cfg.max_positions is not None and cfg.max_positions < len(dset):
        dset = dset.shuffle(seed=42).select(range(cfg.max_positions))
        print(f"[Worker] Subsampled to {len(dset)} positions.")
    else:
        print(f"[Worker] Using full dataset of {len(dset)} positions.")

    split_idx = int(len(dset) * cfg.train_fraction)
    dset_train = dset.select(range(split_idx))
    dset_val = dset.select(range(split_idx, len(dset)))

    print(f"[Worker] Train size: {len(dset_train)}, Val size: {len(dset_val)}")

    train_ds = ChessEvalDataset(dset_train, cp_scale=cfg.cp_scale)
    val_ds = ChessEvalDataset(dset_val, cp_scale=cfg.cp_scale)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = ChessEvalCNN(
        num_blocks=cfg.num_blocks,
        num_channels=cfg.num_channels,
        value_hidden=cfg.value_hidden
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    # Fix deprecated API
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            # Fix deprecated API
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                preds = model(xb)
                loss = criterion(preds, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)
            pbar.set_postfix(loss=total_loss / count)

        train_loss = total_loss / count

        # validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                val_count += xb.size(0)
        val_loss /= val_count

        print(
            f"[Worker] Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save to /models inside the container; backed by Modal Volume
            save_path = os.path.join("/models", cfg.model_out)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                },
                save_path,
            )
            print(f"[Worker] Saved new best model to {save_path}")

            volume.commit()
            print(f"[Worker] Volume committed")

    print(f"[Worker] Training complete. Best val loss: {best_val_loss:.6f}")

    # Commit both volumes
    volume.commit()
    cache_volume.commit()  # Commit cache to persist dataset
    print("[Worker] Volumes committed")
    
    return cfg.model_out


# ---------- Local entrypoint ----------
@app.local_entrypoint()
def main():
    # Launch training on cloud worker
    print("[Local] Starting remote training...")
    model_name = train_remote.remote(CFG.__dict__)
    print(f"[Local] Remote training finished, model saved as {model_name} in volume '{VOLUME_NAME}' at /models/{model_name}")

    # If you want, you can now mount the volume locally using 'modal volume get'
    # from your terminal to download the file, e.g.:
    #   modal volume get chess-models /models/chess_eval_cnn.pt .
    print(
        "\nTo download the model locally, run from your shell:\n"
        f"modal volume get {VOLUME_NAME} /models/{model_name} .\n"
    )
