# train_chess_eval_modal.py

import os
from dataclasses import dataclass

import modal

# ---------- Modal setup ----------
app = modal.App("chess-eval-cnn")

# Persistent storage for models
VOLUME_NAME = "chess-models"


# Image with dependencies installed
# You can add versions if you want: .pip_install(["torch==2.2.0", ...])
image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "numpy",
            "torch",
            "datasets",
            "python-chess",
            "tqdm",
        ]
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ---------- Training Config ----------
from dataclasses import dataclass


@dataclass
class Config:
    model_out: str = "chess_eval_cnn.pt"
    max_positions: int = 200_000
    train_fraction: float = 0.9
    batch_size: int = 256
    epochs: int = 5
    lr: float = 1e-3
    cp_scale: float = 1000.0
    num_workers: int = 4


CFG = Config()

# ---------- Model code moved into the remote worker ----------
# NOTE: heavy dependencies (numpy, torch, datasets, python-chess, etc.) are
# imported inside the remote function so that `modal run` executed locally
# doesn't require these packages to be installed on the developer machine.


# ---------- Remote training function ----------
@app.function(
    image=image,
    gpu="A10G",  # or "T4", "A100", etc. depending on your Modal plan
    volumes={"/models": volume},
    timeout=60 * 60 * 4,  # up to 4 hours, adjust as needed
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

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data[idx]
            fen = row["fen"]
            cp = row["cp"]
            x = fen_to_tensor(fen)
            y = float(cp) / self.cp_scale
            return x, y


    class ChessEvalCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(13, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x = self.conv(x)
            x = self.head(x)
            return x.squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Worker] Using device: {device}")

    print("[Worker] Loading dataset from Hugging Face...")
    dset = load_dataset("Lichess/chess-position-evaluations", split="train")
    dset = dset.filter(lambda ex: ex["cp"] is not None)

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

    model = ChessEvalCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
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

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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
    # Return the path within the volume (for logging)
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
