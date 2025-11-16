# modal_train.py

import modal
import torch

app = modal.App("chess-train")

image = (
    modal.Image.debian_slim()
        .pip_install("datasets", "torch")
        .add_local_python_source("model", "training") 
)

# Data volume
data_vol = modal.Volume.from_name("lichess-preproc-vol", create_if_missing=False)
# A second volume for model checkpoints
model_vol = modal.Volume.from_name("chess-model-vol", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",                  # or "T4" if cheaper
    volumes={"/mnt/data": data_vol, "/checkpoints": model_vol},
    timeout=60 * 60 * 12,        # 12 hours
)
def train():
    from torch.utils.data import DataLoader
    from training.train_loop import ChessDataset, train_model
    from model.neural_network import ChessPolicyNet

    dataset_path = "/mnt/data/lichess_preprocessed.pt"
    train_ds = ChessDataset(dataset_path)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = ChessPolicyNet()
    train_model(model, train_loader, epochs=10, device="cuda")
