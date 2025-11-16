from chess_policy_net import ChessPolicyNet
from training.train_loop import ChessDataset
from torch.utils.data import DataLoader
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ChessDataset("modal_cache/dataset.pt")
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = ChessPolicyNet()
    train_model(model, loader, epochs=5, device=device)

if __name__ == "__main__":
    main()