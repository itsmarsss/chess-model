"""
This module: trains the neural network model with the preprocessed data
- It initializes the model and runs the training loop:
    - Forward pass, compute loss (cross-entropy), backpropagation
- It saves checkpoints/logs
- Validation?

train.py
  - load dataset
  - build model
  - optimizer, scheduler
  - training loop
  - checkpoint saving

"""
import torch
from torch.optim import Adam #optimizer
from torch.utils.data import Dataset, DataLoader #load dataset

# training hyperparameters
batch_size = 256
lr = 1e-3
weight_decay = 1e-4
epochs = 5

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_path):
        data = torch.load(tensor_path)
        X, y = zip(*data)
        self.X = torch.stack(X)
        self.y = torch.stack(y)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#loader
import os
print("Files in /mnt/data:", os.listdir("/mnt/data"))

train_ds = ChessDataset("/mnt/data/lichess_preprocessed.pt")
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

def train_model(model, train_loader, epochs, device="cuda"):
    model.to(device)
    model.train()
    
    #optimizer + LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs*len(train_loader)
    )
    
    #loss function - crossentropy soft-label, kl-divergence
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

    for epoch in range(epochs):
        total_loss = 0.0
        
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            logits = model(X)
            log_probs = torch.log_softmax(logits, dim=1)
            
            loss = loss_fn(log_probs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")
        
        torch.save(model.state_dict(), f"/checkpoints/{epoch+1}.pt")
    
    torch.save(model.state_dict(), "/checkpoints/final.pt")
    from modal import CloudBucket