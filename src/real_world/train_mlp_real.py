import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = "cuda"
elif torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
    device = "mps"
else:
    device = "cpu"


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


def train_epoch(model, dataloader, optimizer, loss_fn):
    losses = []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def evaluate(model, dataloader, loss_fn):
    losses = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            losses.append(loss.item())

    return np.mean(losses)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    n_epochs=50,
    patience=5
):
    train_losses, val_losses = [], []
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fn)

        model.eval()
        val_loss = evaluate(model, val_loader, loss_fn)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training vs Validation Loss")
    plt.show()


def train_mlp_real(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    input_dim,
    n_epochs=50,
    lr=1e-3,
    batch_size=32,
    patience=5,
    plot_training=True
):
    """
    Trains an MLP using train/val splits and returns test probabilities only.
    """

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False
    )

    # Model
    model = MLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)




    loss_fn = nn.BCEWithLogitsLoss()


    # Train
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        patience=patience
    )

    if plot_training:
        plot_losses(train_losses, val_losses)

    # Final test inference
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_pred_probs = torch.sigmoid(logits).cpu().numpy()

    return {
        "model": model,
        "y_pred_probs": y_pred_probs,
        "train_losses": train_losses,
        "val_losses": val_losses
    }