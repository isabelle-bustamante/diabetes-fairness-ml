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

# For Apple MPS / CUDA safety
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "mps" if torch.backends.mps.is_available() else "cpu"



class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.model(x)



def train_epoch(model, dataloader, optimizer, loss_fn):
    losses = []
    correct = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        out = model(X)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += (out.argmax(dim=1) == y).sum().item()

    return np.mean(losses), correct / len(dataloader.dataset)


def evaluate(model, dataloader, loss_fn):
    losses = []
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            out = model(X)
            loss = loss_fn(out, y)

            losses.append(loss.item())
            correct += (out.argmax(dim=1) == y).sum().item()

    return np.mean(losses), correct / len(dataloader.dataset)


def train(model, train_loader, val_loader, optimizer, n_epochs, loss_fn):
    tr_losses, val_losses, tr_accs, val_accs = [], [], [], []

    for epoch in range(n_epochs):
        model.train()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn)

        model.eval()
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        tr_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"Train Loss {tr_loss:.4f}, Acc {tr_acc:.4f} | "
              f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

    return tr_losses, val_losses, tr_accs, val_accs



def plot(train_losses, val_losses, train_accs, val_accs, title):
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["train_loss", "val_loss"])
    plt.title(title + " - Loss")
    plt.show()

    plt.figure()
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.legend(["train_acc", "val_acc"])
    plt.title(title + " - Accuracy")
    plt.show()


def train_mlp_real(X_train_s, y_train, X_test_s, y_test,
                   input_dim, n_epochs=20, lr=0.001,
                   model_class=MLP, plot_training=True):

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

    model = model_class(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    tr_losses, val_losses, tr_accs, val_accs = train(
        model, train_loader, val_loader, optimizer, n_epochs, loss_fn
    )

    if plot_training:
        plot(tr_losses, val_losses, tr_accs, val_accs, title="MLP Real World")

    # predictions
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        pred = logits.argmax(dim=1).cpu().numpy()
        probs = logits.softmax(dim=1)[:, 1].cpu().numpy()

    return {
        "model": model,
        "y_pred": pred,
        "probs": probs,
        "accuracy": (pred == y_test.values).mean(),
        "train_losses": tr_losses,
        "val_losses": val_losses,
        "train_accs": tr_accs,
        "val_accs": val_accs,
    }

