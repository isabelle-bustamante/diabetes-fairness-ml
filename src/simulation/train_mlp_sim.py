from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

device = "mps" if torch.backends.mps.is_available() else "cpu"

__all__ = [
    "MLP",
    "train_epoch",
    "evaluate",
    "train",
    "plot",
    "train_mlp_model"
]




class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # binary logits
        )

    def forward(self, x):
        return self.model(x)



def train_mlp_model(
    X_train_s, y_train,
    X_test_s, y_test,
    input_dim,
    hidden_dim=32,
    n_epochs=30,
    lr=0.001
):
    # tensors
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long).to(device)

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long).to(device)

    # model
    model = MLP(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # train (single loop, no dataloaders)
    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

    # evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()

    return {
        "model": model,
        "accuracy": (pred == y_test).mean(),
        "y_pred": pred,
        "y_proba": probs
    }


