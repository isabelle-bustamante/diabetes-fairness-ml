from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- Logistic Regression for Simulation ----
def train_lr_sim(X_train, y_train, X_test):
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    y_pred_probs = model.predict_proba(X_test)[:, 1]  # probabilities
    return {"model": model, "y_pred_probs": y_pred_probs}


# ---- Simple MLP for Simulation ----
class MLP_Sim(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # single output for binary classification
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

def train_mlp_sim(X_train, y_train, X_test, n_epochs=30, lr=0.001, batch_size=32):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = MLP_Sim(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

    # predict probabilities on test
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_pred_probs = torch.sigmoid(logits).cpu().numpy()

    return {"model": model, "y_pred_probs": y_pred_probs}
