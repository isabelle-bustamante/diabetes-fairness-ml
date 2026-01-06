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
    """
    A simple Multi-Layer Perceptron (MLP) class.

    This class implements a fully connected feedforward neural network with one
    hidden layer and ReLU activation, designed for binary classification tasks.
    It takes in an input of specified dimensions, maps it through a hidden layer
    with a configurable number of units, and outputs logits for two binary classes.

    :ivar model: The sequential neural network model composed of linear layers
        and a ReLU activation function.
    :type model: torch.nn.Sequential
    """
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
    """
    Train and evaluate a Multi-Layer Perceptron (MLP) model using the given data.

    This function takes training and test data, initializes an MLP with specified
    hyperparameters, trains the model, and evaluates it. It returns a dictionary
    containing the trained model, the accuracy score, predicted labels, and predicted
    probabilities for the test data.

    :param X_train_s: Training features as a standard scaled numpy array
    :param y_train: Training target as a pandas Series
    :param X_test_s: Test features as a standard scaled numpy array
    :param y_test: Test target as a pandas Series
    :param input_dim: Input dimension size for the MLP model
    :param hidden_dim: Hidden layer dimension size for the MLP model (default: 32)
    :param n_epochs: Number of epochs for model training (default: 30)
    :param lr: Learning rate for the optimizer (default: 0.001)
    :return: Dictionary containing the trained model, accuracy score, predicted
             labels, and predicted probabilities for the test data
    """
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


