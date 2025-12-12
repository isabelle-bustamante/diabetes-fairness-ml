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
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.model(x)

class MLP_Realistic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)



# This function was adapted from Deep Learning by Prof. Paolo Favaro, University of Bern


def train_epoch(model, train_dataloader, optimizer, loss_fn):
    losses = []
    correct_predictions = 0

    for features, labels in train_dataloader:
        features = features.to(device)
        labels = labels.to(device)

        output = model(features)

        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        predicted = output.argmax(dim=1)
        correct_predictions += (predicted == labels).sum().item()

    mean_loss = np.mean(losses)
    accuracy = 100.0 * correct_predictions / len(train_dataloader.dataset)
    return mean_loss, accuracy


# This function was adapted from Deep Learning by Prof. Paolo Favaro, University of Bern

def evaluate(model, dataloader, loss_fn):
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            output = model(features)
            loss = loss_fn(output, labels)

            losses.append(loss.item())
            predicted = output.argmax(dim=1)
            correct_predictions += (predicted == labels).sum().item()

    mean_loss = np.mean(losses)
    accuracy = 100.0 * correct_predictions / len(dataloader.dataset)
    return mean_loss, accuracy


# This function was adapted from Deep Learning by Prof. Paolo Favaro, University of Bern

def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_fn):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(n_epochs):
        model.train()
        tr_loss, tr_acc = train_epoch(model, train_dataloader, optimizer, loss_fn)

        model.eval()
        val_loss, val_acc = evaluate(model, val_dataloader, loss_fn)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.2f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}")

    return train_losses, val_losses, train_accs, val_accs




# This function was adapted from Deep Learning by Prof. Paolo Favaro, University of Bern

def plot(train_losses, val_losses, train_accuracies, val_accuracies, title):
    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title(f'{title}: Train/val loss')

    plt.figure()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies)
    plt.plot(np.arange(len(val_accuracies)), val_accuracies)
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.title(f'{title}: Train/val accuracy')




def train_mlp_model(
    X_train_s, y_train,
    X_test_s, y_test,
    input_dim,
    hidden_dim=32,
    n_epochs=20,
    lr=0.001,
    model_class=MLP,       # <-- ADD THIS
    plot_training=True
):

    """
    High-level wrapper that:
    - converts numpy → torch tensors
    - builds model
    - trains model
    - returns metrics + predictions
    """

    # Convert to tensors
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long).to(device)

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long).to(device)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

    # Build model
    model = model_class(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, test_loader,
        optimizer, n_epochs, loss_fn
    )

    if plot_training:
        plot(train_losses, val_losses, train_accs, val_accs, "MLP")

    # Evaluate final model
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        pred = logits.argmax(dim=1).cpu().numpy()
        probs = logits.softmax(dim=1)[:, 1].cpu().numpy()

    # Return dictionary (same format as plots)
    return {
        "model": model,
        "pred": pred,
        "probs": probs,
        "accuracy": (pred == y_test).mean(),
        "AUROC": roc_auc_score(y_test, probs),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs
    }


