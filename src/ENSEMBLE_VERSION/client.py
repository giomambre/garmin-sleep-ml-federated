# client.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from config import *
def train_local(model, data, epochs, lr):
    # Ogni client allena il modello solo sui propri dati (X=feature, y=target)
    X, y = data
    # Creiamo un dataset Torch; il dataloader fa batching e shuffle ad ogni epoch
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Adam con weight decay per regolarizzare
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Restituiamo i pesi aggiornati e il numero di esempi visti (serve per la media pesata)
    return model.state_dict(), len(X)

