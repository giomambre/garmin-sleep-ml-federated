# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Due layer densi con dropout per ridurre overfitting su tabellare
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.20)

    def forward(self, x):
        # ReLU + dropout tra i layer, uscita lineare (regressione)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)
