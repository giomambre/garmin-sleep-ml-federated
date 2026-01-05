# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Tre layer densi con LayerNorm (funziona anche con batch=1)
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # ReLU + LayerNorm + dropout tra i layer
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        return self.out(x)
