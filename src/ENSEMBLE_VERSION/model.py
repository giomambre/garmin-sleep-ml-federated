# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_1, HIDDEN_2, HIDDEN_3, DROPOUT

class SleepNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Tre layer densi con LayerNorm (funziona anche con batch=1)
        self.fc1 = nn.Linear(input_dim, HIDDEN_1)
        self.ln1 = nn.LayerNorm(HIDDEN_1)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.ln2 = nn.LayerNorm(HIDDEN_2)
        self.fc3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.ln3 = nn.LayerNorm(HIDDEN_3)
        self.out = nn.Linear(HIDDEN_3, 1)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # ReLU + LayerNorm + dropout tra i layer
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        return self.out(x)
