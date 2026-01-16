# model.py
# Questo file definisce l'architettura della rete neurale
# È il "cervello" che impara a prevedere la qualità del sonno

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_1, HIDDEN_2, HIDDEN_3, DROPOUT

class SleepNet(nn.Module):
    """
    Rete Neurale per predire la qualità del sonno.
    
    Struttura:
    - Input: le caratteristiche estratte (battito, respirazione, etc.)
    - 3 livelli nascosti con neuroni che si riducono gradualmente (256 -> 128 -> 64)
    - LayerNorm: stabilizza l'apprendimento
    - Dropout: spegne casualmente alcuni neuroni per evitare overfitting
    - Output: un singolo numero (il punteggio di sonno previsto)
    
    È come avere 3 "filtri" successivi che raffinano sempre più l'informazione
    finché non arriviamo alla previsione finale.
    """
    def __init__(self, input_dim):
        super().__init__()
        # Primo strato: da input_dim a HIDDEN_1 (256) neuroni
        self.fc1 = nn.Linear(input_dim, HIDDEN_1)
        self.ln1 = nn.LayerNorm(HIDDEN_1)  # normalizzazione per stabilità
        
        # Secondo strato: da 256 a 128 neuroni
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.ln2 = nn.LayerNorm(HIDDEN_2)
        
        # Terzo strato: da 128 a 64 neuroni
        self.fc3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.ln3 = nn.LayerNorm(HIDDEN_3)
        
        # Strato finale: da 64 a 1 (il punteggio)
        self.out = nn.Linear(HIDDEN_3, 1)
        
        # Dropout: durante il training "spegne" casualmente il 20% dei neuroni
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Questa funzione definisce come i dati "fluiscono" attraverso la rete.
        
        Per ogni strato:
        1. Applichiamo la trasformazione lineare (fc)
        2. Normalizziamo (ln)
        3. Applichiamo ReLU (funzione di attivazione: tiene solo i valori positivi)
        4. Applichiamo dropout (casualmente)
        """
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        return self.out(x)  # Nessuna attivazione finale (output lineare)