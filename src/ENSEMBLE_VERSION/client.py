# client.py
# Questo file gestisce l'allenamento locale di ogni singolo utente (client)
# In pratica, ogni utente allena il modello solo sui suoi dati personali

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from config import *

def train_local(model, data, epochs, lr):
    """
    Funzione che allena il modello su un singolo utente (client).
    
    In parole semplici:
    - Prendiamo i dati di un utente (le sue caratteristiche e il suo punteggio di sonno)
    - Alleniamo il modello solo su quei dati per alcune ripetizioni (epochs)
    - Restituiamo i pesi del modello aggiornati e quanti esempi ha visto
    
    Questo serve per il federated learning: ogni utente allena il modello in privato,
    poi tutti condividono solo i pesi (non i dati personali).
    """
    # Ogni client allena il modello solo sui propri dati (X=feature, y=target)
    X, y = data
    
    # Creiamo un dataset PyTorch partendo dai dati numpy
    # Il DataLoader ci permette di processare i dati in piccoli gruppi (batch)
    # e li mescola ad ogni giro per variare l'ordine di apprendimento
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Definiamo la funzione di errore e l'ottimizzatore
    # L1Loss = Mean Absolute Error (MAE) - misura quanto sbagli in media
    criterion = nn.L1Loss()
    # Adam è un algoritmo che aggiusta i pesi del modello per ridurre l'errore
    # weight_decay serve per evitare che il modello impari troppo i dati specifici (overfitting)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Mettiamo il modello in modalità allenamento
    model.train()
    
    # Ripetiamo l'allenamento per più giri (epochs)
    for _ in range(epochs):
        # Per ogni piccolo gruppo di dati (batch)
        for xb, yb in loader:
            # --- TRUCCO: AGGIUNGIAMO UN PO' DI RUMORE AI DATI ---
            # Perché? Nella vita reale i sensori non sono perfetti e fanno piccoli errori
            # Aggiungendo un po' di rumore casuale durante l'allenamento,
            # il modello impara a essere più robusto e funziona meglio su dati nuovi
            if model.training:
                noise = torch.randn_like(xb) * 0.05  # rumore piccolo (5% circa)  # rumore piccolo (5% circa)
                xb = xb + noise
            
            # Questo è il ciclo standard di apprendimento:
            optimizer.zero_grad()           # 1. Azzeriamo i gradienti precedenti
            loss = criterion(model(xb), yb) # 2. Calcoliamo quanto sbagli (errore)
            loss.backward()                 # 3. Calcoliamo come correggere i pesi
            optimizer.step()                # 4. Aggiorniamo i pesi

    # Alla fine restituiamo i pesi aggiornati del modello e quanti dati abbiamo usato
    # Il numero di dati serve al server per fare la media: chi ha più dati pesa di più
    return model.state_dict(), len(X)
