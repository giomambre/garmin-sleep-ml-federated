# server.py
# Questo file gestisce l'aggregazione dei modelli nel Federated Learning
# È il "coordinatore" che combina i modelli di tutti gli utenti

import copy
import torch
from config import WEIGHT_CAP_QUANTILE


def fed_avg(global_model, client_states, client_sizes):
    """
    Federated Averaging: la magia del Federated Learning!
    
    Cosa fa?
    Prende i modelli allenati da tutti gli utenti e li combina in un unico modello globale.
    
    Come?
    Fa una MEDIA PESATA: gli utenti con più dati contano di più.
    MA con un trucco: limitiamo il peso massimo (cap) per evitare che un utente
    con moltissimi dati domini completamente sugli altri.
    
    Esempio:
    - Utente A ha 100 dati -> peso = 100
    - Utente B ha 50 dati -> peso = 50  
    - Utente C ha 1000 dati -> peso limitato al 90° percentile (es. 200)
    
    Il modello finale è: (A*100 + B*50 + C*200) / 350
    """
    # Convertiamo le dimensioni in tensori
    sizes = torch.tensor(client_sizes, dtype=torch.float32)
    
    # Calcoliamo il cap: il 90° percentile delle dimensioni
    cap = torch.quantile(sizes, WEIGHT_CAP_QUANTILE)
    
    # Limitiamo le dimensioni al cap
    eff = torch.minimum(sizes, cap)
    
    # Calcoliamo i pesi normalizzati (somma = 1)
    total = eff.sum().item() if eff.sum() > 0 else 1.0
    weights = [float(s / total) for s in eff]

    # Creiamo il nuovo stato del modello
    new_state = copy.deepcopy(global_model.state_dict())
    
    # Per ogni parametro del modello (pesi e bias di ogni layer)
    for k in new_state.keys():
        # Moltiplichiamo ogni modello client per il suo peso e li sommiamo
        stacked = torch.stack([cs[k] * w for cs, w in zip(client_states, weights)])
        new_state[k] = stacked.sum(dim=0)

    # Aggiorniamo il modello globale con i nuovi pesi
    global_model.load_state_dict(new_state)

