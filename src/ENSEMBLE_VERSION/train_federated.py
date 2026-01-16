# train_federated.py
# Questo è il file PRINCIPALE che allena tutti i modelli
# È il "direttore d'orchestra" che coordina tutto il processo

import torch
import random
import numpy as np
import joblib
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from config import SEED, DATASET_PATH, ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, LR, HIDDEN_1, HIDDEN_2, HIDDEN_3, DROPOUT, TARGET_SCALE, CLIENT_FRACTION, NUM_FOLDS, TOP_FEATURES, ARTIFACTS_DIR, EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR, MIN_LR
from model import SleepNet
from data_utils import load_clients, FederatedScaler, transform_clients, transform_clients_user_wise
from client import train_local
from server import fed_avg
import os

def run_training():
    """
    Funzione principale che allena l'ensemble di modelli con Federated Learning.
    
    STRATEGIA ENSEMBLE + CROSS-VALIDATION:
    Alleniamo 10 modelli diversi (uno per fold), ognuno su un sottoinsieme diverso
    di utenti. Questo rende il sistema più robusto: se un modello sbaglia,
    gli altri possono compensare.
    
    FEDERATED LEARNING:
    Ogni utente allena il modello sui suoi dati in privato, poi condividiamo
    solo i pesi (non i dati). È come un "lavoro di squadra" dove ognuno contribuisce
    senza rivelare informazioni personali.
    
    STRATEGIA DI SCALING INNOVATIVA:
    - Training: normalizzazione PER-UTENTE (impara pattern relativi)
    - Validazione: normalizzazione GLOBALE (simula test set dove non conosciamo l'utente)
    
    Workflow per ogni fold:
    1. Dividi utenti in training e validazione
    2. Calcola scaler globale (serve per Kaggle)
    3. Normalizza training data per-utente
    4. Per ogni round di comunicazione:
       - Alcuni utenti allenano il modello sui loro dati
       - Il server combina i modelli (federated averaging)
       - Valutiamo su validazione set
       - Se migliora, salviamo il modello
    5. Early stopping quando non migliora più
    """
    print("Caricamento dati grezzi...")
    clients_raw = load_clients(DATASET_PATH)
    users = list(clients_raw.keys())

    # GroupKFold: divide gli UTENTI (non i singoli esempi) in fold
    # Così uno stesso utente non finisce sia in train che in validation
    gkf = GroupKFold(n_splits=NUM_FOLDS)
    user_idx = np.arange(len(users))
    all_splits = list(gkf.split(X=user_idx, y=user_idx, groups=user_idx))

    # === TRAINING DELL'ENSEMBLE ===
    # Alleniamo 10 modelli, uno per ogni fold
    for fold, (train_idx, val_idx) in enumerate(all_splits):
        # Seed diverso per ogni fold (per riproducibilità)
        fold_seed = SEED + fold
        print(f"\n{'='*40}")
        print(f"Inizio il Fold {fold+1} di {NUM_FOLDS} (Seed: {fold_seed})")
        print(f"{'='*40}")

        # Impostiamo i seed per avere risultati riproducibili
        torch.manual_seed(fold_seed)
        random.seed(fold_seed)
        np.random.seed(fold_seed)

        # Dividiamo gli utenti tra training e validazione
        train_users = [users[i] for i in train_idx]
        val_users = [users[i] for i in val_idx]

        print(f"Utenti per Training: {len(train_users)}, Utenti per Validazione: {len(val_users)}")

        train_clients_raw = {u: clients_raw[u] for u in train_users}
        val_clients_raw = {u: clients_raw[u] for u in val_users}

        # === PARTE 1: CALCOLO SCALER GLOBALE ===
        # Lo calcoliamo comunque perché serve per il test set di Kaggle!
        # Nel test set non conosciamo l'identità dell'utente, quindi usiamo statistiche globali
        print("Calcolo statistiche Globali (per il Test Set)...")
        fed_scaler = FederatedScaler()
        fed_scaler.fit_federated(train_clients_raw, TOP_FEATURES)
        scaler_path = os.path.join(ARTIFACTS_DIR, f"federated_scaler_fold_{fold}.joblib")
        joblib.dump(fed_scaler, scaler_path)

        # === PARTE 2: NORMALIZZAZIONE PER IL TRAINING ===
        # Usiamo User-Wise Scaling: ogni utente è normalizzato rispetto a se stesso
        # Questo insegna al modello a guardare le VARIAZIONI rispetto alla norma personale
        print("Applicazione User-Wise Scaling (Training)...")
        train_clients = transform_clients_user_wise(train_clients_raw, TOP_FEATURES)
        
        # === PARTE 3: NORMALIZZAZIONE PER LA VALIDAZIONE ===
        # Usiamo Global Scaling per simulare lo scenario di Kaggle
        # dove NON conosciamo l'identità dell'utente
        # Questo ci dà una stima realistica di come il modello funzionerà sul test set
        print("Applicazione Global Scaling (Validation - Simulazione Kaggle)...")
        val_clients = transform_clients(val_clients_raw, fed_scaler)

        # Convertiamo i dati di validazione in tensori PyTorch
        val_clients_data = {}
        for u in val_clients:
            Xc, yc = val_clients[u]
            val_clients_data[u] = (torch.tensor(Xc, dtype=torch.float32), yc)

        # Creiamo il modello globale
        model = SleepNet(len(TOP_FEATURES))

        # Variabili per il controllo dell'allenamento
        best_mae = float('inf')     # Miglior errore ottenuto finora
        current_lr = LR              # Learning rate corrente
        no_improve = 0               # Quante volte non è migliorato (per early stop)
        no_improve_lr = 0            # Quante volte non è migliorato (per ridurre LR)

        print(f"Partenza training federato (Fold {fold+1})...")

        # === IL CUORE DEL FEDERATED LEARNING ===
        # Iteriamo per un certo numero di "round" di comunicazione
        for r in range(ROUNDS):
            # Liste per accumulare i modelli dei client
            client_states = []
            client_sizes = []

            # Campionamento casuale dei client
            # Non tutti gli utenti partecipano ad ogni round (solo il 50%)
            # Questo simula la realtà dove non tutti sono sempre disponibili
            num_sampled_clients = max(1, int(len(train_users) * CLIENT_FRACTION))
            sampled_users = random.sample(train_users, num_sampled_clients)

            # FASE 1: TRAINING LOCALE
            # Ogni utente campionato allena il modello sui suoi dati
            for user in sampled_users:
                data = train_clients[user]
                
                # Creiamo una copia del modello globale per questo client
                local_model = SleepNet(len(TOP_FEATURES))
                local_model.load_state_dict(model.state_dict())

                # Il client allena la sua copia localmente
                state, n_samples = train_local(
                    local_model, data, LOCAL_EPOCHS, current_lr
                )
                
                # Raccogliamo i pesi aggiornati
                client_states.append(state)
                client_sizes.append(n_samples)

            # FASE 2: AGGREGAZIONE
            # Il server combina i modelli di tutti i client in uno solo
            fed_avg(model, client_states, client_sizes)
            
            # === VALUTAZIONE ===
            # Ogni 5 round valutiamo il modello sul validation set
            if (r + 1) % 5 == 0:
                model.eval()  # Modalità valutazione
                with torch.no_grad():
                    all_preds = []
                    all_y_true = []
                    
                    # Facciamo previsioni per tutti gli utenti di validazione
                    for u, (X_u, y_u) in val_clients_data.items():
                        preds_u = model(X_u).numpy().flatten() * TARGET_SCALE
                        y_true_u = y_u.flatten() * TARGET_SCALE
                        all_preds.extend(preds_u)
                        all_y_true.extend(y_true_u)
                    
                    # Calcoliamo l'errore medio
                    global_mae = mean_absolute_error(all_y_true, all_preds)
                    
                    print(f"[Fold {fold+1}] Round {r+1} - Errore MAE: {global_mae:.4f}")
                    
                    # Se è il miglior modello finora, lo salviamo
                    if global_mae < best_mae:
                        best_mae = global_mae
                        no_improve = 0
                        no_improve_lr = 0
                        model_path = os.path.join(ARTIFACTS_DIR, f"best_federated_model_fold_{fold}.pt")
                        torch.save(model.state_dict(), model_path)
                        print(f"  -> Nuovo record! Modello salvato (MAE: {global_mae:.4f})")
                    else:
                        # Se non migliora, incrementiamo i contatori
                        no_improve += 1
                        no_improve_lr += 1
                        
                        # Se non migliora da un po', riduciamo il learning rate
                        # (come "rallentare" per essere più precisi)
                        if no_improve_lr >= LR_REDUCE_PATIENCE and current_lr > MIN_LR:
                            current_lr = max(MIN_LR, current_lr * LR_REDUCE_FACTOR)
                            no_improve_lr = 0
                            print(f"  -> Rallento un attimo... (Learning Rate ridotto a {current_lr:.6f})")
                        
                        # Se non migliora da molto tempo, fermiamo l'allenamento
                        # (non ha senso continuare se non impara più nulla)
                        if no_improve >= EARLY_STOP_PATIENCE:
                            print(f"  -> Early stop fold {fold+1} (MAE min: {best_mae:.4f})")
                            break
                
                model.train()  # Torniamo in modalità training

        print(f"Finito Fold {fold+1}. Miglior errore ottenuto: {best_mae:.4f}")

if __name__ == "__main__":
    run_training()
