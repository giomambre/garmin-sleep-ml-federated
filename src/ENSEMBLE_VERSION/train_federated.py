# train_federated.py
import torch
import random
import numpy as np
import joblib
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from config import SEED, DATASET_PATH, ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, LR, HIDDEN_1, HIDDEN_2, HIDDEN_3, DROPOUT, TARGET_SCALE, CLIENT_FRACTION, NUM_FOLDS, TOP_FEATURES, ARTIFACTS_DIR
from model import SleepNet
from data_utils import load_clients, FederatedScaler, transform_clients
from client import train_local
from server import fed_avg
import os

def run_training():
    # Carichiamo i dati grezzi dal disco
    print("Loading raw data...")
    clients_raw = load_clients(DATASET_PATH)
    users = list(clients_raw.keys())

    # Qui uso GroupKFold per dividere gli utenti in 5 gruppi.
    # Serve per fare Cross-Validation: alleno su 4 gruppi e testo sul 5°.
    gkf = GroupKFold(n_splits=NUM_FOLDS)
    user_idx = np.arange(len(users))
    all_splits = list(gkf.split(X=user_idx, y=user_idx, groups=user_idx))

    # --- INIZIO IL TRAINING DELL'ENSEMBLE ---
    for fold, (train_idx, val_idx) in enumerate(all_splits):
        fold_seed = SEED + fold
        print(f"\n{'='*40}")
        print(f"Inizio il Fold {fold+1} di {NUM_FOLDS} (Seed: {fold_seed})")
        print(f"{'='*40}")

        # Fisso i seed così i risultati sono riproducibili
        torch.manual_seed(fold_seed)
        random.seed(fold_seed)
        np.random.seed(fold_seed)

        # Recupero gli utenti per questo giro di training e validazione
        train_users = [users[i] for i in train_idx]
        val_users = [users[i] for i in val_idx]

        print(f"Utenti per Training: {len(train_users)}, Utenti per Validazione: {len(val_users)}")

        # Preparo i dizionari dei dati
        train_clients_raw = {u: clients_raw[u] for u in train_users}
        val_clients_raw = {u: clients_raw[u] for u in val_users}

        # --- FASE 1: CALCOLO STATISTICHE GLOBALI (PRIVACY-PRESERVING) ---
        # Prima di allenare, devo capire come scalare i dati (media e varianza).
        # Lo faccio in modo federato: chiedo le somme ai client senza vedere i dati.
        print("Calcolo le statistiche federate (Media/Std) per normalizzare i dati...")
        fed_scaler = FederatedScaler()
        fed_scaler.fit_federated(train_clients_raw, TOP_FEATURES)

        # Salvo lo scaler di questo fold per usarlo dopo
        scaler_path = os.path.join(ARTIFACTS_DIR, f"federated_scaler_fold_{fold}.joblib")
        joblib.dump(fed_scaler, scaler_path)

        # Ora ogni client si normalizza i dati in casa sua usando le medie globali
        train_clients = transform_clients(train_clients_raw, fed_scaler)
        val_clients = transform_clients(val_clients_raw, fed_scaler)

        # Preparo i dati di validazione (mi servono per calcolare l'errore ogni tanto)
        val_clients_data = {}
        for u in val_clients:
            Xc, yc = val_clients[u]
            val_clients_data[u] = (torch.tensor(Xc, dtype=torch.float32), yc)

        # Creo il modello "vuoto" per questo fold
        model = SleepNet(len(TOP_FEATURES))

        best_mae = float('inf')
        current_lr = LR
        patience = 0

        print(f"Partenza training federato (Fold {fold+1})...")

        for r in range(ROUNDS):
            client_states = []
            client_sizes = []

            # Non uso tutti i client ogni volta, ne pesco a caso la metà (simula la realtà)
            num_sampled_clients = max(1, int(len(train_users) * CLIENT_FRACTION))
            sampled_users = random.sample(train_users, num_sampled_clients)

            for user in sampled_users:
                data = train_clients[user]
                # Copio il modello globale e lo mando al client
                local_model = SleepNet(len(TOP_FEATURES))
                local_model.load_state_dict(model.state_dict())

                # Il client si allena per un po' sui suoi dati
                state, n_samples = train_local(
                    local_model, data, LOCAL_EPOCHS, current_lr
                )
                client_states.append(state)
                client_sizes.append(n_samples)

            # Il server raccoglie tutto e fa la media pesata dei nuovi modelli
            fed_avg(model, client_states, client_sizes)
            
            # Ogni 5 round controllo come stiamo andando
            if (r + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    all_preds = []
                    all_y_true = []
                    
                    # Faccio previsioni su tutti gli utenti di validazione
                    for u, (X_u, y_u) in val_clients_data.items():
                        preds_u = model(X_u).numpy().flatten() * TARGET_SCALE
                        y_true_u = y_u.flatten() * TARGET_SCALE
                        all_preds.extend(preds_u)
                        all_y_true.extend(y_true_u)
                    
                    global_mae = mean_absolute_error(all_y_true, all_preds)
                    
                    print(f"[Fold {fold+1}] Round {r+1} - Errore MAE: {global_mae:.4f}")
                    
                    # Se il modello è migliorato, lo salvo!
                    if global_mae < best_mae:
                        best_mae = global_mae
                        patience = 0
                        model_path = os.path.join(ARTIFACTS_DIR, f"best_federated_model_fold_{fold}.pt")
                        torch.save(model.state_dict(), model_path)
                        print(f"  -> Nuovo record! Modello salvato (MAE: {global_mae:.4f})")
                    else:
                        # Se non migliora per un po', abbasso il learning rate per andare più piano
                        patience += 1
                        if patience >= 2:
                            current_lr *= 0.5
                            patience = 0
                            print(f"  -> Rallento un attimo... (Learning Rate ridotto a {current_lr:.6f})")
                model.train()

        print(f"Finito Fold {fold+1}. Miglior errore ottenuto: {best_mae:.4f}")

if __name__ == "__main__":
    run_training()