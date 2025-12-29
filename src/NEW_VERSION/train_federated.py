# train_federated.py
import torch
import random
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from config import SEED, DATASET_PATH, ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, LR, HIDDEN_1, HIDDEN_2, HIDDEN_3, DROPOUT, TARGET_SCALE, CLIENT_FRACTION, TOP_FEATURES
from model import SleepNet
from data_utils import load_clients, FederatedScaler, transform_clients
from client import train_local
from server import fed_avg

# Seed
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("Loading data...")
clients_raw = load_clients(DATASET_PATH)

import torch.optim as optim

# Split a livello utente PRIMA del preprocessing per evitare data leakage
users = list(clients_raw.keys())
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
user_idx = np.arange(len(users))
train_idx, val_idx = next(gss.split(user_idx, user_idx, groups=user_idx))
train_users = [users[i] for i in train_idx]
val_users = [users[i] for i in val_idx]

print(f"Train users: {len(train_users)}, Validation users: {len(val_users)}")

# Separiamo i dati raw
train_clients_raw = {u: clients_raw[u] for u in train_users}
val_clients_raw = {u: clients_raw[u] for u in val_users}

# --- FEDERATED STATISTICS PHASE ---
# Calcoliamo le statistiche globali (media, std) aggregando i parziali dei client.
# Nessun dato raw viene condiviso o unito.
print("Computing Federated Statistics (Mean/Std)...")
fed_scaler = FederatedScaler()
fed_scaler.fit_federated(train_clients_raw, TOP_FEATURES)

# Salviamo lo scaler federato per l'inferenza
joblib.dump(fed_scaler, "federated_scaler.joblib")

# Trasformiamo i dati localmente usando le statistiche globali appena calcolate
print("Transforming data locally...")
train_clients = transform_clients(train_clients_raw, fed_scaler)
val_clients = transform_clients(val_clients_raw, fed_scaler)

# Costruiamo il validation set per utente per calcolare metriche per bucket
val_clients_data = {}
for u in val_clients:
    Xc, yc = val_clients[u]
    val_clients_data[u] = (torch.tensor(Xc, dtype=torch.float32), yc)

model = SleepNet(len(TOP_FEATURES))

print("Start Federated Training...")
best_mae = float('inf')
current_lr = LR
patience = 0

for r in range(ROUNDS):
    client_states = []
    client_sizes = []

    # Seleziona un sottoinsieme casuale di client per questo round
    num_sampled_clients = max(1, int(len(train_users) * CLIENT_FRACTION))
    sampled_users = random.sample(train_users, num_sampled_clients)

    for user in sampled_users:
        data = train_clients[user]
        # Ogni client riceve il modello globale corrente come punto di partenza
        local_model = SleepNet(len(TOP_FEATURES))
        local_model.load_state_dict(model.state_dict())

        state, n_samples = train_local(
            local_model, data, LOCAL_EPOCHS, current_lr
        )
        client_states.append(state)
        client_sizes.append(n_samples)

    fed_avg(model, client_states, client_sizes)
    
    # Validazione periodica (ogni 5 round)
    if (r + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_y_true = []
            mae_per_user = []
            
            for u, (X_u, y_u) in val_clients_data.items():
                preds_u = model(X_u).numpy().flatten() * TARGET_SCALE
                y_true_u = y_u.flatten() * TARGET_SCALE
                mae_u = mean_absolute_error(y_true_u, preds_u)
                mae_per_user.append(mae_u)
                all_preds.extend(preds_u)
                all_y_true.extend(y_true_u)
            
            global_mae = mean_absolute_error(all_y_true, all_preds)
            global_r2 = r2_score(all_y_true, all_preds)
            avg_mae_per_user = np.mean(mae_per_user)
            
            print(f"Round {r+1}/{ROUNDS} - Global MAE: {global_mae:.4f}, R²: {global_r2:.4f}, Avg MAE per user: {avg_mae_per_user:.4f}")
            
            if global_mae < best_mae:
                best_mae = global_mae
                patience = 0
                torch.save(model.state_dict(), "best_federated_model.pt")
                print(f"  -> New best model saved! (MAE: {global_mae:.4f})")
            else:
                patience += 1
                if patience >= 2: # Reduce after 2 checks (10 rounds) without improvement
                    current_lr *= 0.5
                    patience = 0
                    print(f"  -> Reducing Learning Rate to {current_lr:.6f}")

        model.train()
    else:
        print(f"Round {r+1}/{ROUNDS} completed")

print(f"Training finished. Best Validation MAE: {best_mae:.4f}")
# Salviamo anche l'ultimo modello per completezza
torch.save(model.state_dict(), "last_federated_model.pt")
