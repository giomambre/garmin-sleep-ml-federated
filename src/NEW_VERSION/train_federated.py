# train_federated.py
import torch
import random
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from config import *
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

# usa top 25 feature RF 
TOP_FEATURES = [
    'act_activeKilocalories', 'act_totalCalories',
    'resp_avgTomorrowSleepRespirationValue',
    'sleep_remSleepSeconds', 'act_distance',
    'str_avgStressLevel', 'sleep_sleepTimeSeconds',
    'sleep_awakeSleepSeconds', 'hr_maxHeartRate',
    'sleep_deepSleepSeconds', 'sleep_lightSleepSeconds',
    'sleep_avgSleepStress', 'hr_lastSevenDaysAvgRestingHeartRate',
    'str_maxStressLevel', 'hr_minHeartRate',
    'hr_restingHeartRate', 'sleep_napTimeSeconds',
    'sleep_lowestRespirationValue', 'sleep_avgHeartRate',
    'resp_highestRespirationValue',
    'resp_avgSleepRespirationValue', 'resp_avgWakingRespirationValue',
    'resp_lowestRespirationValue'
]

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

# Costruiamo il validation set concatenato per la valutazione (solo per monitoraggio)
X_val, y_val = [], []
for u in val_clients:
    Xc, yc = val_clients[u]
    X_val.append(Xc)
    y_val.append(yc)

X_val = np.concatenate(X_val, axis=0)
y_val = np.concatenate(y_val, axis=0)
X_val_torch = torch.tensor(X_val, dtype=torch.float32)

model = SleepNet(len(TOP_FEATURES))

print("Start Federated Training...")
best_mae = float('inf')

for r in range(ROUNDS):
    client_states = []
    client_sizes = []

    for user, data in train_clients.items():
        # Ogni client riceve il modello globale corrente come punto di partenza
        local_model = SleepNet(len(TOP_FEATURES))
        local_model.load_state_dict(model.state_dict())

        state, n_samples = train_local(
            local_model, data, LOCAL_EPOCHS, LR
        )
        client_states.append(state)
        client_sizes.append(n_samples)

    fed_avg(model, client_states, client_sizes)
    
    # Validazione periodica (ogni 5 round)
    if (r + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            preds = model(X_val_torch).numpy().flatten() * TARGET_SCALE
            y_true = y_val.flatten() * TARGET_SCALE
            mae = mean_absolute_error(y_true, preds)
            print(f"Round {r+1}/{ROUNDS} - Val MAE: {mae:.4f}")
            
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), "best_federated_model.pt")
                print(f"  -> New best model saved! (MAE: {mae:.4f})")
        model.train()
    else:
        print(f"Round {r+1}/{ROUNDS} completed")

print(f"Training finished. Best Validation MAE: {best_mae:.4f}")
# Salviamo anche l'ultimo modello per completezza
torch.save(model.state_dict(), "last_federated_model.pt")
