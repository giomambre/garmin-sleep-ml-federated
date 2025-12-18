# train_federated.py
import torch
import random
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from config import *
from model import SleepNet
from data_utils import load_clients, prepare_data
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

clients, scaler = prepare_data(clients_raw, TOP_FEATURES)

# Costruiamo un validation set per utente: tutti i record dello stesso utente
# restano insieme nello stesso split. Così il modello viene testato su utenti
# mai visti in training, scenario più vicino al test Kaggle.
all_X, all_y, groups = [], [], []
for user, (Xc, yc) in clients.items():
    all_X.append(Xc)
    all_y.append(yc)
    groups.extend([user] * len(Xc))

all_X = np.concatenate(all_X, axis=0)
all_y = np.concatenate(all_y, axis=0)
groups = np.array(groups)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(gss.split(all_X, all_y, groups))
X_val, y_val = all_X[val_idx], all_y[val_idx]

model = SleepNet(len(TOP_FEATURES))

print("Start Federated Training...")
for r in range(ROUNDS):
    client_states = []
    client_sizes = []

    for user, data in clients.items():
        # Ogni client riceve il modello globale corrente come punto di partenza
        local_model = SleepNet(len(TOP_FEATURES))
        local_model.load_state_dict(model.state_dict())

        state, n_samples = train_local(
            local_model, data, LOCAL_EPOCHS, LR
        )
        client_states.append(state)
        client_sizes.append(n_samples)

    fed_avg(model, client_states, client_sizes)
    print(f"Round {r+1}/{ROUNDS} completed")

# MAE sul validation per utente: misura l'errore su utenti esclusi dal training,
# quindi è una stima più severa e più vicina al comportamento sul test reale.
model.eval()
with torch.no_grad():
    Xv = torch.tensor(X_val, dtype=torch.float32)
    preds = model(Xv).numpy().flatten() * TARGET_SCALE
    y_true = y_val.flatten() * TARGET_SCALE
    mae = mean_absolute_error(y_true, preds)
    print(f"Hold-out MAE: {mae:.4f}")

torch.save(model.state_dict(), "federated_model.pt")
print("Training finished.")
