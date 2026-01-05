# inference_kaggle.py
import os
import torch
import pandas as pd
import numpy as np
from model import SleepNet
from data_utils import DROP_COLS, FederatedScaler, extract_time_series_features
from config import TARGET_SCALE, BASE_DIR, TOP_FEATURES
import joblib

# Carichiamo lo scaler federato (contiene medie e std globali)
fed_scaler = joblib.load("federated_scaler.joblib")

model = SleepNet(len(TOP_FEATURES))
# Carichiamo il modello migliore salvato durante il training
model.load_state_dict(torch.load("best_federated_model.pt"))
model.eval()

test_path = os.path.join(BASE_DIR, "..", "DATASET", "x_test.csv")
df = pd.read_csv(test_path, sep=';')
df.columns = df.columns.str.strip()

# Save IDs for submission
if 'id' in df.columns:
    ids = df['id'].values
else:
    ids = np.arange(len(df))

# Extract TS features BEFORE dropping columns
df = extract_time_series_features(df)

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Usiamo il metodo transform dello scaler federato
# Questo gestisce internamente: reindex, pulizia -1/-2, imputazione con media globale, scaling con std globale
X_np = fed_scaler.transform(df)
X = torch.tensor(X_np, dtype=torch.float32)

with torch.no_grad():
    preds = model(X).numpy().flatten() * TARGET_SCALE

preds = np.clip(preds, 0, 100).round().astype(int)

submission = pd.DataFrame({'id': ids, 'label': preds})
submission.to_csv("submission.csv", index=False)
