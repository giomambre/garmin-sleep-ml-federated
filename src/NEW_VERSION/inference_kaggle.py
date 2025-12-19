# inference_kaggle.py
import os
import torch
import pandas as pd
import numpy as np
from model import SleepNet
from data_utils import DROP_COLS
from config import TARGET_SCALE, DATASET_PATH
import joblib

# Carichiamo scaler e medie calcolate sul train per replicare identico preprocessing
bundle = joblib.load("scaler.joblib")
scaler = bundle["scaler"]
feature_means = bundle["feature_means"]
TOP_FEATURES = [
    'act_activeKilocalories',
    'act_totalCalories',
    'resp_avgTomorrowSleepRespirationValue',
    'sleep_remSleepSeconds',
    'act_distance',
    'str_avgStressLevel',
    'sleep_sleepTimeSeconds',
    'sleep_awakeSleepSeconds',
    'hr_maxHeartRate',
    'sleep_deepSleepSeconds',
    'sleep_lightSleepSeconds',
    'sleep_avgSleepStress',
    'hr_lastSevenDaysAvgRestingHeartRate',
    'str_maxStressLevel',
    'hr_minHeartRate',
    'hr_restingHeartRate',
    'sleep_napTimeSeconds',
    'sleep_lowestRespirationValue',
    'sleep_avgHeartRate',
  'resp_highestRespirationValue',
  'resp_avgSleepRespirationValue',
  'resp_avgWakingRespirationValue',
  'resp_lowestRespirationValue'
]
  # stessi del training

model = SleepNet(len(TOP_FEATURES))
model.load_state_dict(torch.load("federated_model.pt"))
model.eval()

test_path = os.path.join(DATASET_PATH, "x_test.csv")
df = pd.read_csv(test_path, sep=';')
df.columns = df.columns.str.strip()
ids = df['day'].values if 'day' in df.columns else np.arange(len(df))
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
df = df.reindex(columns=TOP_FEATURES, fill_value=np.nan)

# Forziamo a float per evitare warning dtype
df[TOP_FEATURES] = df[TOP_FEATURES].astype(float)
# I mancanti nel test sono codificati come -1/-2: convertiamo a NaN e riempiamo con le medie del train
neg_mask = df[TOP_FEATURES] < 0
df.loc[:, TOP_FEATURES] = df[TOP_FEATURES].where(~neg_mask, np.nan)
df = df.fillna(feature_means)
X = torch.tensor(scaler.transform(df.values), dtype=torch.float32)

with torch.no_grad():
    preds = model(X).numpy().flatten() * TARGET_SCALE

preds = np.clip(preds, 0, 100).round().astype(int)

submission = pd.DataFrame({'id': ids, 'label': preds})
submission.to_csv("submission.csv", index=False)
