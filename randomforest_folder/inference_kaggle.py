import pandas as pd
import numpy as np
import joblib
import os
from data_utils import extract_ts_features
from config import TOP_FEATURES, MODEL_PATH, DATASET_PATH, TARGET_SCALE

def run_inference():
    print("Caricamento modello...")
    model = joblib.load("rf_federated_model.joblib")
    
    df_test = pd.read_csv(os.path.join(DATASET_PATH, "x_test.csv"), sep=';')
    df_test.columns = df_test.columns.str.strip()
    ids = df_test['day'].values

    # Preprocessing identico al training
    for col, prefix in [('hr_time_series', 'hr'), ('resp_time_series', 'resp')]:
        if col in df_test.columns:
            stats = df_test[col].apply(extract_ts_features)
            df_test[f'{prefix}_ts_mean'] = stats.apply(lambda x: x[0])
            df_test[f'{prefix}_ts_std'] = stats.apply(lambda x: x[1])

    hr_now = pd.to_numeric(df_test['hr_restingHeartRate'], errors='coerce').replace(-1, np.nan)
    hr_hist = pd.to_numeric(df_test['hr_lastSevenDaysAvgRestingHeartRate'], errors='coerce').replace(-1, np.nan)
    df_test['hr_delta_resting'] = hr_now - hr_hist

    X_test = df_test.reindex(columns=TOP_FEATURES).replace([-1, -2], np.nan)
    X_test = X_test.fillna(X_test.mean()).fillna(0)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 100).round().astype(int)

    submission = pd.DataFrame({'id': ids, 'label': preds})
    submission.to_csv("submission_rf.csv", index=False)
    print("Fine! File submission_rf.csv creato.")

if __name__ == "__main__":
    run_inference()