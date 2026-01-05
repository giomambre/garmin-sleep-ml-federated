import os
import sys
import joblib
import torch
import pandas as pd
import numpy as np

# Add src to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir) # Add src/
sys.path.append(os.path.join(current_dir, 'ENSEMBLE_VERSION')) # Add src/ENSEMBLE_VERSION

from ENSEMBLE_VERSION.model import SleepNet
from ENSEMBLE_VERSION.config import TOP_FEATURES, TARGET_SCALE, BASE_DIR, NUM_FOLDS, ARTIFACTS_DIR
from ENSEMBLE_VERSION.data_utils import extract_time_series_features, DROP_COLS

# Qui ho inserito i risultati MAE (Mean Absolute Error) che ho ottenuto durante il training dei 5 fold.
# Fold 3 è stato il migliore (10.12), quindi voglio che conti di più nella media finale!
FOLD_MAES = [11.4472, 10.6274, 11.2824, 10.1237, 10.7091]

FOLD_MAES = [11.4472, 10.6274, 11.2824, 10.1237, 10.7091]

def run_weighted_inference():
    print("Genero la submission pesata (meritocratica)...")
    
    # Calcolo i pesi: uso l'inverso dell'errore.
    # Chi ha sbagliato meno (errore piccolo) avrà un peso più grande.
    # Matematica: Peso = (1 / Errore) / SommaDiTutti
    inv_maes = [1.0 / m for m in FOLD_MAES]
    sum_inv_maes = sum(inv_maes)
    weights = [w / sum_inv_maes for w in inv_maes]
    
    print("Ecco quanto conta ogni modello nella decisione finale:")
    for i, w in enumerate(weights):
        print(f"  Fold {i} (Errore {FOLD_MAES[i]:.4f}): {w*100:.2f}%")
    
    test_path = os.path.join(BASE_DIR, "..", "DATASET", "x_test.csv")
    df_orig = pd.read_csv(test_path, sep=';')
    df_orig.columns = df_orig.columns.str.strip()

    if 'id' in df_orig.columns:
        ids = df_orig['id'].values
    else:
        ids = np.arange(len(df_orig))

    print("Extracting features...")
    df_features = extract_time_series_features(df_orig.copy())
    df_features = df_features.drop(columns=[c for c in DROP_COLS if c in df_features.columns])

    # Accumulator
    weighted_preds = np.zeros(len(df_features))

    for fold in range(NUM_FOLDS):
        weight = weights[fold]
        print(f"Processing Fold {fold} with weight {weight:.4f}...")
        
        # Load Scaler
        scaler_path = os.path.join(ARTIFACTS_DIR, f"federated_scaler_fold_{fold}.joblib")
        fed_scaler = joblib.load(scaler_path)
        
        X_np = fed_scaler.transform(df_features)
        X = torch.tensor(X_np, dtype=torch.float32)

        # Load Model
        model_path = os.path.join(ARTIFACTS_DIR, f"best_federated_model_fold_{fold}.pt")
        model = SleepNet(len(TOP_FEATURES))
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            preds = model(X).numpy().flatten() * TARGET_SCALE
            # Add weighted prediction
            weighted_preds += (preds * weight)
            
    # Finalize
    weighted_preds = np.clip(weighted_preds, 0, 100).round().astype(int)

    submission_path = os.path.join(ARTIFACTS_DIR, "submission_weighted.csv")
    submission = pd.DataFrame({'id': ids, 'label': weighted_preds})
    submission.to_csv(submission_path, index=False)
    print(f"Weighted Ensemble saved to '{submission_path}'.")
    print(submission.head())

if __name__ == "__main__":
    run_weighted_inference()
