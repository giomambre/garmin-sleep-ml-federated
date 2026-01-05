# inference_kaggle.py
import os
import torch
import pandas as pd
import numpy as np
import joblib
from model import SleepNet
from data_utils import DROP_COLS, FederatedScaler, extract_time_series_features
from config import TARGET_SCALE, BASE_DIR, NUM_FOLDS, TOP_FEATURES, ARTIFACTS_DIR

def run_inference():
    print(f"Starting Ensemble Inference with {NUM_FOLDS} models...")
    
    test_path = os.path.join(BASE_DIR, "..", "DATASET", "x_test.csv")
    df_orig = pd.read_csv(test_path, sep=';')
    df_orig.columns = df_orig.columns.str.strip()

    # Save IDs for submission
    if 'id' in df_orig.columns:
        ids = df_orig['id'].values
    else:
        ids = np.arange(len(df_orig))

    # Feature Extraction (Deterministic - can be done once)
    print("Extracting Time Series features...")
    df_features = extract_time_series_features(df_orig.copy())
    
    # Drop unused columns (except those needed by scaler if any logic requires them, but here we just drop DROP_COLS)
    df_features = df_features.drop(columns=[c for c in DROP_COLS if c in df_features.columns])

    # Accumulator for predictions
    final_preds = np.zeros(len(df_features))

    for fold in range(NUM_FOLDS):
        print(f"Processing Fold {fold}...")
        
        # Load Fold Scaler
        scaler_path = os.path.join(ARTIFACTS_DIR, f"federated_scaler_fold_{fold}.joblib")
        if not os.path.exists(scaler_path):
            print(f"Warning: {scaler_path} not found. Skipping this fold.")
            continue
            
        fed_scaler = joblib.load(scaler_path)
        
        # Scale Data (Fold specific)
        X_np = fed_scaler.transform(df_features)
        X = torch.tensor(X_np, dtype=torch.float32)

        # Load Fold Model
        model_path = os.path.join(ARTIFACTS_DIR, f"best_federated_model_fold_{fold}.pt")
        if not os.path.exists(model_path):
             print(f"Warning: {model_path} not found. Skipping this fold.")
             continue

        model = SleepNet(len(TOP_FEATURES))
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            preds = model(X).numpy().flatten() * TARGET_SCALE
            final_preds += preds
            
    # Average predictions
    avg_preds = final_preds / NUM_FOLDS
    
    # Round and clip
    avg_preds = np.clip(avg_preds, 0, 100).round().astype(int)

    submission_path = os.path.join(ARTIFACTS_DIR, "submission_ensemble.csv")
    submission = pd.DataFrame({'id': ids, 'label': avg_preds})
    submission.to_csv(submission_path, index=False)
    print(f"Ensemble inference complete. Saved to '{submission_path}'.")

if __name__ == "__main__":
    run_inference()
    