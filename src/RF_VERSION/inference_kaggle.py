import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import BASE_DIR, TOP_FEATURES, DATASET_DIR
from data_utils import FederatedScaler, extract_time_series_features, _clean_missing

def main():
    print("--- Starting Random Forest Inference ---")
    
    # 1. Paths
    # Risaliamo alla root del progetto per trovare x_test.csv
    # DATASET_DIR punta a CSV_train, noi vogliamo x_test che è un livello sopra
    # src/DATASET/CSV_train -> src/DATASET/x_test.csv
    dataset_folder = os.path.dirname(DATASET_DIR) 
    # fix per il doppio annidamento se presente, o hardcoded path relativo
    # Proviamo path assoluto basato su struttura nota
    project_root = os.path.dirname(os.path.dirname(BASE_DIR)) # CSI project
    test_path = os.path.join(project_root, "src", "DATASET", "x_test.csv")
    
    model_path = os.path.join(BASE_DIR, "federated_rf_model.joblib")
    scaler_path = os.path.join(BASE_DIR, "federated_scaler_rf.joblib")

    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        return

    # 2. Load Resources
    print(f"Loading model from {model_path}...")
    rf_model = joblib.load(model_path)
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    
    # 3. Load & Preprocess Test Data
    print("Loading test data...")
    df_test = pd.read_csv(test_path, sep=';') # Attenzione al separatore, solitamente ; o ,
    
    # Se il csv ha problemi di separatore (es. tutto in una colonna), riproviamo
    if df_test.shape[1] < 5:
        df_test = pd.read_csv(test_path, sep=',')
    
    print(f"Test shape raw: {df_test.shape}")
    df_test.columns = df_test.columns.str.strip()

    # Estrazione feature time series (fondamentale!)
    df_test = extract_time_series_features(df_test)
    
    # 4. Transform using Federated Scaler
    # Lo scaler gestisce internamente imputazione e selezione feature
    # Ma dobbiamo assicurarci che lo scaler sia stato fittato
    print("Transforming test data...")
    try:
        X_test = scaler.transform(df_test)
    except Exception as e:
        print(f"Error during scaling: {e}")
        # Fallback debug
        return

    # 5. Predict
    print("Predicting...")
    y_pred_scaled = rf_model.predict(X_test)
    
    # 6. Post-process (Rescale to 0-100)
    # Nel training avevamo diviso per 100?
    # In data_utils.py: y = (df[LABEL_COL].values / 100.0)
    # Quindi qui dobbiamo moltiplicare per 100
    y_pred = y_pred_scaled * 100.0
    
    # Clip 0-100
    y_pred = np.clip(y_pred, 0, 100)
    y_pred = np.round(y_pred).astype(int) # Kaggle vuole interi di solito? O float?
    # Nel dubbio lasciamo float o int? Il sample submission solitamente ha interi o float.
    # New version fa round. Facciamo round.

    # 7. Create Submission File
    submission = pd.DataFrame({
        'id': df_test.index, # O una colonna ID se esiste
        'sleep_score': y_pred
    })
    
    # Se esiste una colonna id nel test file, usiamola
    if 'id' in df_test.columns:
        submission['id'] = df_test['id']
    elif 'Id' in df_test.columns:
        submission['id'] = df_test['Id']
    
    output_path = os.path.join(project_root, "submission_rf.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print(submission.head())

if __name__ == "__main__":
    main()
