import sys
import os

# Add current directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import joblib

# Adjusted imports (removed relative dots)
from config import (
    DATASET_DIR, TOP_FEATURES, LABEL_COL, ROUNDS, K_FOLDS, RANDOM_SEED, BASE_DIR
)
from data_utils import load_clients, FederatedScaler, transform_clients
from client import FederatedRFClient
from server import FederatedRFServer

def main():
    print("--- Loading Data ---")
    clients_data = load_clients(DATASET_DIR)
    print(f"Loaded {len(clients_data)} clients.")

    # --- Cross Validation Setup ---
    all_users = list(clients_data.keys())
    # Creiamo un array fittizio X e gruppi=users per usare GroupKFold/ShuffleSplit logic
    # Ma qui replichiamo la logica NEW_VERSION: 1 solo fold principale di training/val
    # O facciamo K-Fold completo? NEW_VERSION faceva train su 80% e val su 20%.
    # Seguiamo lo stesso approccio per consistenza: 1 Split 80/20.
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    # Dummy arrays per splitting
    dummy_X = np.zeros((len(all_users), 1))
    train_idx_users, val_idx_users = next(gss.split(dummy_X, groups=all_users))
    
    train_users = [all_users[i] for i in train_idx_users]
    val_users = [all_users[i] for i in val_idx_users]
    
    print(f"Train Users: {len(train_users)}")
    print(f"Val Users: {len(val_users)}")

    # Separiamo i dizionari
    train_clients_data = {u: clients_data[u] for u in train_users}
    val_clients_data = {u: clients_data[u] for u in val_users}

    # --- Federated Scaling ---
    print("--- Fitting Federated Scaler ---")
    scaler = FederatedScaler()
    scaler.fit_federated(train_clients_data, TOP_FEATURES)
    
    # Salviamo lo scaler
    scaler_path = os.path.join(BASE_DIR, "federated_scaler_rf.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # --- Transform Data ---
    print("--- Transforming Data ---")
    train_processed = transform_clients(train_clients_data, scaler)
    val_processed = transform_clients(val_clients_data, scaler)

    # --- Inizio Training Federato (Random Forest) ---
    server = FederatedRFServer()
    
    # A differenza delle reti neurali, con Random Forest non serve fare tanti "round" di avanti e indietro.
    # Ogni client si allena una volta sola sui suoi dati e manda i suoi alberi al server.
    # Il server poi mette tutto insieme (Bagging).
    
    client_estimators_bucket = []
    
    print("Inizio l'addestramento locale sui client...")
    for i, user in enumerate(train_users):
        X_train, y_train = train_processed[user]
        
        # Controllo di sicurezza: se un client non ha dati, lo salto
        if X_train is None or len(X_train) == 0 or y_train is None:
            continue
            
        # Appiattisco l'array delle label perché sklearn lo vuole così
        y_train = y_train.ravel()
        
        client = FederatedRFClient(user, X_train, y_train)
        
        # Il client costruisce la sua piccola foresta locale
        local_trees = client.fit()
        client_estimators_bucket.append(local_trees)

    # --- Aggregazione al Server ---
    print("--- Il server sta raccogliendo tutti gli alberi... ---")
    server.aggregate(client_estimators_bucket)
    
    # --- Creazione Modello Globale ---
    # Ora che ho tutti gli alberi, creo un unico oggetto Random Forest gigante
    # che potrò usare per fare previsioni su chiunque.
    sample_user = train_users[0]
    X_sample, y_sample = train_processed[sample_user]
    global_rf = server.get_global_model(X_sample, y_sample.ravel())
    
    # Salvo il modello su disco
    model_path = os.path.join(BASE_DIR, "federated_rf_model.joblib")
    joblib.dump(global_rf, model_path)
    print(f"Foresta Globale salvata in: {model_path}")
    
    # --- Validation ---
    print("--- Validating ---")
    mae_list = []
    
    for user in val_users:
        X_val, y_val = val_processed[user]
        if X_val is None or len(X_val) == 0:
            continue
            
        y_val = y_val.ravel()
        
        # Predizione
        y_pred_scaled = global_rf.predict(X_val)
        
        # Riportiamo alla scala originale (0-100) per calcolare MAE
        # y_val era diviso per 100 in data_utils, quindi moltiplichiamo
        y_true_orig = y_val * 100.0
        y_pred_orig = y_pred_scaled * 100.0
        
        # Clip predizioni 0-100
        y_pred_orig = np.clip(y_pred_orig, 0, 100)
        
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        mae_list.append(mae)
        
    print(f"Validation MAE (per user avg): {np.mean(mae_list):.4f}")
    print(f"Validation MAE (std): {np.std(mae_list):.4f}")

if __name__ == "__main__":
    main()
