import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
from config import *
from data_utils import load_and_clean
from client import LocalClient
from server import FederatedServer

if __name__ == "__main__":
    print("--- AVVIO TRAINING FEDERATO (Random Forest) ---")

    print("1. Caricamento e pulizia dati...")
    # Carica i dati usando la funzione pulita in data_utils
    all_clients = load_and_clean(DATASET_PATH, TOP_FEATURES)

    if not all_clients:
        print("ERRORE: Nessun dato caricato. Controlla il percorso in config.py")
        exit()

    # Split Utenti
    u_ids = list(all_clients.keys())
    np.random.seed(SEED)
    np.random.shuffle(u_ids)
    split = int(len(u_ids) * 0.8)
    train_ids = u_ids[:split]
    val_ids = u_ids[split:]

    print(f"   Utenti Train: {len(train_ids)} | Utenti Validation: {len(val_ids)}")

    # FASE 1: Training Locale
    print("2. Training Locale sui Client...")
    forest_depot = []
    for uid in train_ids:
        X, y = all_clients[uid]
        if len(X) > 0:
            client = LocalClient(X, y)
            trees = client.train()
            forest_depot.append(trees)

    # FASE 2: Aggregazione
    print(f"3. Aggregazione di {len(forest_depot)} foreste locali...")
    server = FederatedServer(TOP_FEATURES)
    server.aggregate(forest_depot)

    # FASE 3: Validazione
    print("4. Calcolo MAE...")
    maes = []
    for uid in val_ids:
        X_v, y_v = all_clients[uid]
        if len(X_v) > 0:
            preds = server.predict(X_v)
            mae = mean_absolute_error(y_v, preds)
            maes.append(mae)

    print(f"\n>>> RISULTATO MAE: {np.mean(maes):.4f} <<<")

    # Salvataggio
    joblib.dump(server.global_rf, "rf_federated_model.joblib")
    print("Modello salvato.")