# evaluate_cv.py
# Questo file valuta le performance dei modelli allenati con cross-validation
# È come un "esame" per vedere quanto bene funzionano i modelli

import os
import json
import torch
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

from config import DATASET_PATH, NUM_FOLDS, TOP_FEATURES, ARTIFACTS_DIR, TARGET_SCALE, SEED
from data_utils import load_clients, transform_clients
from model import SleepNet


def evaluate_fold(fold, users, clients_raw, splits):
    """
    Valuta un singolo fold della cross-validation.
    
    Cosa fa?
    1. Carica il modello e lo scaler salvati per questo fold
    2. Fa previsioni sui dati di validazione
    3. Calcola metriche di errore (MAE, MedAE, R²)
    4. Calcola l'errore per ogni singolo utente
    
    Restituisce tutte queste metriche in un dizionario.
    """
    train_idx, val_idx = splits[fold]
    val_users = [users[i] for i in val_idx]
    val_clients_raw = {u: clients_raw[u] for u in val_users}

    # Percorsi dei file salvati
    scaler_path = os.path.join(ARTIFACTS_DIR, f"federated_scaler_fold_{fold}.joblib")
    model_path = os.path.join(ARTIFACTS_DIR, f"best_federated_model_fold_{fold}.pt")

    # Se non esistono, questo fold non è stato allenato
    if not (os.path.exists(scaler_path) and os.path.exists(model_path)):
        return None

    # Carichiamo lo scaler e trasformiamo i dati di validazione
    fed_scaler = joblib.load(scaler_path)
    val_clients = transform_clients(val_clients_raw, fed_scaler)

    # Carichiamo il modello allenato
    model = SleepNet(len(TOP_FEATURES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()  # Modalità valutazione (disabilita dropout)

    all_preds = []
    all_true = []
    per_user_mae = {}

    # Facciamo previsioni per ogni utente
    with torch.no_grad():  # Non calcoliamo gradienti (più veloce)
        for u, (X_u, y_u) in val_clients.items():
            X = torch.tensor(X_u, dtype=torch.float32)
            preds = model(X).numpy().flatten() * TARGET_SCALE  # Riportiamo alla scala originale
            y_true = y_u.flatten() * TARGET_SCALE
            all_preds.extend(preds)
            all_true.extend(y_true)
            # Errore specifico per questo utente
            per_user_mae[u] = mean_absolute_error(y_true, preds)

    # Calcoliamo le metriche globali
    mae = mean_absolute_error(all_true, all_preds)      # Errore assoluto medio
    medae = median_absolute_error(all_true, all_preds)  # Errore assoluto mediano
    r2 = r2_score(all_true, all_preds)                  # R²: quanto bene il modello spiega i dati

    return {
        "mae": mae,
        "medae": medae,
        "r2": r2,
        "per_user_mae": per_user_mae,
    }


def main():
    """
    Funzione principale che valuta tutti i fold e produce un report.
    
    Workflow:
    1. Carica i dati
    2. Crea gli split per cross-validation
    3. Valuta ogni fold
    4. Calcola statistiche aggregate (media e deviazione standard)
    5. Salva tutto in un file JSON
    """
    print("Caricamento dati grezzi...")
    clients_raw = load_clients(DATASET_PATH)
    users = list(clients_raw.keys())
    user_idx = np.arange(len(users))
    
    # GroupShuffleSplit: divide gli utenti (non i singoli esempi) in train/validation
    # Usiamo lo stesso split 70/30 usato in fase di training
    gss = GroupShuffleSplit(n_splits=NUM_FOLDS, test_size=0.3, random_state=SEED)
    splits = list(gss.split(user_idx, user_idx, groups=user_idx))

    report = {}
    fold_metrics = []

    # Valutiamo ogni fold
    for fold in range(NUM_FOLDS):
        metrics = evaluate_fold(fold, users, clients_raw, splits)
        if metrics is None:
            print(f"Fold {fold}: modello/scaler mancante, saltato")
            continue
        fold_metrics.append(metrics)
        report[str(fold)] = metrics
        print(f"Fold {fold}: MAE={metrics['mae']:.4f}, MedAE={metrics['medae']:.4f}, R2={metrics['r2']:.4f}")

    # Calcoliamo statistiche aggregate
    if fold_metrics:
        maes = [m["mae"] for m in fold_metrics]
        medaes = [m["medae"] for m in fold_metrics]
        r2s = [m["r2"] for m in fold_metrics]
        report["mean_mae"] = float(np.mean(maes))
        report["std_mae"] = float(np.std(maes))
        report["mean_medae"] = float(np.mean(medaes))
        report["mean_r2"] = float(np.mean(r2s))
        
        # Aggreghiamo anche gli errori per-utente
        per_user = {}
        for m in fold_metrics:
            for u, v in m["per_user_mae"].items():
                per_user.setdefault(u, []).append(v)
        per_user_mean = {u: float(np.mean(vs)) for u, vs in per_user.items()}
        report["per_user_mae_mean"] = per_user_mean
        
        print(f"MAE medio su {len(maes)} fold: {report['mean_mae']:.4f} (std {report['std_mae']:.4f})")
    else:
        print("Nessun fold valutato.")

    # Salviamo il report
    out_path = os.path.join(ARTIFACTS_DIR, "cv_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report salvato in {out_path}")


if __name__ == "__main__":
    main()
