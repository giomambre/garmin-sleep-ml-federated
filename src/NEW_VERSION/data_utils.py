# data_utils.py
import os, glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

DROP_COLS = [
    'day', 'hr_time_series', 'resp_time_series',
    'stress_time_series', 'Unnamed: 0', 'act_activeTime'
]


def _clean_missing(df, features):
    """Replace negative placeholders (-1, -2) with NaN before imputing."""
    # Forziamo a float per poter inserire NaN senza warning di dtype incompatibile
    df = df.copy()
    df[features] = df[features].astype(float)
    # Nel dataset i mancanti sono spesso -1/-2: li convertiamo a NaN così le medie li rimpiazzano.
    mask = df[features] < 0
    df.loc[:, features] = df[features].where(~mask, np.nan)
    return df


def load_clients(base_path):
    clients = {}
    files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)

    for f in files:
        if "x_test" in f:
            continue

        fname = os.path.basename(f)
        # expected: dataset_user_<id>_train.csv
        if "user_" in fname:
            user = fname.split("user_")[1].split("_")[0]
        else:
            user = os.path.basename(os.path.dirname(f))

        df = pd.read_csv(f, sep=';')
        df.columns = df.columns.str.strip()
        # Toglie le colonne non usate e accorpa tutti i file dello stesso utente
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        clients.setdefault(user, []).append(df)

    for u in clients:
        clients[u] = pd.concat(clients[u], ignore_index=True)

    return clients


def prepare_data(clients, top_features):
    scaler = StandardScaler()

    full = pd.concat(clients.values())
    full = _clean_missing(full, top_features)
    # Allineiamo le colonne alle feature attese (mancanti riempite con NaN),
    # calcoliamo le medie SOLO sul train e le useremo anche in inferenza.
    full = full.reindex(columns=top_features + [c for c in full.columns if c not in top_features], fill_value=np.nan)
    feature_means = full[top_features].mean()
    full[top_features] = full[top_features].fillna(feature_means)
    X_all = full[top_features].values
    scaler.fit(X_all)

    for u, df in clients.items():
        # Applichiamo le stesse colonne e le stesse medie calcolate sul train
        df = df.reindex(columns=top_features + [c for c in df.columns if c not in top_features], fill_value=np.nan)
        df = _clean_missing(df, top_features)
        df[top_features] = df[top_features].fillna(feature_means)
        X = scaler.transform(df[top_features].values)
        y = (df['label'].values / 100.0).reshape(-1, 1)
        clients[u] = (X, y)

    # Salviamo scaler + medie per riapplicarli identici in inferenza
    joblib.dump({"scaler": scaler, "feature_means": feature_means}, "scaler.joblib")
    return clients, scaler
