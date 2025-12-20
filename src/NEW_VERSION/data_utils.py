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


class FederatedScaler:
    """
    Uno scaler che si adatta usando statistiche aggregate dai client,
    garantendo che nessun dato grezzo venga condiviso.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0
        self.top_features = None

    def fit_federated(self, clients_data, top_features):
        self.top_features = top_features
        n_features = len(top_features)
        
        # Aggregatori per le statistiche globali
        global_sum = np.zeros(n_features)
        global_sq_sum = np.zeros(n_features)
        global_count_obs = np.zeros(n_features)
        global_count_total = 0

        # --- FASE 1: I client calcolano le statistiche locali (Simulazione) ---
        for u, df in clients_data.items():
            # 1. Pulizia locale
            df = _clean_missing(df, top_features)
            # 2. Assicuriamoci che le colonne esistano
            df = df.reindex(columns=top_features, fill_value=np.nan)
            vals = df[top_features].values
            
            # 3. Calcolo statistiche locali
            # Maschera dei valori osservati (non NaN)
            obs_mask = ~np.isnan(vals)
            
            # Somma dei valori osservati
            local_sum = np.nansum(vals, axis=0)
            
            # Somma dei quadrati dei valori osservati
            local_sq_sum = np.nansum(vals**2, axis=0)
            
            # Conteggio dei valori osservati
            local_count_obs = obs_mask.sum(axis=0)
            
            # Totale righe (inclusi mancanti)
            local_count_total = len(df)

            # --- Il client invia le statistiche al Server ---
            global_sum += local_sum
            global_sq_sum += local_sq_sum
            global_count_obs += local_count_obs
            global_count_total += local_count_total

        # --- FASE 2: Il server aggrega e calcola i parametri globali ---
        
        # 1. Media Globale (usata per imputazione)
        # Evitiamo divisione per zero
        self.mean_ = np.divide(
            global_sum, 
            global_count_obs, 
            out=np.zeros_like(global_sum), 
            where=global_count_obs!=0
        )
        
        # 2. Deviazione Standard Globale (usata per scaling)
        # Ci serve la varianza del dataset *dopo* l'imputazione con la media.
        # Formula: Varianza = (Somma_quadrati_osservati - 2*Media*Somma_osservati + Conteggio_osservati*Media^2) / Totale_Conteggi
        # Nota: Il contributo dei valori imputati alla somma degli errori quadratici (x - media)^2 è 0.
        
        numerator = (
            global_sq_sum 
            - 2 * self.mean_ * global_sum 
            + (self.mean_**2) * global_count_obs
        )
        
        # Varianza
        self.var_ = numerator / global_count_total
        
        # Deviazione Standard
        self.scale_ = np.sqrt(self.var_)
        
        # Gestione feature costanti (scale=0) -> impostiamo a 1 per evitare div per zero
        self.scale_[self.scale_ == 0] = 1.0
        
        self.n_samples_seen_ = global_count_total
        return self

    def transform(self, df):
        """
        Applica imputazione (usando media globale) e scaling (usando std globale).
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler non ancora fittato.")
            
        df = df.copy()
        # Assicuriamoci che le colonne corrispondano
        df = df.reindex(columns=self.top_features + [c for c in df.columns if c not in self.top_features], fill_value=np.nan)
        
        # Pulizia valori mancanti
        df = _clean_missing(df, self.top_features)
        
        # Imputazione con media GLOBALE
        df[self.top_features] = df[self.top_features].fillna(pd.Series(self.mean_, index=self.top_features))
        
        # Scaling con statistiche GLOBALI
        X = df[self.top_features].values
        X_scaled = (X - self.mean_) / self.scale_
        
        return X_scaled


def transform_clients(clients, fed_scaler):
    """
    Trasforma i client usando il FederatedScaler.
    Restituisce un dizionario di tuple (X, y).
    """
    processed_clients = {}
    for u, df in clients.items():
        X = fed_scaler.transform(df)
        
        if 'label' in df.columns:
            y = (df['label'].values / 100.0).reshape(-1, 1)
        else:
            y = None
            
        processed_clients[u] = (X, y)
        
    return processed_clients


def prepare_data(clients, top_features):
    """
    DEPRECATO: Usa FederatedScaler manualmente nello script di training.
    """
    raise NotImplementedError("Usa FederatedScaler in train_federated.py invece.")
