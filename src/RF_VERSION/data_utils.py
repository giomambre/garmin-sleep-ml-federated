import os, glob, ast
import numpy as np
import pandas as pd
from config import LABEL_COL

DROP_COLS = [
    'day', 'Unnamed: 0', 'act_activeTime'
]

def extract_time_series_features(df):
    """
    Parses string columns containing time series lists (e.g., "[1, 2, 3]")
    and extracts statistical features (mean, std, min, max, range).
    Drops the original time series columns afterwards.
    """
    ts_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in ts_cols:
        if col not in df.columns:
            continue
            
        def parse_ts(x):
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                return []
            except:
                return []

        series_data = df[col].apply(parse_ts)
        
        def calc_stats(lst):
            if not lst or len(lst) == 0:
                return pd.Series([np.nan]*5)
            try:
                arr = np.array(lst, dtype=float)
            except (ValueError, TypeError):
                arr = np.array([float(x) if x is not None else np.nan for x in lst])
            
            if np.isnan(arr).all():
                 return pd.Series([np.nan]*5)

            return pd.Series([
                np.nanmean(arr),
                np.nanstd(arr),
                np.nanmin(arr),
                np.nanmax(arr),
                np.nanmax(arr) - np.nanmin(arr)
            ])
            
        stats_df = series_data.apply(calc_stats)
        stats_df.columns = [f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max', f'{col}_range']
        df = pd.concat([df, stats_df], axis=1)
        
    df = df.drop(columns=[c for c in ts_cols if c in df.columns])
    return df

def _clean_missing(df, features):
    """Replace negative placeholders (-1, -2) with NaN before imputing."""
    df = df.copy()
    for f in features:
        if f in df.columns:
            df[f] = df[f].astype(float)
            mask = df[f] < 0
            df.loc[mask, f] = np.nan
    return df


def load_clients(base_path):
    clients = {}
    files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)

    for f in files:
        if "x_test" in f:
            continue

        fname = os.path.basename(f)
        if "user_" in fname:
            try:
                user = fname.split("user_")[1].split("_")[0]
            except IndexError:
                user = os.path.basename(os.path.dirname(f))
        else:
            user = os.path.basename(os.path.dirname(f))

        try:
            df = pd.read_csv(f, sep=';')
            df.columns = df.columns.str.strip()
            df = extract_time_series_features(df)
            df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
            clients.setdefault(user, []).append(df)
        except Exception as e:
            print(f"Skipping file {f}: {e}")

    for u in clients:
        clients[u] = pd.concat(clients[u], ignore_index=True)

    return clients


class FederatedScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0
        self.top_features = None

    def fit_federated(self, clients_data, top_features):
        self.top_features = top_features
        n_features = len(top_features)
        
        global_sum = np.zeros(n_features)
        global_sq_sum = np.zeros(n_features)
        global_count_obs = np.zeros(n_features)
        global_count_total = 0

        for u, df in clients_data.items():
            df = _clean_missing(df, top_features)
            # Reindex assicura che tutte le colonne esistano (aggiunge NaN se mancano)
            df = df.reindex(columns=top_features) 
            vals = df.values
            
            obs_mask = ~np.isnan(vals)
            local_sum = np.nansum(vals, axis=0)
            local_sq_sum = np.nansum(vals**2, axis=0)
            local_count_obs = obs_mask.sum(axis=0)
            local_count_total = len(df)

            global_sum += local_sum
            global_sq_sum += local_sq_sum
            global_count_obs += local_count_obs
            global_count_total += local_count_total

        # Media Globale
        self.mean_ = np.divide(
            global_sum, 
            global_count_obs, 
            out=np.zeros_like(global_sum), 
            where=global_count_obs!=0
        )
        
        # Deviazione Standard Globale
        numerator = (
            global_sq_sum 
            - 2 * self.mean_ * global_sum 
            + (self.mean_**2) * global_count_obs
        )
        self.var_ = numerator / global_count_total
        self.scale_ = np.sqrt(self.var_)
        self.scale_[self.scale_ == 0] = 1.0
        self.n_samples_seen_ = global_count_total
        return self

    def transform(self, df):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler non ancora fittato.")
            
        df = df.copy()
        df = df.reindex(columns=self.top_features + [c for c in df.columns if c not in self.top_features])
        df = _clean_missing(df, self.top_features)
        
        # Imputazione
        df[self.top_features] = df[self.top_features].fillna(pd.Series(self.mean_, index=self.top_features))
        
        # Scaling
        X = df[self.top_features].values
        X_scaled = (X - self.mean_) / self.scale_
        
        return X_scaled


def transform_clients(clients, fed_scaler):
    processed_clients = {}
    for u, df in clients.items():
        X = fed_scaler.transform(df)
        
        if LABEL_COL in df.columns:
            # LABEL_COL è 'sleep_score', valori 0-100.
            # Normalizziamo 0-1 (opzionale per RF, ma utile per coerenza)
            # Ma le RF lavorano bene anche coi raw values. Manteniamo raw per semplicità interpretativa,
            # o seguiamo new_version / 100.0?
            # Seguiamo NEW_VERSION / 100.0 per coerenza di metriche (MAE)
            y = (df[LABEL_COL].values / 100.0)
        else:
            y = None
            
        processed_clients[u] = (X, y)
        
    return processed_clients
