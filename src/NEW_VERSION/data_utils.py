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
    A scaler that fits using aggregated statistics from clients,
    ensuring no raw data is shared.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0
        self.top_features = None

    def fit_federated(self, clients_data, top_features):
        self.top_features = top_features
        n_features = len(top_features)
        
        # Aggregators for global stats
        global_sum = np.zeros(n_features)
        global_sq_sum = np.zeros(n_features)
        global_count_obs = np.zeros(n_features)
        global_count_total = 0

        # --- PHASE 1: Clients calculate local stats (Simulation) ---
        for u, df in clients_data.items():
            # 1. Clean locally
            df = _clean_missing(df, top_features)
            # 2. Ensure columns exist
            df = df.reindex(columns=top_features, fill_value=np.nan)
            vals = df[top_features].values
            
            # 3. Calculate local stats
            # Mask of observed values (not NaN)
            obs_mask = ~np.isnan(vals)
            
            # Sum of observed values
            local_sum = np.nansum(vals, axis=0)
            
            # Sum of squares of observed values
            local_sq_sum = np.nansum(vals**2, axis=0)
            
            # Count of observed values
            local_count_obs = obs_mask.sum(axis=0)
            
            # Total rows (including missing)
            local_count_total = len(df)

            # --- Client sends stats to Server ---
            global_sum += local_sum
            global_sq_sum += local_sq_sum
            global_count_obs += local_count_obs
            global_count_total += local_count_total

        # --- PHASE 2: Server aggregates and computes global parameters ---
        
        # 1. Global Mean (used for imputation)
        # Avoid division by zero
        self.mean_ = np.divide(
            global_sum, 
            global_count_obs, 
            out=np.zeros_like(global_sum), 
            where=global_count_obs!=0
        )
        
        # 2. Global Std (used for scaling)
        # We need the variance of the dataset *after* imputation with the mean.
        # Formula: Variance = (Sum_sq_observed - 2*Mean*Sum_observed + Count_observed*Mean^2) / Total_Count
        # Note: The contribution of imputed values to the sum of squared errors (x - mean)^2 is 0.
        
        numerator = (
            global_sq_sum 
            - 2 * self.mean_ * global_sum 
            + (self.mean_**2) * global_count_obs
        )
        
        # Variance
        self.var_ = numerator / global_count_total
        
        # Std Dev
        self.scale_ = np.sqrt(self.var_)
        
        # Handle constant features (scale=0) -> set to 1 to avoid div by zero
        self.scale_[self.scale_ == 0] = 1.0
        
        self.n_samples_seen_ = global_count_total
        return self

    def transform(self, df):
        """
        Applies imputation (using global mean) and scaling (using global std).
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted yet.")
            
        df = df.copy()
        # Ensure columns match
        df = df.reindex(columns=self.top_features + [c for c in df.columns if c not in self.top_features], fill_value=np.nan)
        
        # Clean missing values
        df = _clean_missing(df, self.top_features)
        
        # Impute with GLOBAL mean
        df[self.top_features] = df[self.top_features].fillna(pd.Series(self.mean_, index=self.top_features))
        
        # Scale with GLOBAL stats
        X = df[self.top_features].values
        X_scaled = (X - self.mean_) / self.scale_
        
        return X_scaled


def transform_clients(clients, fed_scaler):
    """
    Transforms clients using the FederatedScaler.
    Returns a dictionary of (X, y) tuples.
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
    DEPRECATED: Use FederatedScaler manually in training script.
    """
    raise NotImplementedError("Use FederatedScaler in train_federated.py instead.")
