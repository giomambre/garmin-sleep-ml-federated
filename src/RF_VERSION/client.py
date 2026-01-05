import numpy as np
from sklearn.ensemble import RandomForestRegressor
from config import RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_N_JOBS, RANDOM_SEED

class FederatedRFClient:
    def __init__(self, user_id, X_train, y_train):
        self.user_id = user_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def fit(self):
        """
        Allena una Random Forest locale sui dati del client.
        """
        print(f"[Client {self.user_id}] Start training RF ({len(self.X_train)} samples)...")
        
        # Usiamo un seed diverso per ogni client per garantire diversità negli alberi
        # (anche se i dati diversi già aiutano)
        client_seed = RANDOM_SEED + hash(self.user_id) % 10000
        
        self.model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            n_jobs=RF_N_JOBS,
            random_state=client_seed
        )
        
        self.model.fit(self.X_train, self.y_train)
        print(f"[Client {self.user_id}] Training done.")
        
        # Restituiamo la lista degli alberi (estimators_)
        return self.model.estimators_
