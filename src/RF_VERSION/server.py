import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor
from config import RF_N_JOBS

class FederatedRFServer:
    def __init__(self):
        self.global_model = None
        self.all_estimators = []

    def aggregate(self, client_estimators_list):
        """
        Riceve liste di alberi dai client e li unisce in un unico modello globale.
        """
        total_trees_before = len(self.all_estimators)
        
        for estimators in client_estimators_list:
            self.all_estimators.extend(estimators)
            
        total_trees_after = len(self.all_estimators)
        print(f"[Server] Aggregated {total_trees_after - total_trees_before} new trees. Total: {total_trees_after}")

        # Costruiamo il modello globale se non esiste, o lo aggiorniamo
        if self.global_model is None:
            # Creiamo un wrapper vuoto. I parametri non contano molto qui, 
            # contano quelli con cui sono stati allenati i singoli alberi.
            # Tuttavia, sklearn usa questi params per validazione o fit successivi.
            # Prendiamo il primo albero per inferire n_features se necessario, 
            # ma RandomForestRegressor non salva n_features_in_ esplicitamente nella lista estimators.
            
            # Creiamo una shell
            self.global_model = RandomForestRegressor(
                n_estimators=total_trees_after,
                n_jobs=RF_N_JOBS,
                warm_start=True 
            )
            # Dobbiamo fittarlo su dati dummy per inizializzare le strutture interne di sklearn?
            # Sì, sklearn è schizzinoso. Un trucco è fittarlo su 1 campione dummy 
            # e poi sovrascrivere self.global_model.estimators_
            
            # Tuttavia, il modo più pulito è non creare un "nuovo" RF, ma usarne uno come contenitore.
            pass

    def get_global_model(self, X_sample_for_init, y_sample_for_init):
        """
        Restituisce un RandomForestRegressor funzionante contenente tutti gli alberi aggregati.
        Serve un campione di dati (X, y) solo per inizializzare le strutture interne di sklearn
        (n_outputs, n_features_in_, ecc.) se non è mai stato fittato.
        """
        if not self.all_estimators:
            return None
            
        # Inizializziamo un modello vuoto
        rf = RandomForestRegressor(
            n_estimators=len(self.all_estimators),
            n_jobs=RF_N_JOBS
        )
        
        # Fake fit per settare attributi (n_features_, n_outputs_, ecc)
        # Usiamo solo 2 campioni per velocità
        rf.fit(X_sample_for_init[:2], y_sample_for_init[:2])
        
        # SOVRASCRIVIAMO gli alberi con quelli federati
        rf.estimators_ = self.all_estimators
        
        return rf
