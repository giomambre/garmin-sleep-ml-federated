from sklearn.ensemble import RandomForestRegressor
import numpy as np

class FederatedServer:
    def __init__(self, features_list):
        # warm_start=True è cruciale
        self.global_rf = RandomForestRegressor(warm_start=True)
        self.features_list = features_list

    def aggregate(self, list_of_client_trees):
        # 1. Uniamo tutti gli alberi dei client in una lista unica
        all_trees = [tree for sublist in list_of_client_trees for tree in sublist]
        
        if not all_trees:
            print("ATTENZIONE: Nessun albero ricevuto dai client!")
            return

        # 2. Iniettiamo gli alberi nel modello globale vuoto
        self.global_rf.estimators_ = all_trees
        self.global_rf.n_estimators = len(all_trees)
        
        # --- FIX FEDERATO: Impostiamo manualmente i metadati mancanti ---
        # Questo serve perché non abbiamo chiamato .fit() sul server
        
        # Dice al modello che deve prevedere 1 solo valore (la qualità del sonno)
        self.global_rf.n_outputs_ = 1 
        
        # Dice al modello quante colonne aspettarsi
        self.global_rf.n_features_in_ = len(self.features_list)
        
        # Assegna i nomi delle colonne (utile per debug)
        self.global_rf.feature_names_in_ = np.array(self.features_list, dtype=object)
        
        # Imposta l'output come float (regressione)
        # Alcune versioni di sklearn richiedono outputs_ o simili, ma n_outputs_ è il principale
        
        print(f"3. Aggregazione completata: Foresta globale con {len(all_trees)} alberi pronta.")

    def predict(self, X):
        return self.global_rf.predict(X)