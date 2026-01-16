import os, glob, ast
import numpy as np
import pandas as pd

def extract_ts_features(ts_obj):
    """
    Estrae media e deviazione standard. 
    Se qualcosa va storto, restituisce 0.0 senza stampare nulla.
    """
    try:
        # Gestione valori nulli o vuoti
        if pd.isna(ts_obj) or ts_obj == "" or ts_obj == "[]":
            return 0.0, 0.0

        # Se è una stringa (es: "[10, 20]"), convertila
        if isinstance(ts_obj, str):
            # Rimuove spazi e caratteri sporchi
            ts_obj = ts_obj.strip()
            # Parsing sicuro
            ts_list = ast.literal_eval(ts_obj)
        elif isinstance(ts_obj, list):
            ts_list = ts_obj
        else:
            return 0.0, 0.0

        # Conversione in serie numerica
        s = pd.Series(ts_list, dtype=float)
        
        # Filtra i valori non validi (sensore staccato <= 0)
        s = s[s > 0]
        
        if s.empty:
            return 0.0, 0.0
            
        return float(s.mean()), float(s.std())

    except:
        # In caso di errore, ritorna 0 silenziosamente
        return 0.0, 0.0

def load_and_clean(base_path, features_list):
    clients_dict = {}
    print(f"--- Caricamento dati da: {base_path} ---")
    
    # Cerca i file CSV
    files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
    print(f"File trovati: {len(files)}")

    if len(files) == 0:
        return {}

    loaded_count = 0
    
    for f in files:
        # Salta file non pertinenti
        if "x_test" in f or os.path.basename(f).startswith("._"): 
            continue
            
        fname = os.path.basename(f)
        
        # Estrai User ID
        if "user_" in fname:
            user = fname.split("user_")[1].split("_")[0]
        elif "dataset_" in fname:
             user = fname.split("_")[1]
        else:
            user = "unknown"

        try:
            # 1. LETTURA CSV
            # Usa engine='python' che è più robusto per i separatori
            df = pd.read_csv(f, sep=';', engine='python')
            
            # Se ha letto male le colonne (tutto in una riga), riprova con la virgola
            if len(df.columns) <= 1:
                df = pd.read_csv(f, sep=',', engine='python')
            
            df.columns = df.columns.str.strip() # Pulisce i nomi colonne

            if 'label' not in df.columns:
                print(f"SKIP {fname}: Colonna 'label' assente.")
                continue

            # 2. ESTRAZIONE FEATURE TIME SERIES
            # Definiamo le colonne da cercare
            ts_cols = ['hr_time_series', 'resp_time_series']
            prefixes = ['hr', 'resp']
            
            for col_raw, prefix in zip(ts_cols, prefixes):
                if col_raw in df.columns:
                    stats = df[col_raw].apply(extract_ts_features)
                    df[f'{prefix}_ts_mean'] = stats.apply(lambda x: x[0])
                    df[f'{prefix}_ts_std'] = stats.apply(lambda x: x[1])
                else:
                    df[f'{prefix}_ts_mean'] = 0.0
                    df[f'{prefix}_ts_std'] = 0.0

            # 3. CALCOLO DELTA (con protezione errori)
            if 'hr_restingHeartRate' in df.columns:
                hr_now = pd.to_numeric(df['hr_restingHeartRate'], errors='coerce').fillna(0)
                # Cerca la colonna storica (gestisce nomi diversi)
                hist_col = 'hr_lastSevenDaysAvgRestingHeartRate'
                if hist_col not in df.columns: hist_col = 'hr_restingHeartRate'
                hr_hist = pd.to_numeric(df[hist_col], errors='coerce').fillna(0)
                
                df['hr_delta_resting'] = hr_now - hr_hist
            else:
                df['hr_delta_resting'] = 0.0

            # 4. PULIZIA FINALE
            # Rimuove NaN e infiniti
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0) # Riempie tutto con 0 per sicurezza Random Forest
            
            # Seleziona solo le colonne richieste dal Config
            # Crea un dataframe vuoto con le colonne giuste e riempilo
            X = pd.DataFrame(0.0, index=np.arange(len(df)), columns=features_list)
            for feat in features_list:
                if feat in df.columns:
                    X[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
            
            y = pd.to_numeric(df['label'], errors='coerce').fillna(0)
            
            clients_dict[user] = (X, y)
            loaded_count += 1
            
        except Exception as e:
            # Stampa solo il TIPO di errore, non i dati
            print(f"ERRORE su {fname}: {type(e).__name__}")
            continue

    print(f"--- Utenti caricati correttamente: {loaded_count} ---")
    return clients_dict