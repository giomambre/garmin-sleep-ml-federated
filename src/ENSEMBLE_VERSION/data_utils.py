# data_utils.py
# Questo file si occupa di caricare e preparare i dati per l'allenamento
# È come il "cuoco" che prepara gli ingredienti prima di cucinare

import os, glob, ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, find_peaks, stft
import joblib
from config import MISSING_FILL_METHOD, APPLY_LOWPASS, LOWPASS_CUTOFF, LOWPASS_ORDER, ENABLE_FREQ_FEATURES

# Colonne che non ci servono e quindi eliminiamo
DROP_COLS = [
    'day', 'Unnamed: 0', 'act_activeTime'
]

def _parse_ts_cell(x):
    """
    Converte una cella che contiene una stringa con una lista
    (tipo "[1, 2, 3]") in una vera lista Python.
    
    Serve perché le serie temporali nei CSV sono salvate come testo.
    """
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            return []
    return []


def _ffill(arr):
    """
    Riempimento in avanti (forward fill):
    Se manca un valore, usiamo l'ultimo valore valido che abbiamo visto.
    Esempio: [1, NaN, NaN, 4] becomes [1, 1, 1, 4]
    """
    out = arr.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def _bfill(arr):
    """
    Riempimento all'indietro (backward fill):
    Se manca un valore, usiamo il prossimo valore valido.
    Esempio: [1, NaN, NaN, 4] diventa [1, 4, 4, 4]
    """
    out = arr.copy()
    nxt = np.nan
    for i in range(len(out) - 1, -1, -1):
        if np.isnan(out[i]):
            out[i] = nxt
        else:
            nxt = out[i]
    return out


def _linear_interp(arr):
    """
    Interpolazione lineare:
    Se mancano dei valori, li "inventiamo" tracciando una linea retta
    tra i valori che conosciamo.
    Esempio: [1, NaN, NaN, 4] diventa [1, 2, 3, 4]
    """
    if not np.isnan(arr).any():
        return arr
    idx = np.arange(len(arr))
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return np.zeros_like(arr)
    if mask.sum() == 1:
        return np.full_like(arr, arr[mask][0])
    arr[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
    return arr


def fill_missing(arr, method):
    """
    Riempie i valori mancanti in un array usando il metodo specificato.
    È la funzione "principale" che chiama le altre (_ffill, _bfill, _linear_interp).
    """
    if method == "ffill":
        out = _ffill(arr)
        if np.isnan(out[0]):
            out = _bfill(out)
        return np.nan_to_num(out)
    if method == "bfill":
        out = _bfill(arr)
        if np.isnan(out[-1]):
            out = _ffill(out)
        return np.nan_to_num(out)
    return np.nan_to_num(_linear_interp(arr))


def lowpass(arr, cutoff, order):
    """
    Filtro passa-basso (lowpass filter):
    Rimuove le oscillazioni troppo veloci dai segnali (il "rumore").
    È come levigare una superficie ruvida.
    
    Esempio pratico: il battito cardiaco ha piccole variazioni casuali
    dovute al sensore - questo filtro le elimina mantenendo il trend generale.
    """
    if len(arr) < (order * 3 + 1):
        # Se il segnale è troppo corto per un filtro vero, usiamo una media mobile semplice
        if len(arr) < 3:
            return arr
        kernel = np.ones(3, dtype=float) / 3.0
        return np.convolve(arr, kernel, mode='same')
    
    # Applichiamo il filtro di Butterworth (un filtro "standard" in signal processing)
    nyq = 0.5
    cutoff = float(np.clip(cutoff, 1e-3, nyq - 1e-3))
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    try:
        return filtfilt(b, a, arr)
    except Exception:
        return arr


def extract_time_series_features(df):
    """
    Questa è una delle funzioni più importanti!
    
    Prende i dati grezzi delle serie temporali (battito, respirazione, stress)
    e li trasforma in caratteristiche utili per il modello.
    
    Invece di dare al modello 1000 valori di battito cardiaco, gli diamo:
    - la media del battito
    - quanto varia (deviazione standard)
    - il valore minimo e massimo
    - quanti "picchi" ci sono
    - etc.
    
    In questo modo il modello capisce meglio i pattern nel sonno.
    """
    ts_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    # --- PARTE 1: PROPORZIONI DEL SONNO ---
    # Calcoliamo le percentuali: quanto tempo passiamo in fase REM, profonda, leggera
    if 'sleep_sleepTimeSeconds' in df.columns:
        total_sleep = df['sleep_sleepTimeSeconds'].replace(0, np.nan)
        
        if 'sleep_remSleepSeconds' in df.columns:
            df['ratio_rem'] = df['sleep_remSleepSeconds'] / total_sleep
        
        if 'sleep_deepSleepSeconds' in df.columns:
            df['ratio_deep'] = df['sleep_deepSleepSeconds'] / total_sleep
            
        if 'sleep_lightSleepSeconds' in df.columns:
            df['ratio_light'] = df['sleep_lightSleepSeconds'] / total_sleep

    # --- PARTE 2: STATISTICHE DELLE SERIE TEMPORALI ---
    # Per ogni tipo di segnale (battito, respirazione, stress)
    for col in ts_cols:
        if col not in df.columns:
            continue

        # Convertiamo le stringhe in liste di numeri
        series_data = df[col].apply(_parse_ts_cell)

        def calc_stats(lst):
            """
            Questa funzione interna calcola tutte le statistiche per UNA serie temporale.
            Per esempio, prende 1000 valori di battito cardiaco e restituisce:
            media, deviazione standard, min, max, percentili, numero di picchi, etc.
            """
            # Se la lista è vuota, restituiamo valori NaN
            if not lst:
                base = [np.nan] * 8
                extra = [np.nan, np.nan]
                freq = [np.nan, np.nan]
                return pd.Series(base + extra + freq)

            # Convertiamo in array numpy e puliamo i dati
            arr = np.array(lst, dtype=float)
            arr[~np.isfinite(arr)] = np.nan
            if np.isnan(arr).all():
                arr = np.zeros_like(arr)
            
            # Riempiamo i valori mancanti
            arr = fill_missing(arr, MISSING_FILL_METHOD)
            
            # Applichiamo il filtro per ridurre il rumore
            if APPLY_LOWPASS:
                arr = lowpass(arr, LOWPASS_CUTOFF, LOWPASS_ORDER)

            # Statistiche di base
            mean = np.mean(arr)         # Media
            std = np.std(arr)           # Deviazione standard (quanto varia)
            var = np.var(arr)           # Varianza
            amin = np.min(arr)          # Valore minimo
            amax = np.max(arr)          # Valore massimo
            rng = amax - amin           # Range (differenza tra max e min)
            p25 = np.percentile(arr, 25)  # 25° percentile
            p50 = np.median(arr)          # Mediana (50° percentile)
            p75 = np.percentile(arr, 75)  # 75° percentile

            # Zero crossings: quante volte il segnale attraversa la media
            # Indica quanto "oscilla" il segnale
            centered = arr - mean
            zc = np.mean(centered[:-1] * centered[1:] < 0) if len(arr) > 1 else 0.0
            
            # Numero di picchi: quante "montagne" ci sono nel segnale
            if len(arr) > 3 and rng > 0:
                peaks, _ = find_peaks(arr, prominence=(0.05 * rng))
                n_peaks = float(len(peaks))
            else:
                n_peaks = 0.0

            # Skewness e Kurtosis (per ora non li usiamo molto)
            if std < 1e-6:
                skew_val = 0.0
                kurt_val = 0.0
            else:
                skew_val = float(np.nan_to_num(pd.Series(arr).skew()))
                kurt_val = float(np.nan_to_num(pd.Series(arr).kurt()))

            # Feature in frequenza (opzionali, di solito disabilitate)
            freq_feats = [np.nan, np.nan]
            if ENABLE_FREQ_FEATURES and len(arr) >= 8:
                try:
                    _, _, Zxx = stft(arr, nperseg=min(64, len(arr)))
                    mag = np.abs(Zxx)
                    freq_feats = [float(np.mean(mag)), float(np.max(mag))]
                except Exception:
                    freq_feats = [np.nan, np.nan]

            base = [mean, std, amin, amax, rng, p25, p50, p75]
            extra = [zc, n_peaks]
            base.insert(2, var)  # Manteniamo l'ordine legacy
            return pd.Series(base + extra + freq_feats)

        # Applichiamo calc_stats a tutte le righe per questa colonna
        stats_df = series_data.apply(calc_stats)
        
        # Diamo nomi chiari alle colonne generate
        stats_df.columns = [
            f'{col}_mean', f'{col}_std', f'{col}_var', f'{col}_min', f'{col}_max', f'{col}_range',
            f'{col}_p25', f'{col}_p50', f'{col}_p75', f'{col}_zero_cross', f'{col}_n_peaks',
            f'{col}_freq_mean', f'{col}_freq_max'
        ]

        # Aggiungiamo le nuove colonne al dataframe
        df = pd.concat([df, stats_df], axis=1)
    
    # Eliminiamo le colonne originali delle serie temporali (non servono più)
    df = df.drop(columns=[c for c in ts_cols if c in df.columns])
    return df

def _clean_missing(df, features):
    """
    Pulisce i valori "segnaposto" negativi (-1, -2) che indicano dati mancanti
    e li sostituisce con NaN (il modo standard per indicare valori mancanti).
    """
    df = df.copy()
    for col in features:
        if col in df.columns:
            df[col] = df[col].astype(float)
            mask = df[col] < 0
            df.loc[mask, col] = np.nan
    return df


def load_clients(base_path):
    """
    Carica tutti i dati degli utenti dalla cartella specificata.
    
    In pratica:
    1. Cerca tutti i file CSV nella cartella
    2. Per ogni file, capisce di quale utente si tratta
    3. Estrae le caratteristiche dalle serie temporali
    4. Restituisce un dizionario: {nome_utente: dataframe_con_dati}
    
    Questo è il punto di partenza: carichiamo tutti i dati grezzi.
    """
    clients = {}
    files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)

    for f in files:
        # Ignoriamo i file di test
        if "x_test" in f:
            continue

        # Capiamo il nome dell'utente dal nome del file
        fname = os.path.basename(f)
        if "user_" in fname:
            user = fname.split("user_")[1].split("_")[0]
        else:
            user = os.path.basename(os.path.dirname(f))

        # Leggiamo il CSV
        df = pd.read_csv(f, sep=';')
        df.columns = df.columns.str.strip()
        
        # Estraiamo le feature dalle serie temporali
        df = extract_time_series_features(df)
        # Eliminiamo colonne inutili
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        clients.setdefault(user, []).append(df)

    # Se un utente ha più file, li uniamo
    for u in clients:
        clients[u] = pd.concat(clients[u], ignore_index=True)

    return clients


class FederatedScaler:
    """
    Scaler Federato Globale.
    
    Cosa fa?
    Calcola la media e la deviazione standard di TUTTI gli utenti insieme.
    Poi usa questi valori per "normalizzare" i dati (portarli sulla stessa scala).
    
    Quando lo usiamo?
    Per il test set di Kaggle, dove NON conosciamo l'identità dell'utente.
    In quel caso usiamo le statistiche "globali" di tutti gli utenti.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.top_features = None

    def fit_federated(self, clients_data, top_features):
        """
        Calcola media e deviazione standard globali su tutti gli utenti.
        È come fare una "media delle medie" ma in modo matematicamente corretto.
        """
        self.top_features = top_features
        n_features = len(top_features)
        
        # Accumulatori per le statistiche globali
        global_sum = np.zeros(n_features)
        global_sq_sum = np.zeros(n_features)  # somma dei quadrati
        global_count_obs = np.zeros(n_features)
        global_count_total = 0

        # Raccogliamo le statistiche da ogni utente
        for u, df in clients_data.items():
            df = _clean_missing(df, top_features)
            df = df.reindex(columns=top_features, fill_value=np.nan)
            vals = df[top_features].values
            
            obs_mask = ~np.isnan(vals)
            local_sum = np.nansum(vals, axis=0)
            local_sq_sum = np.nansum(vals**2, axis=0)
            local_count_obs = obs_mask.sum(axis=0)
            local_count_total = len(df)

            global_sum += local_sum
            global_sq_sum += local_sq_sum
            global_count_obs += local_count_obs
            global_count_total += local_count_total

        # Calcoliamo media globale
        self.mean_ = np.divide(global_sum, global_count_obs, out=np.zeros_like(global_sum), where=global_count_obs!=0)
        
        # Calcoliamo deviazione standard globale
        numerator = global_sq_sum - 2 * self.mean_ * global_sum + (self.mean_**2) * global_count_obs
        self.var_ = numerator / global_count_total
        self.scale_ = np.sqrt(self.var_)
        self.scale_[self.scale_ == 0] = 1.0  # evitiamo divisioni per zero
        return self

    def transform(self, df):
        """
        Applica la normalizzazione ai dati usando le statistiche globali.
        Formula: (valore - media) / deviazione_standard
        """
        if self.mean_ is None:
            raise ValueError("Scaler non ancora allenato! Chiama prima fit_federated.")
        df = df.copy()
        df = df.reindex(columns=self.top_features + [c for c in df.columns if c not in self.top_features], fill_value=np.nan)
        df = _clean_missing(df, self.top_features)
        # Riempiamo i NaN con la media globale
        df[self.top_features] = df[self.top_features].fillna(pd.Series(self.mean_, index=self.top_features))
        X = df[self.top_features].values
        # Normalizziamo: (X - media) / std
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled


def transform_clients_user_wise(clients, top_features):
    
    """
    NUOVA STRATEGIA: Normalizzazione Per-Utente.
    
    Invece di usare statistiche globali, ogni utente viene normalizzato
    usando la SUA media e la SUA deviazione standard.
    
    Perché?
    Immagina due persone: una ha battito cardiaco medio di 60, l'altra di 80.
    Con normalizzazione globale, uno sembrerà "lento" e l'altro "veloce".
    Con normalizzazione per-utente, entrambi vengono portati alla loro "normalità".
    
    Il modello impara così a guardare le VARIAZIONI rispetto alla norma dell'utente,
    non i valori assoluti. Questo lo rende più robusto quando vede persone nuove.
    """
    
    processed_clients = {}
    
    for u, df in clients.items():
        df = df.copy()
        
        # 1. Puliamo i valori mancanti
        df = _clean_missing(df, top_features)
        
        # 2. Imputazione e Scaling Locali (per questo specifico utente)
        # Usiamo lo StandardScaler di sklearn che fa esattamente questo
        scaler = StandardScaler()
        
        # Selezioniamo le feature che ci interessano
        X_raw = df.reindex(columns=top_features).values
        
        # Gestiamo i NaN prima dello scaling (imputazione con media locale)
        # Se una colonna è tutta NaN per un utente, usiamo 0 come default
        all_nan_cols = np.isnan(X_raw).all(axis=0)
        col_mean = np.zeros(X_raw.shape[1], dtype=float)
        if (~all_nan_cols).any():
            with np.errstate(invalid='ignore'):
                col_mean[~all_nan_cols] = np.nanmean(X_raw[:, ~all_nan_cols], axis=0)
        col_mean = np.nan_to_num(col_mean, nan=0.0)
        
        # Troviamo dove ci sono i NaN e li sostituiamo con la media
        inds = np.where(np.isnan(X_raw))
        X_raw[inds] = np.take(col_mean, inds[1])
        
        # Applichiamo lo scaling usando SOLO i dati di questo utente
        X_scaled = scaler.fit_transform(X_raw)
        
        # Prendiamo anche le etichette (punteggi di sonno) se presenti
        if 'label' in df.columns:
            y = (df['label'].values / 100.0).reshape(-1, 1)  # dividiamo per 100 per scaling
        else:
            y = None
            
        processed_clients[u] = (X_scaled, y)
        
    return processed_clients


def transform_clients(clients, fed_scaler):
    """
    Funzione per compatibilità: usa lo scaling GLOBALE (vecchio metodo).
    
    Quando la usiamo?
    - Per la validazione (simula il test set dove non conosciamo l'utente)
    - Per il test set di Kaggle (dove non sappiamo chi è l'utente)
    
    La differenza con transform_clients_user_wise:
    - Quella usa statistiche PER-UTENTE (per il training)
    - Questa usa statistiche GLOBALI (per test/validazione)
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
