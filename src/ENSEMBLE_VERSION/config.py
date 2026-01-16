# config.py
# Questo file contiene tutte le impostazioni del progetto
# Come una "centrale di controllo" dove decidiamo tutti i parametri

import os

# === PERCORSI DEI FILE ===
# Dove si trova questo file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Dove sono i dati di allenamento
DATASET_PATH = os.path.join(BASE_DIR, "..", "DATASET", "CSV_train")
# Dove salviamo i modelli allenati e altri file generati
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "..", "artifacts")

# Creiamo la cartella artifacts se non esiste
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

# Seme per la casualità - così i risultati sono riproducibili
SEED = 42

# === PARAMETRI DI ALLENAMENTO ===
ROUNDS = 60              # Quanti "giri" di comunicazione tra utenti e server
LOCAL_EPOCHS = 5         # Quante volte ogni utente allena il modello sui suoi dati
BATCH_SIZE = 16          # Quanti esempi processare insieme per volta
LR = 5e-4                # Learning rate: quanto velocemente impara (0.0005)
CLIENT_FRACTION = 0.5    # Quanti utenti partecipano ad ogni round (50%)
NUM_FOLDS = 10           # In quante parti dividiamo i dati per la validazione incrociata           # In quante parti dividiamo i dati per la validazione incrociata

# === ARCHITETTURA DELLA RETE NEURALE ===
# Il nostro modello ha 3 "strati nascosti" con questi numeri di neuroni:
HIDDEN_1 = 256           # Primo strato: 256 neuroni
HIDDEN_2 = 128           # Secondo strato: 128 neuroni
HIDDEN_3 = 64            # Terzo strato: 64 neuroni
DROPOUT = 0.20           # Dropout: "spegniamo" il 20% dei neuroni a caso per evitare overfitting

# === SCALA DEI DATI ===
TARGET_SCALE = 100.0     # Dividiamo i punteggi per 100 per facilitare l'apprendimento     # Dividiamo i punteggi per 100 per facilitare l'apprendimento

# === PRE-ELABORAZIONE DEI SEGNALI ===
# Come gestiamo i dati mancanti e il rumore nei segnali
MISSING_FILL_METHOD = "linear"   # Interpoliamo linearmente i valori mancanti
APPLY_LOWPASS = False             # Applichiamo un filtro per ridurre il rumore
LOWPASS_CUTOFF = 0.12            # Frequenza di taglio del filtro (normalizzata)
LOWPASS_ORDER = 3                # Ordine del filtro (più alto = più forte)

# === CARATTERISTICHE AVANZATE ===
ENABLE_FREQ_FEATURES = False     # Per ora disabilitiamo le feature in frequenza (troppo complesse)     # Per ora disabilitiamo le feature in frequenza (troppo complesse)

# === CONTROLLO DELL'ALLENAMENTO ===
# Meccanismi per evitare di sprecare tempo se il modello non migliora
EARLY_STOP_PATIENCE = 4          # Se non migliora per 4 valutazioni, fermiamo l'allenamento
LR_REDUCE_PATIENCE = 2           # Se non migliora per 2 valutazioni, riduciamo la velocità
LR_REDUCE_FACTOR = 0.5           # Riduciamo la velocità della metà
MIN_LR = 1e-5                    # Velocità minima di apprendimento

# === BILANCIAMENTO FEDERATED LEARNING ===
WEIGHT_CAP_QUANTILE = 0.90       # Limitiamo il peso degli utenti con troppi dati (al 90° percentile)

# Features - VENGEANCE CONFIGURATION (expanded with TS temporal/frequency)
TOP_FEATURES = [
    'act_activeKilocalories', 'act_totalCalories', 'act_distance',
    
    # Ratios (Sempre utili per logica di dominio)
    'ratio_rem', 'ratio_deep', 'ratio_light',
    
    # --- DURATE DEL SONNO ---
    # Quanto tempo si dorme, quanto tempo si è svegli, le varie fasi
    'sleep_sleepTimeSeconds', 'sleep_awakeSleepSeconds',
    'sleep_remSleepSeconds', 'sleep_deepSleepSeconds', 'sleep_lightSleepSeconds',
    'sleep_napTimeSeconds',
    
    # --- BATTITO CARDIACO (Heart Rate) ---
    # Valori massimi, minimi, a riposo e medie degli ultimi giorni
    'hr_maxHeartRate', 'hr_minHeartRate', 'hr_restingHeartRate',
    'hr_lastSevenDaysAvgRestingHeartRate', 'sleep_avgHeartRate',
    
    # --- STRESS E RESPIRAZIONE ---
    # Livelli di stress e frequenza respiratoria durante sonno/veglia
    'str_avgStressLevel', 'str_maxStressLevel', 'sleep_avgSleepStress',
    'sleep_lowestRespirationValue', 'resp_highestRespirationValue',
    'resp_avgSleepRespirationValue', 'resp_avgWakingRespirationValue',
    'resp_lowestRespirationValue',
    
    # --- STATISTICHE DELLE SERIE TEMPORALI: BATTITO CARDIACO ---
    # Analisi dettagliata del battito durante tutta la notte: media, variabilità, min, max, percentili
    'hr_time_series_mean', 'hr_time_series_std', 'hr_time_series_min', 'hr_time_series_max', 'hr_time_series_range',
    'hr_time_series_p25', 'hr_time_series_p50', 'hr_time_series_p75',
    
    # --- STATISTICHE DELLE SERIE TEMPORALI: RESPIRAZIONE ---
    # Come per il battito, ma per la respirazione durante la notte
    'resp_time_series_mean', 'resp_time_series_std', 'resp_time_series_min', 'resp_time_series_max', 'resp_time_series_range',
    'resp_time_series_p25', 'resp_time_series_p50', 'resp_time_series_p75',
    'resp_time_series_zero_cross', 'resp_time_series_n_peaks', 'resp_time_series_freq_mean', 'resp_time_series_freq_max',

    # --- STATISTICHE DELLE SERIE TEMPORALI: STRESS ---
    # Analisi dello stress nel tempo: variabilità, picchi, pattern
    'stress_time_series_mean', 'stress_time_series_std', 'stress_time_series_min', 'stress_time_series_max', 'stress_time_series_range',
    'stress_time_series_p25', 'stress_time_series_p50', 'stress_time_series_p75',
    'stress_time_series_zero_cross', 'stress_time_series_n_peaks', 'stress_time_series_freq_mean', 'stress_time_series_freq_max'
]