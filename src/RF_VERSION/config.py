import os

# --- Percorsi ---
# Percorso base (relativo a dove lanci lo script, si assume dalla root del progetto)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) # Risale fino a CSI project
DATASET_DIR = os.path.join(PROJECT_ROOT, "src", "DATASET", "CSV_train")

# --- Parametri Federati ---
ROUNDS = 1  # Con le Random Forest spesso basta 1 solo round (ogni client traina, il server unisce)
            # Se facciamo boosting servono più round, ma per Bagging semplice 1 round è lo standard.
K_FOLDS = 5
RANDOM_SEED = 42

# --- Parametri Random Forest Client ---
# Ogni client allenerà una foresta con questi parametri
# ANTI-OVERFITTING CONFIGURATION
RF_N_ESTIMATORS = 20   # Ridotto da 50 a 20 per client (Totale alberi globale sarà comunque alto: ~900)
RF_MAX_DEPTH = 5       # Ridotto drasticamente da 15 a 5 per forzare generalizzazione
RF_MIN_SAMPLES_SPLIT = 8 # Aumentato da 5 a 8: serve un gruppo di dati consistente per splittare
RF_N_JOBS = -1         # Usa tutti i core CPU

# --- Feature ---
TOP_FEATURES = [
    'act_activeKilocalories', 'act_totalCalories',
    'resp_avgTomorrowSleepRespirationValue',
    'sleep_remSleepSeconds', 'act_distance',
    'str_avgStressLevel', 'sleep_sleepTimeSeconds',
    'sleep_awakeSleepSeconds', 'hr_maxHeartRate',
    'sleep_deepSleepSeconds', 'sleep_lightSleepSeconds',
    'sleep_avgSleepStress', 'hr_lastSevenDaysAvgRestingHeartRate',
    'str_maxStressLevel', 'hr_minHeartRate',
    'hr_restingHeartRate', 'sleep_napTimeSeconds',
    'sleep_lowestRespirationValue', 'sleep_avgHeartRate',
    'resp_highestRespirationValue',
    'resp_avgSleepRespirationValue', 'resp_avgWakingRespirationValue',
    'resp_lowestRespirationValue',
    # New Time Series Features
    'hr_time_series_mean', 'hr_time_series_std', 'hr_time_series_min', 'hr_time_series_max', 'hr_time_series_range',
    'resp_time_series_mean', 'resp_time_series_std', 'resp_time_series_min', 'resp_time_series_max', 'resp_time_series_range',
    'stress_time_series_mean', 'stress_time_series_std', 'stress_time_series_min', 'stress_time_series_max', 'stress_time_series_range'
]

LABEL_COL = 'label'
