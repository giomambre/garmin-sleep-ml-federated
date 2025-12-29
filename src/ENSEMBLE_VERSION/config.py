# config.py


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "DATASET", "CSV_train")

SEED = 42

# Training
ROUNDS = 50         # federated rounds (più iterazioni per convergere)
LOCAL_EPOCHS = 5    # epochs per client (ridotto per evitare overfitting)
BATCH_SIZE = 16
LR = 5e-4            # LR più basso per convergenza stabile
CLIENT_FRACTION = 0.5 # Fraction of clients to sample in each round (e.g., 0.5 for 50%)
NUM_FOLDS = 5       # Numero di modelli nell'ensemble

# Model
HIDDEN_1 = 256      # Increased capacity
HIDDEN_2 = 128      # Increased capacity
HIDDEN_3 = 64       # Increased capacity
DROPOUT = 0.2       # Increased dropout for regularization

# Data
TARGET_SCALE = 100.0

# Features
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