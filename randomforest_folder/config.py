import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# IL TUO PERCORSO (Lascialo invariato se funziona)
DATASET_PATH = r"C:\Users\isare\poli_progetti\CSI_PROGETTO4\garmin-sleep-ml-federated\dataset\CSV_train\CSV_train"

SEED = 42

# --- PARAMETRI "ROBUST" (ANTI-OVERFITTING) ---
# Aumentiamo gli alberi ma li rendiamo molto "bassi" e generalisti
N_TREES_PER_CLIENT = 50   # Più alberi per stabilizzare il voto
MAX_DEPTH = 4             # Bassissimo: costringe a trovare solo le regole vere
MIN_SAMPLES_LEAF = 5      # Ogni regola deve valere per almeno 5 notti (ignora le eccezioni)

# --- LE "ELITE 8" FEATURES ---
# Teniamo solo quelle che hanno una correlazione biologica certa
TOP_FEATURES = [
    'hr_delta_resting',        # Il re delle feature (Fatica)
    'hr_ts_std',               # Variabilità cardiaca (Recupero)
    'act_activeKilocalories',  # Attività fisica
    'sleep_deepSleepSeconds',  # Qualità sonno profondo
    'sleep_remSleepSeconds',   # Qualità sonno mentale
    'sleep_awakeSleepSeconds', # Frammentazione
    'str_avgStressLevel',      # Stress medio
    'resp_ts_std'              # Stabilità respiro
]

TARGET_SCALE = 100.0