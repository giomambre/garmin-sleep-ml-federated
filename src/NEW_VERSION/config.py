# config.py

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "..", "dataset", "CSV_train")

SEED = 42

# Training
ROUNDS = 150         # federated rounds (più iterazioni per convergere)
LOCAL_EPOCHS = 15    # epochs per client
BATCH_SIZE = 16
LR = 5e-4            # LR più basso per convergenza stabile

# Model
HIDDEN_1 = 256
HIDDEN_2 = 128
HIDDEN_3 = 64
DROPOUT = 0.15       # dropout ridotto (meno underfit)

# Data
TOP_K_FEATURES = 25
TARGET_SCALE = 100.0


