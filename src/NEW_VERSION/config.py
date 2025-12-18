# config.py

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "..", "dataset", "CSV_train")

SEED = 42

# Training
ROUNDS = 90          # federated rounds
LOCAL_EPOCHS = 10    # epochs per client
BATCH_SIZE = 16
LR = 7e-4

# Model
HIDDEN_1 = 128
HIDDEN_2 = 64
DROPOUT = 0.20

# Data
TOP_K_FEATURES = 25
TARGET_SCALE = 100.0


