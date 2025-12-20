"""Server configuration constants.

Keep training / federation-related constants here. These variables are
imported by `server/server_flwr.py` and `server/strategy.py`.
"""

# Number of federated learning rounds to run
NUM_ROUNDS = 10

# Address the Flower server listens on (host:port)
SERVER_ADDRESS = "localhost:8080"

# Directory where the server persists the global model
MODEL_DIR = "checkpoints"

# Minimum available clients and fit/evaluate minimums
MIN_CLIENTS = 4
MIN_FIT_CLIENTS = 4
MIN_EVALUATE_CLIENTS = 4

# Local training defaults sent to clients
LOCAL_EPOCHS = 3
BATCH_SIZE = 8

# Fraction of clients used for fit/evaluate each round (1.0 = all available)
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0
