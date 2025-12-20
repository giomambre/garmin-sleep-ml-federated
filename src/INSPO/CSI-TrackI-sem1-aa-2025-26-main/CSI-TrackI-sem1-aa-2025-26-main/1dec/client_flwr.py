import os
import sys
import argparse
from pathlib import Path

# Ensure repo root is on sys.path so imports work when cwd != repo root
repo_root = Path(__file__).resolve().parent
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

try:
    import flwr as fl
except Exception as e:
    print("Error importing Flower (flwr). Make sure the 'flwr' package is installed in your Python environment.")
    raise

from shared.logger import setup_logger
logger = setup_logger(__name__)

# Defer heavy imports (model, data) until runtime to make startup errors clearer
Model = None
load_dataset = None


class FlwClient(fl.client.NumPyClient):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        X_train, X_test, y_train, y_test = load_dataset(self.csv_path)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        input_size = self.X_train.shape[1]
        self.model = Model(input_size)
        self.model.fit_scaler(self.X_train)
        logger.info(f"Initialized client. Reading from csv: {csv_path}")

    def get_parameters(self, config):
        logger.info("Getting model parameters to send to server")
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        epochs = int(config.get("local_epochs", 3))
        batch_size = int(config.get("batch_size", 8))
        logger.info(f"Training on {len(self.X_train)} samples for {epochs} epochs")

        # leak_corr_threshold=None → disabilita la leak-check per velocizzare il FL
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            leak_corr_threshold=None,
        )
        logger.info("Training complete. Sending updated weights to server")
        return self.model.get_weights(), len(self.X_train), {}


    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc, errors = self.model.evaluate(self.X_test, self.y_test)
        logger.info(f"Evaluation results: Loss={loss:.6f}, Acc={acc:.6f}, Errors={errors}/{len(self.y_test)}")
        return loss, len(self.X_test), {"accuracy": acc, "errors": errors}


def start_flower_client(server_address: str, csv_path: str = None, client_id: str = None):
    if csv_path is None:
        if client_id is None:
            raise ValueError("Provide either --csv_path or --client_id")
        csv_path = str(repo_root.joinpath('dataset', f'dataset_{client_id}.csv'))
    # ensure dataset exists and is accessible
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # import model and data loader here (may be heavy)
    global Model, load_dataset
    if Model is None or load_dataset is None:
        try:
            from model.model import Model as _Model
            from data.load_data import load_dataset as _load_dataset
        except Exception:
            logger.exception("Failed to import Model or load_dataset")
            raise
        Model = _Model
        load_dataset = _load_dataset

    client = FlwClient(csv_path)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a Flower federated client.")
    parser.add_argument("--server_address", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=False, default=None)
    parser.add_argument("--client_id", type=str, required=False, default=None, help="Client short id (G,I,L,S) to pick dataset/dataset_<id>.csv")
    args = parser.parse_args()
    # allow dataset files in repo root `dataset/` or absolute path
    if args.csv_path is None and args.client_id is not None:
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset', f'dataset_{args.client_id}.csv'))
    else:
        csv_path = args.csv_path
    start_flower_client(server_address=args.server_address, csv_path=csv_path, client_id=args.client_id)
