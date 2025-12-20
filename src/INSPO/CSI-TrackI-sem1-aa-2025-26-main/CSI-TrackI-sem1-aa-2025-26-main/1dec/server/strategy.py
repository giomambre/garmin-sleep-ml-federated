import os
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict
from .config import (
    MIN_CLIENTS, MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS,
    FRACTION_FIT, FRACTION_EVALUATE,
    LOCAL_EPOCHS, BATCH_SIZE, MODEL_DIR
)


os.makedirs(MODEL_DIR, exist_ok=True)


class SaveModelStrategy(FedAvg):
    """FedAvg strategy with custom behavior for this project.

    - Saves the global model weights to `MODEL_DIR` after each fit round.
    - Aggregates evaluation metrics to compute a weighted accuracy and sum errors.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_fit_config_fn(self, server_round: int) -> Dict[str, int]:
        """Return configuration (dict) sent to clients for local training.

        This must match the keys expected by the clients (`local_epochs`, `batch_size`).
        """
        return {"local_epochs": LOCAL_EPOCHS, "batch_size": BATCH_SIZE}

    def aggregate_fit(self, server_round, results, failures):
        """Call base aggregation then persist the aggregated global weights.

        The aggregated parameters are converted to ndarrays and saved as a
        NumPy `.npz` file for inspection or later loading.
        """
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            file_path = os.path.join(MODEL_DIR, "global_model.npz")
            np.savez(file_path, *weights)
            print(f"[Server] Saved global model at round {server_round} to {file_path}")

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results to compute weighted accuracy and sum errors.

        Uses `num_examples` as weights for accuracy averaging and sums any
        `errors` metric reported by clients.
        """
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        total_examples = 0
        acc_weighted_sum = 0.0
        total_errors = 0

        for client_proxy, eval_res in results:
            n = eval_res.num_examples
            total_examples += n

            acc = eval_res.metrics.get("accuracy")
            if acc is not None:
                acc_weighted_sum += acc * n

            err = eval_res.metrics.get("errors")
            if err is not None:
                try:
                    total_errors += int(err)
                except Exception:
                    pass

        weighted_accuracy = None
        if total_examples > 0 and acc_weighted_sum > 0:
            weighted_accuracy = acc_weighted_sum / total_examples

        acc_str = f"{weighted_accuracy:.6f}" if weighted_accuracy is not None else "n/a"
        loss_str = f"{aggregated_loss:.6f}" if aggregated_loss is not None else "n/a"
        print(f"[Server] Round {server_round} — loss: {loss_str}, acc(w): {acc_str}, errors(sum): {total_errors}")

        return aggregated_loss, {"accuracy": weighted_accuracy, "errors": total_errors}


def get_strategy() -> SaveModelStrategy:
    """Factory returning the tuned strategy instance for the server."""
    return SaveModelStrategy(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_available_clients=MIN_CLIENTS,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
    )
