"""Run multiple Flower clients from a single script using the unified client.

This runner starts multiple instances of the single `client_flwr.py` placed at
the repo root. Datasets are expected in `dataset/` as `dataset_G.csv`,
`dataset_I.csv`, `dataset_L.csv`, `dataset_S.csv`.

Usage:
    python3 server_flwr.py
    python3 run_clients.py --server_address localhost:8080

Options:
    --clients  Optional comma-separated list (default: G,I,L,S)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Try to pick default server address from server config if available
try:
    from server.config import SERVER_ADDRESS as DEFAULT_SERVER_ADDRESS
except Exception:
    DEFAULT_SERVER_ADDRESS = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CLIENTS = ["G", "I", "L", "S"]


def start_clients(server_address: str, clients: list[str]):
    procs = []
    client_script = BASE_DIR / "client_flwr.py"
    if not client_script.exists():
        print(f"Error: unified client script not found: {client_script}")
        return procs

    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    for cid in clients:
        dataset_path = BASE_DIR / "dataset" / f"dataset_{cid}.csv"
        if not dataset_path.exists():
            print(f"Warning: dataset not found for client {cid}: {dataset_path}. Skipping client.")
            continue

        logfile = logs_dir / f"client_{cid}.log"
        cmd = [sys.executable, str(client_script), "--server_address", server_address, "--csv_path", str(dataset_path)]

        # create a per-client working directory so training_log.csv and other
        # artifacts do not collide when running multiple clients concurrently
        client_workdir = BASE_DIR / "client_workdirs" / cid
        client_workdir.mkdir(parents=True, exist_ok=True)

        print(f"Starting client {cid}: {cmd}")
        f = open(logfile, "w")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(client_workdir))
        procs.append((cid, proc, logfile))
        print(f"  PID={proc.pid}, log={logfile}")

    return procs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple Flower clients using unified client_flwr.py")
    parser.add_argument("--server_address", required=False, help="Server address (default from server.config if available)", default=None)
    parser.add_argument("--clients", required=False, help="Comma-separated client ids to run (G,I,L,S)", default=None)
    args = parser.parse_args()

    server_address = args.server_address or DEFAULT_SERVER_ADDRESS
    if server_address is None:
        parser.error("--server_address is required if server.config.SERVER_ADDRESS is not available")

    clients = DEFAULT_CLIENTS if args.clients is None else [c.strip() for c in args.clients.split(",") if c.strip()]
    procs = start_clients(server_address, clients)

    if not procs:
        print("No clients started.")
        sys.exit(1)

    print("\nAll clients started. To stop them, kill their PIDs or use `pkill -f client_flwr.py`.")
    print("Tail logs with e.g.: tail -F logs/client_G.log logs/client_I.log")
