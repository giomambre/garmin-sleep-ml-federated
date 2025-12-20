
"""Server entrypoint for Flower federated learning.

This file supports two invocation styles:
- `python -m server.server_flwr` (recommended)
- `python3 server/server_flwr.py` (convenient)

When executed as a script, the parent repository root is added to
`sys.path` so absolute imports like `server.config` work correctly.
"""

import os
import sys

# If running the script directly, ensure the repository root is on sys.path
# so `import server.config` resolves correctly.
if __package__ is None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import flwr as fl
from server.config import SERVER_ADDRESS, NUM_ROUNDS
from server.strategy import get_strategy


def main():
    """Start the Flower server with chosen strategy and config."""
    strategy = get_strategy()
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
