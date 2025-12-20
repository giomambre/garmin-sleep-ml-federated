# Installation Guide

To set up the project, follow these steps:

1. **Navigate to the desired folder** where you want to download the project
    Use the `cd` command to navigate to the folder where you downloaded the project from GitHub. 
    For example:
    ```bash
    cd /pathtoyourfolder
    ```

2. **Create a virtual environment** named for example flvenv
    ```bash
    python3 -m venv flvenv
    ```
    To deactivate it
    ```bash
    source deactivate
    ```

3. **Activate the virtual environment**
    ```bash
    source flcsi/bin/activate
    ```

4. **Upgrade pip**
    ```bash
    pip install --upgrade pip
    ```

4. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

5. **Run server and clients**
    ```bash
    python3 server/server_flwr.py
    python3 run_clients.py --server_address localhost:8080
    ```