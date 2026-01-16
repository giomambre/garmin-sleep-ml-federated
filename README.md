# Garmin Sleep Analysis - Federated Learning Project

This repository contains the source code for our project on classifying sleep and activity patterns using accelerometer data. Our primary focus is on implementing a Federated Learning architecture to ensure data privacy while maintaining high classification accuracy.

## Project Overview

The goal of this project is to build a robust machine learning model that can predict sleep stages without requiring raw user data to be centralized. We simulate a distributed network where individual users (clients) train models locally, and a central server aggregates the knowledge.

## Our Approach

We adopted a privacy-first methodology using Federated Learning. Here is how our system works:

### 1. Distributed Training

The training process simulates multiple clients. Each client holds a specific subset of the data (representing a single user). The clients compute model updates locally based on their private data. Only the model parameters (weights) are sent to the central server, ensuring that raw sensor data never leaves the local environment.

### 2. Ensemble Strategy

To improve the stability and accuracy of our predictions, we moved beyond a single-model approach. We implemented a Cross-Validation Ensemble strategy.

- We split the training data into multiple folds.
- A distinct federated global model is trained for each fold.
- During inference, the predictions from these distinct models are averaged. This helps in reducing overfitting and provides a more generalized solution for unseen users.

### 3. Model Architecture

We use a deep learning model optimized for time-series classification. The model processes raw accelerometer sequences to detect patterns indicative of different sleep states or wakefulness.

## Structure

- **src/ENSEMBLE_VERSION**: Contains the core logic for the federated server, clients, and model definitions.
- **artifacts**: Stores the trained model checkpoints and final submission files.
- **Slide**: Contains project presentation materials and progress reports.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

1. **Install Dependencies**
   The project relies on standard machine learning libraries such as PyTorch, Scikit-learn, Pandas, and NumPy.

   ```bash
   pip install -r requirements.txt
   ```

   _Manual installation of key libraries:_

   ```bash
   pip install torch pandas numpy scikit-learn joblib
   ```

2. **Navigate to the Source Directory**
   All the execution scripts are located in the `src/ENSEMBLE_VERSION` folder.
   ```bash
   cd src/ENSEMBLE_VERSION
   ```

### How to Run

#### 1. Train the Federated Ensemble

This script coordinates the entire training process. It will:

- Split users into folds for cross-validation.
- Simulate the federated learning rounds (training local models and aggregating them).
- Save the best global model for each fold in the `artifacts/` folder.

```bash
python train_federated.py
```

_Note: This process may take some time as it simulates multiple training rounds across different data folds._

#### 2. Generate Predictions (Inference)

Once the models are trained, use this script to generate predictions for the test set. It will:

- Load the trained models for all folds.
- Make predictions using each model.
- Average the results (Ensemble) to produce the final `submission_ensemble.csv`.

```bash
python inference_kaggle.py
```

#### 3. Check Results

Your final submission file will be available at:
`artifacts/submission_ensemble.csv`
