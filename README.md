# Garmin Sleep – Federated Machine Learning Project

Questo progetto implementa un sistema di **Federated Learning** per la previsione del punteggio del sonno ("sleep score") basato su dati wearable. L'obiettivo è addestrare modelli predittivi garantendo la privacy degli utenti: i dati grezzi non lasciano mai i dispositivi locali, ma vengono scambiati solo i pesi del modello e statistiche aggregate.

## 📂 Struttura del Progetto

Il codice è organizzato per essere modulare e mantenere pulita la directory principale:

- **`src/ENSEMBLE_VERSION/`**: Implementazione principale basata su un Ensemble di Reti Neurali (MLP). Utilizza una strategia di K-Fold Cross-Validation federata.
- **`src/RF_VERSION/`**: Implementazione basata su Random Forest Federata (Distributed Bagging).
- **`src/DATASET/`**: Directory contenente i dati di training e test.
- **`artifacts/`**: Cartella centralizzata dove vengono salvati tutti i modelli addestrati (`.pt`, `.joblib`) e i file di submission generati.
- **`requirements.txt`**: Librerie necessarie (PyTorch, Scikit-Learn, Pandas, ecc.).

## 🧠 Modelli e Strategie

### Federated Neural Network (Ensemble)
Il modello principale consiste in una rete neurale profonda con tre layer densi, LayerNorm e Dropout. Per massimizzare la capacità di generalizzazione, il sistema addestra 5 modelli indipendenti su diversi subset di utenti (5-fold). La predizione finale viene calcolata come media pesata basata sulle performance di validazione di ogni fold.

### Feature Engineering
È stato implementato un sistema di estrazione di feature dalle serie temporali (battito cardiaco, respirazione, stress) per catturare statistiche rilevanti come trend, variabilità e picchi, migliorando significativamente il MAE rispetto all'uso dei soli dati tabellari semplici.

## 🚀 Istruzioni per l'uso

Tutti i comandi devono essere eseguiti dalla root del progetto con l'ambiente virtuale attivo.

1. **Addestramento dell'Ensemble**:
   ```bash
   python src/ENSEMBLE_VERSION/train_federated.py
   ```
   Questo script esegue il training federato per i 5 fold e salva i modelli migliori in `artifacts/`.

2. **Generazione della Submission Pesata (Consigliata)**:
   ```bash
   python src/make_weighted_ensemble.py
   ```
   Il comando caricherà i modelli salvati e genererà `submission_weighted.csv` all'interno della cartella `artifacts/`.

3. **Random Forest (Alternativa)**:
   ```bash
   python src/RF_VERSION/train_federated.py
   ```

## 📊 Risultati Previsti
Il sistema è stato ottimizzato per ridurre l'overfitting locale (specialmente nella versione RF). La validazione incrociata indica un MAE atteso inferiore a **11.0**, posizionando il modello competitivamente per la classifica Kaggle.