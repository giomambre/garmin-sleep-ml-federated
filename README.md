# Progetto Federated Learning - Previsione Qualità del Sonno 😴

Ciao! Questa è la repository del mio progetto per il corso di CSI. L'obiettivo è prevedere il punteggio del sonno ("sleep score") degli utenti Garmin usando tecniche di Machine Learning Federato.

In pratica, cerchiamo di imparare dai dati di tutti gli utenti **senza mai guardare i loro dati personali**, ma scambiandoci solo i "pesi" dei modelli o delle statistiche aggregate. Privacy first! 🔒

## 🧠 L'Idea in Breve

Ho sperimentato con diversi approcci per capire quale funzionasse meglio in un contesto federato, dove ogni utente ha pochi dati (qualche settimana di rilevazioni).

Alla fine ho sviluppato tre "versioni" del sistema:

1.  **Rete Neurale (Ensemble)**: È il mio modello principale. Invece di allenare una sola rete, ne alleno 5 diverse su gruppi di utenti diversi e poi faccio la media. Funziona decisamente meglio perché riduce gli errori casuali.
2.  **Random Forest Federata**: Volevo vedere come se la cavavano gli alberi decisionali. Ogni utente si costruisce i suoi alberelli e poi il server li mette tutti insieme in una foresta gigante. È robusto, ma tende a generalizzare un po' peggio della rete neurale.
3.  **L'Ibrido**: Uno script che prende il meglio dei due mondi, facendo una media pesata tra le predizioni della Rete Neurale e quelle della Random Forest.

## 📂 Com'è organizzato il codice

- **`src/ENSEMBLE_VERSION/`**: Qui c'è la "magia". È il codice della rete neurale federata.
    - `client.py`: Simula l'allenamento sul dispositivo dell'utente.
    - `server.py`: Il "cervello" centrale che aggrega i modelli.
    - `train_federated.py`: Lo script che fa partire tutto il processo di training.
- **`src/RF_VERSION/`**: La versione con Random Forest (più semplice, ma utile come confronto).
- **`src/make_weighted_ensemble.py`**: Lo script che crea la submission finale "intelligente", dando più peso ai modelli che hanno performato meglio durante i test.

## 🛠️ Cosa ho fatto sui dati (Feature Engineering)

La parte difficile era che alcune colonne contenevano liste di numeri (es. la serie temporale del battito cardiaco durante la notte).
Invece di ignorarle, ho creato una funzione che estrae delle statistiche utili da queste liste:
- Qual è stato il picco massimo?
- Quanto è variato il battito? (Deviazione Standard)
- Qual è la media?

Questo "trucco" ha migliorato un sacco le prestazioni del modello! 🚀

Inoltre, per gestire i dati mancanti in modo federato (senza vederli), ho creato uno `Scaler` speciale che calcola la media globale chiedendo solo i totali parziali ai client. Matematica semplice ma efficace.

## 🚀 Come lanciarlo

Se vuoi replicare i miei risultati (o generare i file per Kaggle), ecco i comandi da lanciare nel terminale:

1.  **Attiva l'ambiente**: Assicurati di essere nel venv.
2.  **Allena l'Ensemble (ci mette un po')**:
    ```powershell
    python src/ENSEMBLE_VERSION/train_federated.py
    ```
    Questo creerà 5 modelli diversi nella cartella principale.
3.  **Genera la Submission Migliore**:
    ```powershell
    python src/make_weighted_ensemble.py
    ```
    Troverai un file `submission_weighted.csv` pronto per essere caricato.

## 📊 Risultati

Dai miei test locali (validazione incrociata), il modello sbaglia in media di circa **10.8 punti** (MAE). Considerando che il punteggio va da 0 a 100, direi che è un risultato onesto, soprattutto considerando i vincoli del federato!

---
*Progetto sviluppato per il corso di CSI - Dicembre 2025*
