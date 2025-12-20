# Roadmap Miglioramenti Futuri (Federated Learning)

Questo documento raccoglie le idee per evolvere il progetto da una simulazione federata di base a un sistema più robusto, performante e realistico.

## 1. Qualità dei Dati e Feature Engineering (Alta Priorità)

Questi miglioramenti impattano direttamente la capacità predittiva del modello.

- [ ] **Rimuovere Data Leakage ("Tomorrow")**:
  - La feature `resp_avgTomorrowSleepRespirationValue` è sospetta. Se contiene dati del futuro, va rimossa immediatamente. Se è un errore di naming, va rinominata.
- [ ] **Feature "Weekend"**:
  - Prima di droppare la colonna `day`, estrarre una feature booleana `is_weekend` (Venerdì/Sabato notte). Le abitudini di sonno cambiano drasticamente nel fine settimana.
- [ ] **Sfruttare le Time Series (Avanzato)**:
  - Invece di droppare `hr_time_series`, `stress_time_series`, ecc., estrarre statistiche sintetiche:
    - Deviazione standard (variabilità del battito/stress).
    - Numero di picchi sopra una certa soglia (es. stress > 75).
    - Trend (pendenza della retta di regressione sulla serie).

## 2. Architettura Federata e Privacy (Cose che potremmo mettere nella presentazione sembrano interssanti, ma non penso urgenti)

Miglioramenti per rendere il sistema più fedele a un vero scenario distribuito.

- [ ] **Campionamento dei Client (Client Sampling)**:
  - Invece di allenare su _tutti_ i client a ogni round (full participation), selezionarne un sottoinsieme casuale (es. 10% o 20%).
  - Questo simula meglio la realtà (dispositivi spenti/offline) e riduce il costo computazionale per round.
- [ ] **Differential Privacy (DP)**:
  - Aggiungere rumore gaussiano ai gradienti/pesi inviati dai client al server per impedire la ricostruzione dei dati originali (Membership Inference Attacks).
  - Implementare il "Clipping" della norma dei gradienti.
- [ ] **Secure Aggregation**:
  - Implementare un protocollo dove il server vede solo la somma dei pesi, non i pesi individuali dei client (crittografia omomorfica o masking).

## 3. Monitoraggio e Validazione (Media Priorità)

Per capire meglio cosa succede durante il training "scatola nera".

- [ ] **Logging Granulare per Client**:
  - Ogni client dovrebbe salvare un log locale con la loss per epoca. Utile per individuare client "tossici" o con dati corrotti.
- [ ] **Validazione Locale vs Globale**:
  - Oltre al validation set globale (utenti mai visti), ogni client dovrebbe calcolare il MAE su un proprio validation set locale (ultimi 20% dei giorni).
  - Il server aggrega questi MAE locali per monitorare la "personalizzazione" del modello.

## 4. Ottimizzazione del Modello (Bassa Priorità)

Raffinamenti per spremere gli ultimi punti percentuali di performance.

- [ ] **Personalizzazione (Fine-tuning)**:
  - Dopo il training federato globale, permettere a ogni client di fare qualche epoca extra _solo_ sui propri dati, mantenendo i layer bassi congelati. Questo adatta il modello allo specifico utente.
- [ ] **Learning Rate Decay**:
  - Ridurre il Learning Rate progressivamente durante i round federati per favorire la convergenza fine.
- [ ] **Modelli Ibridi**:
  - Sperimentare con architetture diverse (es. 1D-CNN o LSTM) se si decide di usare le time series grezze invece delle feature aggregate.

## Note Generali

- **Riproducibilità**: Mantenere sempre fissato il `SEED` in config.
- **Versionamento**: Se si cambiano le feature (`TOP_FEATURES`), ricordarsi di rigenerare sempre `federated_scaler.joblib` e `federated_model.pt`, altrimenti l'inferenza fallirà.
