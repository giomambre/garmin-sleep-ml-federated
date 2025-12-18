# Migliorie consigliate (ordine di priorità)

## 1) Feature engineering leggera (più segnale)

- **Rapporti sonno**: aggiungi colonne derivate
  - `deep_ratio = sleep_deepSleepSeconds / sleep_sleepTimeSeconds`
  - `rem_ratio = sleep_remSleepSeconds / sleep_sleepTimeSeconds`
  - `awake_ratio = sleep_awakeSleepSeconds / sleep_sleepTimeSeconds`
- **Delta cardio/respirazione**: cattura variazioni tra veglia e sonno
  - `hr_delta = sleep_avgHeartRate - hr_restingHeartRate`
  - `resp_delta = resp_avgSleepRespirationValue - resp_avgWakingRespirationValue`
- **Stress normalizzato**: stress rispetto al tempo di sonno o attività
  - `stress_per_sleep = str_avgStressLevel / (sleep_sleepTimeSeconds + 1)`
  - `stress_per_activity = str_avgStressLevel / (act_totalCalories + 1)`

**Perché**: rapporti e delta riducono l’effetto della scala assoluta e danno più informazione di contesto. **Come**: calcola queste colonne in `prepare_data` (prima dello scaler) e aggiungile a `TOP_FEATURES` in train e inferenza. Ricorda di rigenerare scaler e modello.

## 2) Client sampling per round

- Allena solo una frazione di client a ogni round (es. 60–70%).
- **Perché**: riduce overfitting ai client visti, simula meglio FL reale e aggiunge rumore benefico all’ottimizzazione.
- **Come**: in `train_federated.py`, prima del loop interno, campiona casualmente un sottoinsieme di `clients.items()` con `random.sample` o `np.random.choice`.

## 3) Scheduler del learning rate

- Usa un `StepLR` o `CosineAnnealingLR` sull’optimizer locale.
- **Perché**: step finali più piccoli raffinano la convergenza; utile con più round.
- **Come**: in `train_local`, crea uno scheduler (es. `StepLR(optimizer, step_size=3, gamma=0.7)`) e chiama `scheduler.step()` a fine epoch.

## 4) Regularizzazione aggiuntiva

- Dropout a 0.25 e weight_decay a 5e-4 se vedi gap tra hold-out e pubblico.
- **Perché**: più robustezza a shift di distribuzione.
- **Come**: modifica `config.py` e `model.py` (dropout) e `client.py` (weight_decay).

## 5) Ensemble semplice (evitabile non possiamo usare GBDT mi sa )


- Media tra il MLP federato e un modello GBDT (XGBoost/LightGBM) addestrato centralmente sulle stesse feature.
- **Perché**: su tabellare gli alberi catturano interazioni non lineari, spesso abbassano il MAE.
- **Come**: addestra il GBDT su feature già scalate/imputate, salva le predizioni su `x_test`, poi fai media (o peso 0.5/0.5) con le predizioni del MLP prima di salvare `submission.csv`.

## 6) Aumentare round con LR leggermente più basso

- Porta `ROUNDS` a 110–120 e `LR` a 6e-4.
- **Perché**: più passi globali con passo più fine possono ridurre il MAE, specie dopo feature extra.
- **Come**: modifica `config.py` e riesegui training/inferenza.

## 7) Calibrazione del target

- Applica SmoothL1Loss (Huber) al posto di L1 pura.
- **Perché**: meno sensibile agli outlier, utile se il target ha code.
- **Come**: in `client.py` usa `nn.SmoothL1Loss(beta=1.0)` al posto di `nn.L1Loss()`.

Note generali:

- Ogni volta che cambi feature: aggiorna `TOP_FEATURES` in train e inferenza, rigenera scaler (`scaler.joblib`) e modello (`federated_model.pt`).
- Non versionare dataset, artifacts, **pycache**, venv: tieni `.gitignore` aggiornato.
- Testa sempre il MAE hold-out per utente dopo ogni modifica prima di inviare a Kaggle.
