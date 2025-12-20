# Garmin Sleep – Federated MLP (NEW_VERSION)

Questa cartella contiene una versione semplice del training federato basato su una piccola rete neurale (MLP). L'obiettivo e' predire il punteggio del sonno della notte successiva usando i dati giornalieri e della notte appena conclusa.

## Cosa fa ogni file (NEW_VERSION)

- `config.py`: tutti gli iperparametri in un posto (round federati, learning rate, batch size, feature usate, path del dataset).
- `data_utils.py`: contiene `FederatedScaler`, una classe che calcola medie e deviazioni standard globali aggregando statistiche parziali dai client (somme, somme dei quadrati, conteggi) senza mai condividere i dati grezzi.
- `model.py`: definisce l'MLP (tre layer densi 256->128->64 con LayerNorm e dropout) che produce una singola predizione del punteggio.
- `client.py`: esegue il training locale di un client su un batch loader con SmoothL1Loss (Huber) e restituisce i pesi aggiornati e il numero di campioni visti (serve per la media pesata).
- `server.py`: fa la Federated Averaging pesata in base ai campioni per ogni client.
- `train_federated.py`: orchestratore. Carica i dati, divide gli utenti in train/val. Usa `FederatedScaler` per calcolare le statistiche globali in modo privacy-preserving. Esegue i round federati e valida ogni 5 round salvando il modello migliore (`best_federated_model.pt`).
- `inference_kaggle.py`: carica modello e `federated_scaler.joblib`; applica lo stesso preprocessing (imputazione e scaling con parametri globali) e genera `submission.csv`.

## Come usare (passi brevi)

1. Crea e attiva l'ambiente virtuale, poi installa i requisiti:

```powershell
cd "c:\Users\gioma\Desktop\CSI project\garmin-sleep-ml-federated"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Allena il modello federato e ottieni il MAE di validazione per utente:

```powershell
python src/NEW_VERSION/train_federated.py
```

L'output mostrerà il calcolo delle statistiche federate, poi il MAE ogni 5 round.

3. Genera il file di submission per Kaggle:

```powershell
python src/NEW_VERSION/inference_kaggle.py
```

Troverai `submission.csv` nella root del progetto.

## Come funziona il flusso

1. **Caricamento dati**: ogni file `dataset_user_*_train.csv` e' associato al suo utente.
2. **Split Train/Val**: gli utenti vengono divisi in due gruppi (80/20) **prima** di toccare i dati.
3. **Statistiche Federate (Privacy-Preserving)**:
   - Invece di unire i dati, ogni client calcola localmente: somma delle feature, somma dei quadrati e conteggio.
   - Il server aggrega questi numeri per ottenere Media e Deviazione Standard globali.
   - **Nessun dato grezzo lascia mai il client.**
   - Salviamo queste statistiche in `federated_scaler.joblib`.
4. **Preprocessing Locale**: Ogni client usa le statistiche globali ricevute per imputare i mancanti e scalare i propri dati.
5. **Training locale**: ogni client allena l'MLP sui propri dati per `LOCAL_EPOCHS`.
6. **Aggregazione**: il server fa una media pesata dei pesi del modello.
7. **Validazione periodica**: ogni 5 round calcoliamo il MAE sul validation set.
8. **Inferenza**: carica il modello migliore e lo scaler federato, applica lo stesso preprocessing e produce `submission.csv`.

## Spiegato passo-passo (anche se non hai mai usato Torch/Sklearn/FL)

- **Preparazione dati (Federated Statistics)**

  - Per scalare i dati (renderli confrontabili tra utenti diversi) serve la media e la varianza globale.
  - Invece di inviare i dati al server, inviamo solo i totali parziali. La matematica garantisce che la media calcolata dai totali è identica alla media calcolata sui dati uniti, ma la privacy è salva.

- **Che cos'e' un MLP (il modello)**

  - E' una rete di layer densi: prende un vettore di feature e lo trasforma con moltiplicazioni di matrici e funzioni `ReLU`.
  - Usiamo tre layer nascosti (256, 128 e 64 neuroni) con LayerNorm (normalizza le attivazioni per stabilizzare il training) e dropout 15% per ridurre overfitting.

- **Training di un singolo client (Torch)**

  - Convertiamo i numpy array in tensori Torch (`torch.tensor`).
  - `DataLoader` crea batch e mescola i dati a ogni epoca.
  - Per ogni batch: forward (model(x)), calcolo della loss SmoothL1 (Huber, meno sensibile a outlier), backward (`loss.backward()`), e aggiornamento dei pesi con Adam (`optimizer.step()`).
  - Il client restituisce i propri pesi aggiornati e quanti esempi ha visto.

- **Federated Averaging (server)**

  - Riceve i pesi di tutti i client e li combina con una media pesata sul numero di esempi: piu' dati = piu' peso.
  - Aggiorna il modello globale con questa media.
  - Un "round" = tutti i client (o un sottoinsieme) allenano localmente + una media globale.

- **Validazione onesta**

  - Usiamo `GroupShuffleSplit`: tutti i record di un utente finiscono o in train o in validation, mai spezzati.
  - Calcoliamo il MAE sul validation per stimare come il modello generalizza su utenti mai visti.

- **Inferenza (predizione su test)**
  - Carichiamo modello e bundle scaler+medie salvati.
  - Applichiamo la stessa imputazione (medie) e lo stesso scaler usati in training, poi passiamo i dati al modello per ottenere le predizioni, che vengono riportate su scala 0-100 e arrotondate.

## Perche' queste scelte (e come ci siamo arrivati)

- **Split per utente (GroupShuffleSplit)**: lo split casuale mescolava lo stesso utente in train/val e sovrastimava il MAE; con split per utente il MAE locale e' piu' vicino al comportamento Kaggle.
- **Imputazione con media + scaler salvato**: prima i NaN andavano a 0 e lo scaler veniva ricreato; ora salviamo medie e scaler del train e li riapplichiamo in inferenza per evitare drift tra train e test. I placeholder -1/-2 vengono convertiti in NaN prima di imputare.
- **Feature set**: partiamo dalle top feature della Random Forest baseline (importanza >0.01) e aggiungiamo tre feature respiratorie per piu' segnale. Se modifichi la lista, rigenera scaler e modello.
- **MLP piu' capiente con LayerNorm**: tre layer (256, 128, 64) con LayerNorm e dropout 0.15. LayerNorm invece di BatchNorm perche' funziona anche con batch di 1 elemento.
- **SmoothL1Loss (Huber)**: meno sensibile agli outlier rispetto a L1Loss pura; migliora la stabilita' del training.
- **Weight decay e LR**: weight decay 1e-5 + LR 5e-4 danno training stabile su tabellare con poche feature.
- **Piu' round federati (ROUNDS=150)**: servono piu' passi globali per convergere; il numero di round non e' il numero di client, ma di iterazioni di aggregazione.
- **Piu' epoche locali (LOCAL_EPOCHS=15)**: ogni client allena piu' a lungo prima di aggregare.
- **Media pesata in FedAvg**: ogni client contribuisce proporzionalmente ai propri campioni; prima la media semplice sbilanciava se un client aveva pochi dati.

## Note utili

- Se aggiungi o togli feature, aggiorna `TOP_FEATURES` sia in `train_federated.py` sia in `inference_kaggle.py`, poi riesegui il training per rigenerare modello e scaler.
- Assicurati di lanciare i comandi dalla root del progetto, cosi' i path relativi ai dataset funzionano.
- Il numero di round (`ROUNDS`) e' il numero di iterazioni di federated averaging, non il numero di client.
- Per debug rapido: riduci `ROUNDS` e `LOCAL_EPOCHS`; per massimizzare le prestazioni, puoi aumentare `ROUNDS` o ridurre leggermente il learning rate.

Con questi passi dovresti riuscire a: preparare i dati, allenare il modello federato, validarlo in modo realistico e generare un file di submission pronto per Kaggle. Se qualcosa non torna, guarda prima log del MAE hold-out e verifica di aver rigenerato `federated_model.pt` e `scaler.joblib` dopo ogni modifica.
