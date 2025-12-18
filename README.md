# Garmin Sleep – Federated MLP (NEW_VERSION)

Questa cartella contiene una versione semplice del training federato basato su una piccola rete neurale (MLP). L’obiettivo è predire il punteggio del sonno della notte successiva usando i dati giornalieri e della notte appena conclusa.

## Cosa fa ogni file (NEW_VERSION)

- `config.py`: tutti gli iperparametri in un posto (round federati, learning rate, batch size, feature usate, path del dataset).
- `data_utils.py`: carica i CSV di tutti gli utenti, pulisce le colonne inutili, riempie i valori mancanti con la media del train, applica lo scaler e salva sia scaler sia medie in `scaler.joblib`.
- `model.py`: definisce l’MLP (due layer densi con dropout) che produce una singola predizione del punteggio.
- `client.py`: esegue il training locale di un client su un batch loader e restituisce i pesi aggiornati e il numero di campioni visti (serve per la media pesata).
- `server.py`: fa la Federated Averaging pesata in base ai campioni per ogni client.
- `train_federated.py`: orchestratore. Carica i dati di tutti gli utenti, prepara feature e scaler, esegue i round federati e calcola un MAE di validazione “onesta” con uno split per utente (GroupShuffleSplit). Salva il modello in `federated_model.pt` e il bundle scaler/medie in `scaler.joblib`.
- `inference_kaggle.py`: carica modello e scaler, riempie i NaN con le medie del train, applica lo scaling e genera `submission.csv` per Kaggle.

## Come usare (passi brevi)

1. Crea e attiva l’ambiente virtuale, poi installa i requisiti:

```powershell
cd "c:\Users\gioma\Desktop\CSI project\garmin-sleep-ml-federated"
python -m venv .venv
..\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Allena il modello federato e ottieni il MAE di validazione per utente:

```powershell
python src/NEW_VERSION/train_federated.py
```

L’output stampa “Hold-out MAE: …” con split per utente, più realistico di uno split casuale.

3. Genera il file di submission per Kaggle:

```powershell
python src/NEW_VERSION/inference_kaggle.py
```

Troverai `submission.csv` nella root del progetto.

## Come funziona il flusso

1. **Caricamento dati**: ogni file `dataset_user_*_train.csv` è associato al suo utente; niente aggregazione per cartella di gruppo.
2. **Pulizia**: colonne irrilevanti vengono droppate (`DROP_COLS`), le feature vengono reindicizzate per avere tutte le colonne attese. I valori mancanti vengono riempiti con la media calcolata sul train.
3. **Scaling**: lo `StandardScaler` viene fittato su tutte le feature del train; scaler e medie vengono salvati in `scaler.joblib` e riusati in inferenza.
4. **Training locale**: ogni client allena l’MLP sui propri dati per `LOCAL_EPOCHS` con Adam e un leggero weight decay per stabilizzare.
5. **Aggregazione**: il server fa una media pesata dei pesi del modello usando il numero di campioni per client.
6. **Validazione**: lo split è per utente (GroupShuffleSplit) per evitare leakage e avere una stima del MAE più vicina allo scenario Kaggle.
7. **Inferenza**: carica `federated_model.pt` e `scaler.joblib`, riempie i NaN con le medie del train, scala le feature e produce `submission.csv` con colonne `id` (day) e `label` (0–100, arrotondato).

## Spiegato passo-passo (anche se non hai mai usato Torch/Sklearn/FL)

- **Preparazione dati**

  - Leggiamo i CSV con Pandas (`pd.read_csv`), togliamo colonne inutili e creiamo un unico DataFrame per utente.
  - Calcoliamo la media di ogni feature sul train e usiamo queste medie per riempire i valori mancanti (imputazione).
  - Applichiamo lo `StandardScaler`: sottrae la media e divide per la deviazione standard di ogni colonna; lo salviamo per riusarlo in inferenza.

- **Che cos’è un MLP (il modello)**

  - È una rete di layer densi: prende un vettore di feature e lo trasforma con moltiplicazioni di matrici e funzioni `ReLU`.
  - Usiamo due layer nascosti (128 e 64 neuroni) più dropout (spegne casualmente neuroni durante il training) per ridurre overfitting.

- **Training di un singolo client (Torch)**

  - Convertiamo i numpy array in tensori Torch (`torch.tensor`).
  - `DataLoader` crea batch e mescola i dati a ogni epoca.
  - Per ogni batch: forward (model(x)), calcolo della loss MAE (`nn.L1Loss()`), backward (`loss.backward()`), e aggiornamento dei pesi con Adam (`optimizer.step()`).
  - Il client restituisce i propri pesi aggiornati e quanti esempi ha visto.

- **Federated Averaging (server)**

  - Riceve i pesi di tutti i client e li combina con una media pesata sul numero di esempi: più dati = più peso.
  - Aggiorna il modello globale con questa media.
  - Un “round” = tutti i client (o un sottoinsieme) allenano localmente + una media globale.

- **Validazione onesta**

  - Usiamo `GroupShuffleSplit`: tutti i record di un utente finiscono o in train o in validation, mai spezzati.
  - Calcoliamo il MAE sul validation per stimare come il modello generalizza su utenti mai visti.

- **Inferenza (predizione su test)**
  - Carichiamo modello e bundle scaler+medie salvati.
  - Applichiamo la stessa imputazione (medie) e lo stesso scaler usati in training, poi passiamo i dati al modello per ottenere le predizioni, che vengono riportate su scala 0–100 e arrotondate.

## Perché queste scelte (e come ci siamo arrivati)

- **Split per utente (GroupShuffleSplit)**: lo split casuale mescolava lo stesso utente in train/val e sovrastimava il MAE; con split per utente il MAE locale è più vicino al comportamento Kaggle.
- **Imputazione con media + scaler salvato**: prima i NaN andavano a 0 e lo scaler veniva ricreato; ora salviamo medie e scaler del train e li riapplichiamo in inferenza per evitare drift tra train e test.
- **Feature set**: partiamo dalle top feature della Random Forest baseline (importanza >0.01) e aggiungiamo tre feature respiratorie per più segnale. Se modifichi la lista, rigenera scaler e modello.
- **MLP semplice ma più capiente**: due layer (128, 64) con dropout 0.20. Prima era più piccolo e underfit; l’aumento di capacità ha ridotto il MAE hold-out.
- **Weight decay e LR più basso**: weight decay 1e-4 + LR 7e-4 danno training più stabile su tabellare con poche feature; il LR è stato ridotto dopo aver osservato MAE più alto con 1e-3.
- **Più round federati (ROUNDS=90)**: servono più passi globali per convergere; il numero di round non è il numero di client, ma di iterazioni di aggregazione.
- **Media pesata in FedAvg**: ogni client contribuisce proporzionalmente ai propri campioni; prima la media semplice sbilanciava se un client aveva pochi dati.

## Note utili

- Se aggiungi o togli feature, aggiorna `TOP_FEATURES` sia in `train_federated.py` sia in `inference_kaggle.py`, poi riesegui il training per rigenerare modello e scaler.
- Assicurati di lanciare i comandi dalla root del progetto, così i path relativi ai dataset funzionano.
- Il numero di round (`ROUNDS`) è il numero di iterazioni di federated averaging, non il numero di client.
- Per debug rapido: riduci `ROUNDS` e `LOCAL_EPOCHS`; per massimizzare le prestazioni, puoi aumentare `ROUNDS` o ridurre leggermente il learning rate.

Con questi passi dovresti riuscire a: preparare i dati, allenare il modello federato, validarlo in modo realistico e generare un file di submission pronto per Kaggle. Se qualcosa non torna, guarda prima log del MAE hold-out e verifica di aver rigenerato `federated_model.pt` e `scaler.joblib` dopo ogni modifica.
