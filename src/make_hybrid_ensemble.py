import pandas as pd
import numpy as np

# Configurazione pesi (Regolabile)
WEIGHT_RF = 0.60  # Il Random Forest ha MAE 10.50 (Molto forte)
WEIGHT_NN = 0.40  # L'Ensemble NN ha MAE ~10.80 (Buono, riduce varianza)

def main():
    print(f"Generating Hybrid Submission (RF: {WEIGHT_RF}, NN: {WEIGHT_NN})...")
    
    # Carica le submission parziali
    try:
        sub_rf = pd.read_csv("submission_rf.csv")
        sub_nn = pd.read_csv("submission_ensemble.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure both submission_rf.csv and submission_ensemble.csv exist.")
        return

    # Ordina per ID per sicurezza
    sub_rf = sub_rf.sort_values('id').reset_index(drop=True)
    sub_nn = sub_nn.sort_values('id').reset_index(drop=True)
    
    # Verifica allineamento
    if not np.array_equal(sub_rf['id'].values, sub_nn['id'].values):
        print("Error: IDs do not match between submissions!")
        return
        
    # Calcolo media pesata (RF label è 'sleep_score', NN label è 'label' -> rinominiamo per chiarezza)
    # Assumiamo che la colonna target sia l'ultima o si chiami 'label' o 'sleep_score'
    col_rf = 'sleep_score' if 'sleep_score' in sub_rf.columns else 'label'
    col_nn = 'label' if 'label' in sub_nn.columns else 'sleep_score'
    
    preds_rf = sub_rf[col_rf].values
    preds_nn = sub_nn[col_nn].values
    
    hybrid_preds = (preds_rf * WEIGHT_RF) + (preds_nn * WEIGHT_NN)
    
    # Arrotondamento e clip
    hybrid_preds = np.clip(hybrid_preds, 0, 100).round().astype(int)
    
    # Salvataggio
    submission = pd.DataFrame({'id': sub_rf['id'], 'label': hybrid_preds})
    submission.to_csv("submission_hybrid.csv", index=False)
    
    print("Hybrid Ensemble saved to 'submission_hybrid.csv'")
    print("Head:")
    print(submission.head())

if __name__ == "__main__":
    main()