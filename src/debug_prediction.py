"""
DEBUG della predizione - trova il problema esatto
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import os
from datetime import datetime
import traceback

def debug_data_loading(filepath: str):
    """Analizza in dettaglio il file dati."""
    print("=" * 60)
    print("DEBUG CARICAMENTO DATI")
    print("=" * 60)
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Errore caricamento file: {e}")
        return None
    
    print(f"File: {filepath}")
    print(f"Righe: {len(df)}")
    print(f"Colonne: {len(df.columns)}")
    print(f"Nomi colonne: {list(df.columns)}")
    print("\n" + "-" * 60)
    
    # Data types - correggi questa parte
    print("DATA TYPES:")
    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].iloc[0] if len(df) > 0 else "N/A"
        # CORREZIONE: converti dtype a string prima di formattare
        dtype_str = str(dtype)
        print(f"  {col:20} -> {dtype_str:10} | Esempio: {sample}")
    
    print("\n" + "-" * 60)
    
    # Statistiche base solo per colonne numeriche
    print("STATISTICHE NUMERICHE:")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        print(numeric_df.describe())
    else:
        print("Nessuna colonna numerica trovata")
    
    print("\n" + "-" * 60)
    
    # Controlla valori null
    print("VALORI NULL:")
    null_counts = df.isnull().sum()
    has_nulls = False
    for col, count in null_counts.items():
        if count > 0:
            has_nulls = True
            print(f"  {col}: {count} valori null ({count/len(df)*100:.2f}%)")
    
    if not has_nulls:
        print("  Nessun valore null trovato")
    
    return df

def debug_model_loading(model_path: str):
    """Analizza il modello caricato."""
    print("\n" + "=" * 60)
    print("DEBUG CARICAMENTO MODELLO")
    print("=" * 60)
    
    print(f"Modello: {model_path}")
    
    # Carica metadata
    metadata_path = model_path.replace('.h5', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ Metadata validi")
            print(f"   Nome modello: {metadata.get('model_name', 'N/A')}")
            print(f"   Sequence length: {metadata.get('sequence_length', 'N/A')}")
            print(f"   Numero features: {metadata.get('n_features', 'N/A')}")
            
            if 'feature_names' in metadata:
                print(f"   Feature names: {metadata.get('feature_names')}")
        except Exception as e:
            print(f"❌ Errore lettura metadata: {e}")
    else:
        print("❌ Metadata NON TROVATO")
    
    # Prova a caricare il modello
    model = None
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"\n✅ Modello caricato correttamente")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Numero layer: {len(model.layers)}")
        
        # Estrai info shape
        if len(model.input_shape) == 3:
            seq_len = model.input_shape[1]
            n_feats = model.input_shape[2]
            print(f"   Sequence length attesa: {seq_len}")
            print(f"   Features attese: {n_feats}")
            
            # Aggiorna metadata se necessario
            if 'sequence_length' not in metadata:
                metadata['sequence_length'] = int(seq_len)
            if 'n_features' not in metadata:
                metadata['n_features'] = int(n_feats)
        
    except Exception as e:
        print(f"\n❌ ERRORE caricamento modello: {str(e)}")
        print("Tentativo di caricamento con custom_objects...")
        
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'mse': tf.keras.metrics.MeanSquaredError(),
                    'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                    'mae': tf.keras.metrics.MeanAbsoluteError(),
                    'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError(),
                }
            )
            print(f"✅ Modello caricato con custom_objects")
        except Exception as e2:
            print(f"❌ Anche custom_objects fallito: {e2}")
    
    return model, metadata

def debug_prepare_input(df: pd.DataFrame, sequence_length: int = 20, n_features: int = 9):
    """Prepara input e mostra esattamente cosa viene passato."""
    print("\n" + "=" * 60)
    print("DEBUG PREPARAZIONE INPUT")
    print("=" * 60)
    
    if df is None:
        print("❌ DataFrame è None")
        return None, []
    
    if len(df) < sequence_length:
        print(f"⚠️  ATTENZIONE: Solo {len(df)} righe, servono almeno {sequence_length}")
        print(f"   Uso tutte le {len(df)} righe disponibili")
        sequence_length = len(df)
    
    recent_data = df.iloc[-sequence_length:]
    
    # Identifica tutte le colonne numeriche tranne 'price' (target)
    all_numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != 'price']
    
    print(f"Colonne numeriche disponibili (escluso 'price'): {len(all_numeric_cols)}")
    print(f"Lista: {all_numeric_cols}")
    
    # Seleziona le features
    if len(all_numeric_cols) >= n_features:
        # Abbiamo abbastanza features, prendi le prime n_features
        feature_columns = all_numeric_cols[:n_features]
        print(f"✅ Features selezionate ({len(feature_columns)}):")
    else:
        # Non abbiamo abbastanza features
        print(f"⚠️  Solo {len(all_numeric_cols)} features numeriche, servono {n_features}")
        feature_columns = all_numeric_cols
        
        # Aggiungi colonne dummy
        dummy_needed = n_features - len(feature_columns)
        for i in range(dummy_needed):
            feature_columns.append(f"_dummy_{i}")
        print(f"⚠️  Aggiunte {dummy_needed} colonne dummy")
    
    for i, col in enumerate(feature_columns):
        print(f"   {i:2}: {col}")
    
    # Crea array di input
    input_data = np.zeros((sequence_length, len(feature_columns)), dtype=np.float32)
    
    # Copia i dati reali
    real_features = [col for col in feature_columns if not col.startswith('_dummy')]
    if real_features:
        real_data = recent_data[real_features].values.astype(np.float32)
        input_data[:, :len(real_features)] = real_data
    
    print(f"\nShape input array: {input_data.shape}")
    print(f"Dtype: {input_data.dtype}")
    
    # Controlla valori
    print(f"\nStatistiche input array:")
    print(f"  Min:    {input_data.min():.6f}")
    print(f"  Max:    {input_data.max():.6f}")
    print(f"  Mean:   {input_data.mean():.6f}")
    print(f"  Std:    {input_data.std():.6f}")
    
    # Controlla NaN/Inf
    nan_count = np.isnan(input_data).sum()
    inf_count = np.isinf(input_data).sum()
    
    if nan_count > 0 or inf_count > 0:
        print(f"⚠️  Problemi nei dati: {nan_count} NaN, {inf_count} Inf")
        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Reshape finale per modello
    final_input = input_data.reshape(1, sequence_length, len(feature_columns))
    print(f"\n✅ Input finale shape: {final_input.shape}")
    
    return final_input, feature_columns

def test_prediction(model, input_data):
    """Prova la predizione e mostra risultati."""
    print("\n" + "=" * 60)
    print("TEST PREDIZIONE")
    print("=" * 60)
    
    if model is None:
        print("❌ Modello non caricato")
        return None
    
    if input_data is None:
        print("❌ Input data non valido")
        return None
    
    try:
        print(f"Input shape: {input_data.shape}")
        print(f"Input dtype: {input_data.dtype}")
        
        # Verifica che il modello possa accettare questo input
        expected_shape = model.input_shape
        print(f"Modello si aspetta: {expected_shape}")
        
        if len(expected_shape) == 3:
            expected_seq = expected_shape[1]
            expected_feats = expected_shape[2]
            actual_seq = input_data.shape[1]
            actual_feats = input_data.shape[2]
            
            if actual_seq != expected_seq:
                print(f"⚠️  Mismatch sequence length: {actual_seq} != {expected_seq}")
            
            if actual_feats != expected_feats:
                print(f"⚠️  Mismatch features: {actual_feats} != {expected_feats}")
        
        # Esegui predizione
        print("\nEseguendo predizione...")
        prediction = model.predict(input_data, verbose=0)
        
        print(f"\n✅ PREDIZIONE RIUSCITA!")
        print(f"Valore predetto: {prediction[0][0]:.6f}")
        print(f"Raw prediction: {prediction}")
        print(f"Output shape: {prediction.shape}")
        
        # Interpretazione
        pred_value = prediction[0][0]
        print(f"\n📊 INTERPRETAZIONE:")
        if pred_value > 0.05:
            print(f"  FORTE SEGNALE COMPRA: +{pred_value:.4%}")
        elif pred_value > 0.01:
            print(f"  SEGNALE COMPRA: +{pred_value:.4%}")
        elif pred_value < -0.05:
            print(f"  FORTE SEGNALE VENDITA: {pred_value:.4%}")
        elif pred_value < -0.01:
            print(f"  SEGNALE VENDITA: {pred_value:.4%}")
        else:
            print(f"  NEUTRO: {pred_value:.4%}")
        
        return prediction
        
    except Exception as e:
        print(f"\n❌ ERRORE durante la predizione: {str(e)}")
        print("Traceback completo:")
        traceback.print_exc()
        return None

def main():
    """Script principale di debug."""
    
    print("🔍 DEBUG COMPLETO SISTEMA DI PREDIZIONE")
    print("=" * 60)
    
    # File di test
    test_file = "test.csv"
    print(f"Test file: {test_file}")
    
    if not os.path.exists(test_file):
        print(f"❌ File non trovato: {test_file}")
        return
    
    # 1. Debug dati
    print("\n" + "=" * 60)
    print("FASE 1: ANALISI DATI")
    print("=" * 60)
    
    df = debug_data_loading(test_file)
    if df is None:
        print("❌ Impossibile continuare senza dati")
        return
    
    # 2. Trova ultimo modello
    print("\n" + "=" * 60)
    print("FASE 2: RICERCA MODELLO")
    print("=" * 60)
    
    import glob
    model_files = glob.glob("models/*.h5")
    if not model_files:
        print("❌ Nessun modello trovato in models/")
        return
    
    model_files.sort(key=os.path.getmtime, reverse=True)
    model_path = model_files[0]
    print(f"Modello selezionato: {os.path.basename(model_path)}")
    
    # 3. Carica modello
    print("\n" + "=" * 60)
    print("FASE 3: CARICAMENTO MODELLO")
    print("=" * 60)
    
    model, metadata = debug_model_loading(model_path)
    if model is None:
        print("❌ Impossibile caricare il modello")
        return
    
    # Estrai parametri
    seq_length = metadata.get('sequence_length', 20)
    n_feats = metadata.get('n_features', 9)
    
    print(f"\nParametri modello:")
    print(f"  Sequence length: {seq_length}")
    print(f"  Numero features: {n_feats}")
    
    # 4. Prepara input
    print("\n" + "=" * 60)
    print("FASE 4: PREPARAZIONE INPUT")
    print("=" * 60)
    
    input_data, feature_columns = debug_prepare_input(df, seq_length, n_feats)
    
    if input_data is None:
        print("❌ Impossibile preparare input")
        return
    
    # 5. Test predizione
    print("\n" + "=" * 60)
    print("FASE 5: TEST PREDIZIONE")
    print("=" * 60)
    
    prediction = test_prediction(model, input_data)
    
    # 6. Conclusioni
    print("\n" + "=" * 60)
    print("CONCLUSIONI")
    print("=" * 60)
    
    if prediction is not None:
        print("✅ DEBUG COMPLETATO CON SUCCESSO")
        print(f"   Il sistema funziona correttamente")
        print(f"   Valore predetto: {prediction[0][0]:.6f}")
    else:
        print("❌ DEBUG FALLITO")
        print("   Il sistema ha problemi di predizione")
        
        print("\n🔧 SUGGERIMENTI:")
        print("1. Verifica che il modello sia stato addestrato con le stesse features")
        print("2. Controlla che non ci siano valori NaN o Inf nei dati")
        print("3. Assicurati di avere abbastanza dati (almeno 20 righe)")
        print("4. Prova a normalizzare i dati di input")

if __name__ == "__main__":
    main()