#!/usr/bin/env python3
"""
Script per fixare problemi di compatibilità Keras.
Esegui questo script prima di usare il sistema.
"""

import os
import sys
import json
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_keras_models():
    """Corregge i modelli Keras per compatibilità."""
    
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("❌ Cartella 'models' non trovata")
        return False
    
    fixed_count = 0
    
    for model_file in models_dir.glob("*.h5"):
        try:
            logger.info(f"🔧 Verificando: {model_file.name}")
            
            # Verifica metadata
            metadata_file = model_file.with_suffix('.json')
            if metadata_file.name.endswith('.h5.json'):
                metadata_file = Path(str(metadata_file).replace('.h5.json', '_metadata.json'))
            
            if not metadata_file.exists():
                # Cerca qualsiasi file metadata
                for meta_file in models_dir.glob(f"{model_file.stem}*metadata*.json"):
                    metadata_file = meta_file
                    break
            
            # Carica e fixa metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Aggiorna formato a .keras se possibile
                if metadata.get('model_file', '').endswith('.h5'):
                    metadata['model_file'] = metadata['model_file'].replace('.h5', '.keras')
                
                # Assicura che ci sia keras_version
                if 'keras_version' not in metadata:
                    import tensorflow as tf
                    metadata['keras_version'] = tf.__version__
                
                # Salva metadata aggiornati
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"  ✅ Metadata aggiornati")
                fixed_count += 1
            
            # Verifica scaler
            scaler_patterns = [
                f"{model_file.stem}_scaler.pkl",
                f"{model_file.stem}*scaler*.pkl"
            ]
            
            scaler_found = False
            for pattern in scaler_patterns:
                scaler_files = list(models_dir.glob(pattern))
                if scaler_files:
                    scaler_file = scaler_files[0]
                    
                    # Verifica che lo scaler sia caricabile
                    try:
                        scaler = joblib.load(scaler_file)
                        if hasattr(scaler, 'n_features_in_'):
                            logger.info(f"  ✅ Scaler OK: {scaler.n_features_in_} features")
                        else:
                            logger.warning(f"  ⚠️  Scaler senza n_features_in_")
                    except Exception as e:
                        logger.error(f"  ❌ Errore scaler: {e}")
                    
                    scaler_found = True
                    break
            
            if not scaler_found:
                logger.warning(f"  ⚠️  Scaler non trovato per {model_file.name}")
        
        except Exception as e:
            logger.error(f"❌ Errore processando {model_file.name}: {e}")
    
    # Cerca anche file .keras
    for model_file in models_dir.glob("*.keras"):
        logger.info(f"✅ Modello moderno: {model_file.name}")
        fixed_count += 1
    
    logger.info(f"\n📊 RIEPILOGO:")
    logger.info(f"   Modelli verificati: {fixed_count}")
    
    return fixed_count > 0

def create_minimal_test_data():
    """Crea dati di test minimi se la cartella data/ è vuota."""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    test_file = data_dir / "EURUSD_H4_test.csv"
    
    if test_file.exists():
        logger.info(f"✅ File di test esistente: {test_file}")
        return True
    
    # Crea dati di test minimali
    import pandas as pd
    import numpy as np
    
    logger.info("📝 Creazione dati di test...")
    
    # Crea 100 righe di dati forex realistici
    np.random.seed(42)
    n_samples = 100
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='4H')
    
    # Prezzo base EUR/USD
    base_price = 1.1000
    prices = base_price + np.cumsum(np.random.randn(n_samples) * 0.001)
    
    # Indicatori tecnici
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'BB_UPPER': prices + np.random.uniform(0.001, 0.003, n_samples),
        'BB_MIDDLE': prices + np.random.uniform(-0.0005, 0.0005, n_samples),
        'BB_LOWER': prices - np.random.uniform(0.001, 0.003, n_samples),
        'MACD_MAIN': np.random.randn(n_samples) * 0.001,
        'MACD_SIGNAL': np.random.randn(n_samples) * 0.0008,
        'MOMENTUM': np.random.uniform(-100, 100, n_samples),
        'RSI': np.random.uniform(30, 70, n_samples),
        'BULLPOWER': np.random.randn(n_samples) * 0.002,
        'BEARPOWER': np.random.randn(n_samples) * 0.002
    })
    
    # Salva
    df.to_csv(test_file, index=False)
    logger.info(f"✅ Dati di test creati: {test_file}")
    logger.info(f"   Righe: {len(df)}, Colonne: {len(df.columns)}")
    
    return True

def main():
    """Script principale."""
    print("=" * 60)
    print("🔧 SCRIPT DI FIX COMPATIBILITÀ")
    print("=" * 60)
    
    # 1. Crea dati di test se necessario
    create_minimal_test_data()
    
    # 2. Fixa modelli Keras
    fix_keras_models()
    
    # 3. Verifica struttura directory
    directories = ['models', 'data', 'analysis', 'logs', 'orders']
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"📁 {dir_name}: {'✅ ESISTE' if dir_path.exists() else '❌ MANCANTE'}")
    
    print("\n" + "=" * 60)
    print("✅ SCRIPT COMPLETATO")
    print("\nOra puoi eseguire:")
    print("   python run_corrected.py")
    print("\nPer installare dipendenze:")
    print("   pip install -r requirements.txt")
    
    return True

if __name__ == "__main__":
    main()