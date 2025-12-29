"""
fix_models.py
Script per fixare modelli salvati con problemi di compatibilità
"""

import os
import joblib
import logging
from tensorflow import keras
from tensorflow.keras import metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_tensorflow_compatibility():
    """Fix per problemi di compatibilità TensorFlow"""
    keras.utils.get_custom_objects().update({
        "MeanSquaredError": metrics.MeanSquaredError,
        "MeanAbsoluteError": metrics.MeanAbsoluteError,
        "mse": metrics.MeanSquaredError,
        "mae": metrics.MeanAbsoluteError
    })
    
    original_load = keras.models.load_model
    def patched_load(filepath, **kwargs):
        try:
            return original_load(filepath, **kwargs)
        except Exception as e:
            if "not a KerasSaveable subclass" in str(e):
                logger.warning(f"Fixing: {filepath}")
                model = original_load(filepath, compile=False)
                model.compile(
                    optimizer="adam",
                    loss="mse",
                    metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
                )
                return model
            raise
    
    keras.models.load_model = patched_load
    return True

def fix_existing_models(model_dir="models"):
    """Converte i modelli .h5 esistenti in .keras"""
    fix_tensorflow_compatibility()
    
    if not os.path.exists(model_dir):
        logger.error(f"Cartella {model_dir} non trovata")
        return
    
    h5_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    for h5_file in h5_files:
        try:
            h5_path = os.path.join(model_dir, h5_file)
            keras_path = h5_path.replace('.h5', '.keras')
            
            logger.info(f"Convertendo {h5_file}...")
            
            # Carica il modello .h5
            model = keras.models.load_model(h5_path)
            
            # Salva in formato .keras
            model.save(keras_path, save_format='keras')
            
            logger.info(f"✅ Convertito: {h5_file} -> {os.path.basename(keras_path)}")
            
            # Opzionale: elimina il vecchio file .h5
            # os.remove(h5_path)
            
        except Exception as e:
            logger.error(f"Errore nella conversione di {h5_file}: {e}")

if __name__ == "__main__":
    logger.info("🔧 Fix modelli TensorFlow...")
    fix_existing_models()
    logger.info("✅ Done!")