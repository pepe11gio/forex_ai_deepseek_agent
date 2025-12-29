"""
patch_tensorflow.py
Patch per risolvere bug serializzazione metriche TensorFlow.
Importa questo file PRIMA di qualsiasi operazione con TensorFlow.
"""

import tensorflow as tf
from tensorflow.keras import metrics
import sys

def apply_tensorflow_patch():
    """Applica patch a TensorFlow per risolvere bug serializzazione."""
    
    print("🔧 Applicando patch TensorFlow...")
    
    # 1. Registra metriche globalmente per compatibilità
    tf.keras.utils.get_custom_objects().update({
        "MeanSquaredError": metrics.MeanSquaredError,
        "MeanAbsoluteError": metrics.MeanAbsoluteError,
        # Alias per compatibilità con modelli vecchi
        "mse": metrics.MeanSquaredError,
        "mae": metrics.MeanAbsoluteError
    })
    
    # 2. Patch della funzione load_model per gestire errori
    original_load_model = tf.keras.models.load_model
    
    def patched_load_model(filepath, **kwargs):
        """
        Versione patchata di load_model che gestisce metriche corrotte.
        """
        try:
            return original_load_model(filepath, **kwargs)
        except Exception as e:
            error_msg = str(e)
            
            # Se è il bug "not a KerasSaveable subclass"
            if "not a KerasSaveable subclass" in error_msg or "Could not deserialize" in error_msg:
                print(f"⚠️  Rilevato bug serializzazione in: {filepath}")
                print("   Applico patch di emergenza...")
                
                # Carica senza compilare
                model = original_load_model(filepath, compile=False)
                
                # Ricompila con metriche corrette
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
                )
                
                print("✅ Modello riparato con successo")
                return model
            else:
                # Rilancia altri errori
                raise
    
    # Sostituisci la funzione globale
    tf.keras.models.load_model = patched_load_model
    
    print("✅ Patch TensorFlow applicata con successo")
    return True

# Applica patch automaticamente all'import
apply_tensorflow_patch()
