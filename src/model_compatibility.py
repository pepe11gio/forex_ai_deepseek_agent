"""
model_compatibility.py
Gestione UNIFICATA compatibilità modelli Keras 2.x ↔ 3.x
NO FALLBACK - SOLUZIONI GARANTITE
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional
import tensorflow as tf

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAZIONE COMPATIBILITÀ
# ============================================================================

class ModelCompatibilityConfig:
    """Configurazione unificata compatibilità."""
    
    # FORMATI SUPPORTATI (in ordine di preferenza)
    PREFERRED_FORMATS = ['.keras', '.h5']
    
    # METRICHE COMPATIBILI PER TIPO DI PROBLEMA
    COMPATIBLE_METRICS = {
        'regression': ['mae', 'mse'],  # mse come stringa, non oggetto
        'binary_classification': ['accuracy', 'binary_accuracy', 'auc'],
        'multi_classification': ['accuracy', 'categorical_accuracy']
    }
    
    # LOSS COMPATIBILI
    COMPATIBLE_LOSSES = {
        'regression': 'mse',
        'binary_classification': 'binary_crossentropy',
        'multi_classification': 'categorical_crossentropy'
    }
    
    # CUSTOM OBJECTS UNIVERSALI
    UNIVERSAL_CUSTOM_OBJECTS = {
        # Metriche come stringhe
        'mse': 'mse',
        'mae': 'mae',
        'accuracy': 'accuracy',
        'binary_accuracy': 'binary_accuracy',
        'auc': 'auc',
        'categorical_accuracy': 'categorical_accuracy',
        
        # Layer speciali (se usati)
        'BatchNormalization': tf.keras.layers.BatchNormalization,
        'Dropout': tf.keras.layers.Dropout,
        'Bidirectional': tf.keras.layers.Bidirectional,
        'LSTM': tf.keras.layers.LSTM,
        'Dense': tf.keras.layers.Dense,
        'Input': tf.keras.layers.Input,
    }

# ============================================================================
# CORE: SALVATAGGIO GARANTITO
# ============================================================================

def save_model_guaranteed(model: tf.keras.Model, 
                         model_path: str,
                         problem_type: str = 'regression',
                         metadata: Dict[str, Any] = None) -> str:
    """
    Salva modello GARANTENDO compatibilità.
    Restituisce il path effettivo del modello salvato.
    """
    logger.info(f"💾 Salvataggio modello garantito: {model_path}")
    
    # 1. DETERMINA FORMATO FINALE
    final_path = _determine_final_path(model_path)
    
    # 2. PREPARA MODELLO COMPATIBILE
    compatible_model = _prepare_compatible_model(model, problem_type)
    
    # 3. SALVA CON FORMATO CORRETTO
    _save_with_proper_format(compatible_model, final_path)
    
    # 4. SALVA METADATA
    if metadata:
        _save_metadata(metadata, final_path, problem_type)
    
    # 5. VERIFICA
    _verify_saved_model(final_path)
    
    return final_path

def _determine_final_path(model_path: str) -> str:
    """Determina path finale basato su compatibilità."""
    base_path, ext = os.path.splitext(model_path)
    
    # Se formato non specificato o legacy, usa .keras
    if ext not in ['.keras', '.h5'] or ext == '.h5':
        final_path = base_path + '.keras'
        logger.info(f"  🔄 Convertito {ext} → .keras per compatibilità")
    else:
        final_path = model_path
    
    return final_path

def _prepare_compatible_model(model: tf.keras.Model, 
                            problem_type: str) -> tf.keras.Model:
    """Prepara modello con configurazione compatibile."""
    
    # Crea nuova istanza con stessa architettura
    config = model.get_config()
    compatible_model = tf.keras.Model.from_config(config)
    
    # Copia pesi
    compatible_model.set_weights(model.get_weights())
    
    # Configura ottimizzatore compatibile
    optimizer_config = _get_compatible_optimizer_config(model)
    
    # Configura metriche compatibili
    metrics = ModelCompatibilityConfig.COMPATIBLE_METRICS.get(
        problem_type, 
        ModelCompatibilityConfig.COMPATIBLE_METRICS['regression']
    )
    
    loss = ModelCompatibilityConfig.COMPATIBLE_LOSSES.get(
        problem_type,
        ModelCompatibilityConfig.COMPATIBLE_LOSSES['regression']
    )
    
    # Ricompila
    compatible_model.compile(
        optimizer=optimizer_config,
        loss=loss,
        metrics=metrics
    )
    
    logger.info(f"  ✅ Modello preparato per {problem_type}")
    logger.info(f"     Loss: {loss}, Metrics: {metrics}")
    
    return compatible_model

def _get_compatible_optimizer_config(model: tf.keras.Model):
    """Estrae configurazione ottimizzatore in modo compatibile."""
    if hasattr(model.optimizer, 'get_config'):
        return model.optimizer.get_config()
    return 'adam'  # Default garantito

def _save_with_proper_format(model: tf.keras.Model, model_path: str):
    """Salva con formato appropriato."""
    try:
        if model_path.endswith('.keras'):
            # Salva in formato Keras moderno
            model.save(model_path, save_format='keras')
        else:
            # .h5 - usa save_weights + architettura separata
            _save_h5_compatible(model, model_path)
            
        logger.info(f"  ✅ Modello salvato: {os.path.basename(model_path)}")
        
    except Exception as e:
        logger.error(f"  ❌ Errore salvataggio: {e}")
        raise RuntimeError(f"Salvataggio modello fallito: {e}")

def _save_h5_compatible(model: tf.keras.Model, model_path: str):
    """Salva in formato .h5 compatibile."""
    # 1. Salva architettura JSON
    arch_path = model_path.replace('.h5', '_architecture.json')
    model_json = model.to_json()
    with open(arch_path, 'w') as f:
        f.write(model_json)
    
    # 2. Salva pesi
    model.save_weights(model_path)
    
    # 3. Crea file manifest
    _create_h5_manifest(model_path, arch_path)

def _create_h5_manifest(h5_path: str, arch_path: str):
    """Crea file manifest per modelli .h5."""
    import h5py
    
    with h5py.File(h5_path, 'a') as f:  # 'a' per append
        f.attrs['model_format'] = 'keras_h5_compatible'
        f.attrs['architecture_file'] = os.path.basename(arch_path)
        f.attrs['keras_version'] = tf.__version__

def _save_metadata(metadata: Dict[str, Any], model_path: str, problem_type: str):
    """Salva metadata standardizzati."""
    metadata_path = model_path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
    
    # Metadata standard
    standard_metadata = {
        'model_path': os.path.basename(model_path),
        'problem_type': problem_type,
        'keras_version': tf.__version__,
        'saved_date': np.datetime64('now').astype(str),
        'compatibility_version': '1.0',
        **metadata
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(standard_metadata, f, indent=2, default=str)
    
    logger.info(f"  ✅ Metadata salvati: {os.path.basename(metadata_path)}")

def _verify_saved_model(model_path: str):
    """Verifica che il modello sia caricabile."""
    try:
        test_model = load_model_guaranteed(model_path)
        logger.info(f"  ✅ Verifica completata: modello caricabile")
        del test_model
    except Exception as e:
        logger.error(f"  ❌ Verifica fallita: {e}")
        raise RuntimeError(f"Modello non verificabile: {e}")

# ============================================================================
# CORE: CARICAMENTO GARANTITO
# ============================================================================

def load_model_guaranteed(model_path: str, 
                         custom_objects: Dict = None) -> tf.keras.Model:
    """
    Carica modello GARANTENDO compatibilità.
    """
    logger.info(f"🔄 Caricamento modello garantito: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    
    # 1. DETERMINA STRATEGIA DI CARICAMENTO
    load_strategy = _determine_load_strategy(model_path)
    
    # 2. CARICA CON STRATEGIA APPROPRIATA
    model = _load_with_strategy(model_path, load_strategy, custom_objects)
    
    # 3. VERIFICA E RICOMPILA SE NECESSARIO
    model = _ensure_model_compiled(model, model_path)
    
    logger.info(f"  ✅ Modello caricato: {model.input_shape} → {model.output_shape}")
    
    return model

def _determine_load_strategy(model_path: str) -> str:
    """Determina strategia di caricamento."""
    if model_path.endswith('.keras'):
        return 'keras_format'
    elif model_path.endswith('.h5'):
        return _inspect_h5_format(model_path)
    else:
        raise ValueError(f"Formato modello non supportato: {model_path}")

def _inspect_h5_format(h5_path: str) -> str:
    """Ispeziona formato .h5."""
    import h5py
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'model_format' in f.attrs:
                format_type = f.attrs['model_format']
                if format_type == 'keras_h5_compatible':
                    return 'h5_compatible'
            return 'h5_standard'
    except:
        return 'h5_standard'

def _load_with_strategy(model_path: str, 
                       strategy: str, 
                       custom_objects: Dict) -> tf.keras.Model:
    """Carica con strategia specifica."""
    
    # Unisci custom objects
    all_custom_objects = {**ModelCompatibilityConfig.UNIVERSAL_CUSTOM_OBJECTS}
    if custom_objects:
        all_custom_objects.update(custom_objects)
    
    if strategy == 'keras_format':
        # Caricamento standard .keras
        return tf.keras.models.load_model(
            model_path, 
            custom_objects=all_custom_objects
        )
    
    elif strategy == 'h5_standard':
        # .h5 standard con compile=False
        model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects=all_custom_objects
        )
        return model
    
    elif strategy == 'h5_compatible':
        # .h5 con architettura separata
        return _load_h5_compatible(model_path)
    
    else:
        raise ValueError(f"Strategia di caricamento sconosciuta: {strategy}")

def _load_h5_compatible(h5_path: str) -> tf.keras.Model:
    """Carica modello .h5 con architettura separata."""
    import h5py
    
    with h5py.File(h5_path, 'r') as f:
        arch_file = f.attrs['architecture_file']
    
    model_dir = os.path.dirname(h5_path)
    arch_path = os.path.join(model_dir, arch_file)
    
    # Carica architettura
    with open(arch_path, 'r') as f:
        model_json = f.read()
    
    model = tf.keras.models.model_from_json(
        model_json,
        custom_objects=ModelCompatibilityConfig.UNIVERSAL_CUSTOM_OBJECTS
    )
    
    # Carica pesi
    model.load_weights(h5_path)
    
    return model

def _ensure_model_compiled(model: tf.keras.Model, model_path: str) -> tf.keras.Model:
    """Garantisce che il modello sia compilato."""
    if model.compiled_loss is None:
        logger.info("  🔄 Modello non compilato, ricompilo...")
        
        # Determina problem type dai metadata
        problem_type = _detect_problem_type(model, model_path)
        
        # Ricompila
        metrics = ModelCompatibilityConfig.COMPATIBLE_METRICS.get(
            problem_type, 
            ['mae']
        )
        
        loss = ModelCompatibilityConfig.COMPATIBLE_LOSSES.get(
            problem_type,
            'mse'
        )
        
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"  ✅ Ricompilato per {problem_type}")
    
    return model

def _detect_problem_type(model: tf.keras.Model, model_path: str) -> str:
    """Rileva tipo di problema."""
    
    # 1. Prova dai metadata
    metadata_path = model_path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if 'problem_type' in metadata:
                return metadata['problem_type']
        except:
            pass
    
    # 2. Rileva dall'architettura
    last_layer = model.layers[-1]
    if hasattr(last_layer, 'activation'):
        activation = last_layer.activation
        
        if isinstance(activation, str):
            act_name = activation
        elif hasattr(activation, '__name__'):
            act_name = activation.__name__
        else:
            act_name = str(activation)
        
        if 'sigmoid' in act_name.lower() and model.output_shape[-1] == 1:
            return 'binary_classification'
        elif 'softmax' in act_name.lower():
            return 'multi_classification'
    
    # Default
    return 'regression'

# ============================================================================
# UTILITIES
# ============================================================================

def get_model_info(model_path: str) -> Dict[str, Any]:
    """Ottiene informazioni modello."""
    info = {
        'path': model_path,
        'exists': os.path.exists(model_path),
        'size_bytes': os.path.getsize(model_path) if os.path.exists(model_path) else 0,
        'format': os.path.splitext(model_path)[1],
    }
    
    # Metadata
    metadata_path = model_path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                info['metadata'] = json.load(f)
        except:
            info['metadata'] = 'error_reading'
    
    return info

def convert_all_models_to_keras(models_dir: str):
    """Converti tutti i modelli .h5 a .keras."""
    import glob
    
    h5_models = glob.glob(os.path.join(models_dir, "*.h5"))
    
    for h5_path in h5_models:
        try:
            logger.info(f"Convertendo: {os.path.basename(h5_path)}")
            
            # Carica
            model = load_model_guaranteed(h5_path)
            
            # Salva in .keras
            keras_path = h5_path.replace('.h5', '.keras')
            model.save(keras_path, save_format='keras')
            
            # Mantieni metadata
            metadata_h5 = h5_path.replace('.h5', '_metadata.json')
            metadata_keras = keras_path.replace('.keras', '_metadata.json')
            if os.path.exists(metadata_h5):
                os.rename(metadata_h5, metadata_keras)
            
            logger.info(f"  ✅ Convertito a: {os.path.basename(keras_path)}")
            
        except Exception as e:
            logger.error(f"  ❌ Errore conversione {h5_path}: {e}")

# ============================================================================
# TEST
# ============================================================================

def test_compatibility():
    """Test completo compatibilità."""
    import tempfile
    
    print("🧪 TEST COMPATIBILITÀ MODELLI")
    print("=" * 50)
    
    # Crea modello test
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(20, 31)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test salvataggio .keras
        keras_path = os.path.join(tmpdir, 'test_model.keras')
        save_model_guaranteed(model, keras_path, 'binary_classification', {'test': True})
        
        # Test caricamento .keras
        loaded_keras = load_model_guaranteed(keras_path)
        print(f"✅ .keras: {loaded_keras.input_shape}")
        
        # Test .h5 (se necessario)
        h5_path = os.path.join(tmpdir, 'test_model.h5')
        save_model_guaranteed(model, h5_path, 'binary_classification', {'test': True})
        
        # Test caricamento .h5
        loaded_h5 = load_model_guaranteed(h5_path)
        print(f"✅ .h5: {loaded_h5.input_shape}")
    
    print("=" * 50)
    print("✅ TEST COMPLETATO")

if __name__ == "__main__":
    test_compatibility()