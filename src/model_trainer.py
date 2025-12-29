"""
model_trainer.py
Modulo per l'addestramento del modello LSTM con compatibilità TensorFlow
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import metrics
import joblib
from datetime import datetime
import logging

# Applica il patch
def apply_tensorflow_patch():
    """Patch per problemi di compatibilità TensorFlow"""
    logging.info("🔧 Applicazione patch TensorFlow...")
    
    # Registra metriche
    tf.keras.utils.get_custom_objects().update({
        "MeanSquaredError": metrics.MeanSquaredError,
        "MeanAbsoluteError": metrics.MeanAbsoluteError,
        "mse": metrics.MeanSquaredError,
        "mae": metrics.MeanAbsoluteError
    })
    
    # Patch load_model
    original_load = tf.keras.models.load_model
    def patched_load(filepath, **kwargs):
        try:
            return original_load(filepath, **kwargs)
        except Exception as e:
            if "not a KerasSaveable subclass" in str(e):
                logging.warning(f"⚠️  Applicazione patch per: {filepath}")
                model = original_load(filepath, compile=False)
                model.compile(
                    optimizer="adam",
                    loss="mse",
                    metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
                )
                return model
            raise
    
    tf.keras.models.load_model = patched_load
    return True

# Applica patch all'avvio
apply_tensorflow_patch()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModelTrainer:
    """Classe per l'addestramento di modelli LSTM"""
    
    def __init__(self, model_dir="models"):
        """Inizializza il trainer"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Configurazione del modello
        self.sequence_length = 10
        self.n_features = None
        self.model = None
        self.scaler = None
        
    def build_model(self, input_shape):
        """Costruisce il modello LSTM"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True, dropout=0.2),
            LSTM(30, dropout=0.2),
            Dense(20, activation='relu'),
            Dropout(0.2),
            Dense(1)  # Output: prezzo predetto
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']  # Usa stringhe per compatibilità
        )
        
        return model
    
    def set_scaler(self, scaler):
        """Imposta lo scaler utilizzato per la normalizzazione"""
        self.scaler = scaler
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Addestra il modello"""
        logger.info(f"Addestramento modello con {len(X_train)} campioni")
        
        # Determina la forma di input
        self.sequence_length = X_train.shape[1]
        self.n_features = X_train.shape[2]
        
        # Costruisce il modello
        self.model = self.build_model((self.sequence_length, self.n_features))
        logger.info("Modello costruito con successo")
        self.model.summary(print_fn=logger.info)
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Usa estensione .keras per compatibilità
        model_path = os.path.join(self.model_dir, f"trading_model_{timestamp}.keras")
        checkpoint_path = os.path.join(self.model_dir, f"best_model_{timestamp}.keras")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                checkpoint_path, 
                monitor='val_loss', 
                save_best_only=True,
                save_format='keras'  # Specifica il formato
            )
        ]
        
        # Addestramento
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks if validation_data else [],
            verbose=1
        )
        
        # Salva il modello finale in formato Keras moderno
        self.model.save(model_path, save_format='keras')
        logger.info(f"Modello salvato in: {model_path}")
        
        # Se disponibile, carica il modello migliore
        if os.path.exists(checkpoint_path):
            try:
                self.model = keras.models.load_model(checkpoint_path)
                logger.info(f"Caricato modello migliore da: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Impossibile caricare il modello migliore: {e}")
                logger.info("Utilizzo dell'ultimo modello addestrato")
        
        # Salva lo scaler se disponibile
        if self.scaler is not None:
            scaler_path = os.path.join(self.model_dir, f"trading_model_{timestamp}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler salvato in: {scaler_path}")
        
        # Salva anche metadati del modello
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'timestamp': timestamp,
            'model_format': 'keras'
        }
        
        metadata_path = os.path.join(self.model_dir, f"trading_model_{timestamp}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        logger.info(f"Metadati salvati in: {metadata_path}")
        
        return history, model_path
    
    def evaluate(self, X_test, y_test):
        """Valuta il modello sui dati di test"""
        if self.model is None:
            raise ValueError("Modello non addestrato. Chiamare train() prima di evaluate()")
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Loss (MSE): {loss:.6f}")
        logger.info(f"Test MAE: {mae:.6f}")
        
        return loss, mae
    
    def predict(self, X):
        """Effettua previsioni"""
        if self.model is None:
            raise ValueError("Modello non addestrato. Chiamare train() prima di predict()")
        
        predictions = self.model.predict(X, verbose=0)
        
        if self.scaler is not None:
            # Ricostruisci array per inverse transform
            dummy_array = np.zeros((len(predictions), X.shape[2] + 1))
            dummy_array[:, -1] = predictions.flatten()
            predictions_original = self.scaler.inverse_transform(dummy_array)[:, -1]
            return predictions_original
        
        return predictions.flatten()
    
    def save_model(self, filepath=None):
        """Salva il modello e lo scaler"""
        if self.model is None:
            raise ValueError("Nessun modello da salvare")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.model_dir, f"trading_model_{timestamp}.keras")
        
        # Salva modello in formato Keras
        self.model.save(filepath, save_format='keras')
        logger.info(f"Modello salvato in: {filepath}")
        
        # Salva scaler se presente
        if self.scaler is not None:
            scaler_path = filepath.replace('.keras', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler salvato in: {scaler_path}")
        
        return filepath
    
    @classmethod
    def load_model(cls, model_path, scaler_path=None):
        """Carica un modello salvato"""
        logger.info(f"Caricamento modello da: {model_path}")
        
        # Crea istanza
        trainer = cls()
        
        # Carica modello usando il patched load_model
        try:
            trainer.model = keras.models.load_model(model_path)
            logger.info("Modello caricato con successo")
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {e}")
            raise
        
        # Carica scaler se disponibile
        if scaler_path and os.path.exists(scaler_path):
            try:
                trainer.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler caricato da: {scaler_path}")
            except Exception as e:
                logger.warning(f"Impossibile caricare lo scaler: {e}")
        
        return trainer