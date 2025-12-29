"""
predictor.py
Modulo per le previsioni del modello
"""

import os
import numpy as np
import joblib
import logging
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import metrics

# Patch per compatibilità (come nel model_trainer)
def apply_tensorflow_patch():
    """Patch per problemi di compatibilità TensorFlow"""
    logging.info("🔧 Applicazione patch TensorFlow in predictor...")
    
    tf = keras.backend
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
                logging.warning(f"⚠️  Patch predictor per: {filepath}")
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

# Applica patch all'import
apply_tensorflow_patch()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictor:
    """Classe per effettuare previsioni di prezzo"""
    
    def __init__(self, model_dir="models"):
        """Inizializza il predictor"""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.sequence_length = None
        self.n_features = None
        
    def load_latest_model(self):
        """Carica l'ultimo modello disponibile"""
        logger.info("Ricerca dell'ultimo modello...")
        
        try:
            # Trova tutti i file .keras o .h5 nella cartella models
            model_files = []
            if os.path.exists(self.model_dir):
                for file in os.listdir(self.model_dir):
                    if file.endswith(('.keras', '.h5')) and 'trading_model_' in file:
                        model_files.append(file)
            
            if not model_files:
                logger.warning("Nessun modello trovato")
                return False
            
            # Ordina per timestamp (dal più recente)
            model_files.sort(reverse=True)
            latest_model = model_files[0]
            model_path = os.path.join(self.model_dir, latest_model)
            
            # Cerca scaler corrispondente
            model_name = latest_model.split('.')[0]
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
            
            # Cerca metadati
            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.pkl")
            
            # Carica metadati se disponibili
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.sequence_length = metadata.get('sequence_length', 10)
                self.n_features = metadata.get('n_features', 5)
                logger.info(f"Metadati caricati: seq_len={self.sequence_length}, n_features={self.n_features}")
            
            # Carica modello
            logger.info(f"Caricamento modello da: {model_path}")
            
            try:
                self.model = keras.models.load_model(model_path)
                logger.info("Modello caricato con successo")
            except Exception as e:
                logger.error(f"Errore nel caricamento del modello: {e}")
                # Prova con compile=False
                try:
                    self.model = keras.models.load_model(model_path, compile=False)
                    self.model.compile(
                        optimizer="adam",
                        loss="mse",
                        metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
                    )
                    logger.info("Modello caricato con compile=False e riconfigurato")
                except Exception as e2:
                    logger.error(f"Errore anche con compile=False: {e2}")
                    return False
            
            # Carica scaler se disponibile
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"Scaler caricato da: {scaler_path}")
                except Exception as e:
                    logger.warning(f"Impossibile caricare lo scaler: {e}")
                    self.scaler = None
            else:
                logger.warning(f"Scaler non trovato: {scaler_path}")
                self.scaler = None
            
            logger.info("✅ Modello e scaler caricati con successo")
            return True
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {e}")
            return False
    
    def prepare_sequence(self, data):
        """Prepara una sequenza per la predizione"""
        if self.sequence_length is None:
            self.sequence_length = 10  # Default
        
        if len(data) < self.sequence_length:
            logger.warning(f"Dati insufficienti: {len(data)} < {self.sequence_length}")
            return None
        
        # Prendi gli ultimi N tick
        sequence = data[-self.sequence_length:]
        
        # Converti in array numpy
        sequence_array = np.array(sequence)
        
        # Normalizza se scaler è disponibile
        if self.scaler is not None:
            sequence_array = self.scaler.transform(sequence_array)
        
        # Riformatta per LSTM: (1, sequence_length, n_features)
        sequence_array = sequence_array.reshape(1, self.sequence_length, -1)
        
        return sequence_array
    
    def predict_next_price(self, recent_ticks):
        """
        Predice il prossimo prezzo basandosi sugli ultimi tick
        
        Args:
            recent_ticks: Lista di dizionari o array con features
        
        Returns:
            dict: Previsione e metadati
        """
        if self.model is None:
            logger.error("Modello non caricato")
            return None
        
        # Prepara i dati
        sequence = self.prepare_sequence(recent_ticks)
        if sequence is None:
            return None
        
        try:
            # Fai la predizione
            prediction_normalized = self.model.predict(sequence, verbose=0)
            
            # Se abbiamo lo scaler, denormalizza
            if self.scaler is not None:
                # Crea un array dummy per inverse transform
                dummy = np.zeros((1, sequence.shape[2] + 1))
                dummy[:, -1] = prediction_normalized.flatten()
                prediction = self.scaler.inverse_transform(dummy)[0, -1]
            else:
                prediction = prediction_normalized.flatten()[0]
            
            # Calcola statistiche
            current_price = recent_ticks[-1][0] if isinstance(recent_ticks[-1], (list, np.ndarray)) else recent_ticks[-1]['price']
            price_diff = prediction - current_price
            price_change_percent = (price_diff / current_price) * 100
            
            result = {
                'predicted_price': float(prediction),
                'current_price': float(current_price),
                'price_difference': float(price_diff),
                'price_change_percent': float(price_change_percent),
                'timestamp': datetime.now().isoformat(),
                'confidence': self._calculate_confidence(sequence),
                'sequence_length': self.sequence_length
            }
            
            logger.info(f"Predizione: {prediction:.4f} (Δ: {price_change_percent:+.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Errore durante la predizione: {e}")
            return None
    
    def _calculate_confidence(self, sequence):
        """Calcola una misura di confidenza per la predizione"""
        # Implementazione semplice basata sulla varianza degli ultimi prezzi
        try:
            if sequence.shape[2] > 0:  # Se abbiamo la colonna price
                prices = sequence[0, :, 0]
                price_std = np.std(prices)
                confidence = max(0, 1 - price_std * 10)  # Normalizza
                return float(min(confidence, 1.0))
        except:
            pass
        return 0.5  # Default
    
    def predict_sequence(self, data, steps=5):
        """
        Predice una sequenza di prezzi futuri
        
        Args:
            data: Dati storici
            steps: Numero di passi da predire
        
        Returns:
            list: Sequenza di previsioni
        """
        predictions = []
        current_data = data.copy()
        
        for i in range(steps):
            prediction = self.predict_next_price(current_data)
            if prediction is None:
                break
            
            predictions.append(prediction)
            
            # Aggiungi la predizione ai dati (usando l'ultimo tick come base)
            last_tick = current_data[-1].copy() if hasattr(current_data[-1], 'copy') else current_data[-1]
            
            # Aggiorna il prezzo con la predizione
            if isinstance(last_tick, dict):
                new_tick = last_tick.copy()
                new_tick['price'] = prediction['predicted_price']
                # Aggiorna timestamp (simulato)
                new_tick['timestamp'] = datetime.now().isoformat()
                current_data.append(new_tick)
            elif isinstance(last_tick, (list, np.ndarray)):
                new_tick = last_tick.copy()
                new_tick[0] = prediction['predicted_price']  # Assumendo che price sia il primo elemento
                current_data.append(new_tick)
            else:
                # Se non possiamo aggiornare, aggiungiamo la predizione come nuovo dato
                current_data.append([prediction['predicted_price']] * len(last_tick))
            
            # Mantieni solo gli ultimi N tick
            if len(current_data) > self.sequence_length:
                current_data = current_data[-self.sequence_length:]
        
        return predictions
    
    def get_model_info(self):
        """Restituisce informazioni sul modello caricato"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "status": "Model loaded",
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "scaler_available": self.scaler is not None,
            "model_layers": len(self.model.layers),
            "model_metrics": [m.name for m in self.model.metrics] if hasattr(self.model, 'metrics') else []
        }
        
        return info