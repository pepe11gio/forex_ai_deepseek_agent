"""
predictor.py
Predizione con TP/SL - Versione semplificata ma completa
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Any, Union
import joblib
import logging
from datetime import datetime
import warnings
from timestamp_features import TimestampFeatureExtractor
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingPredictor:
    """Predictor per trading con TP/SL."""
    
    def __init__(self, model_path: str = None, 
                 scaler_path: str = None,
                 sequence_length: int = None,
                 feature_names: List[str] = None,
                 confidence_level: float = 0.95):
        self.model = None
        self.scaler = None
        self.sequence_length = sequence_length
        self.feature_names = feature_names
        self.confidence_level = confidence_level
        self.model_metadata = {}
        self.timestamp_extractor = TimestampFeatureExtractor()
        
        if model_path:
            self.load_model(model_path, scaler_path)
        
        logger.info(f"TradingPredictor inizializzato")
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """Carica modello e scaler."""
        try:
            logger.info(f"Caricamento modello: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # CRITICO: Carica SEMPRE lo scaler
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler caricato: {scaler_path}")
            else:
                logger.warning("⚠️  Scaler non trovato, cerco alternativa...")
                
                # Cerca scaler con stesso nome del modello
                import glob
                model_dir = os.path.dirname(model_path)
                model_base = os.path.basename(model_path).replace('.h5', '')
                
                # Cerca scaler con pattern simile
                scaler_pattern = os.path.join(model_dir, f"{model_base}*scaler*.pkl")
                scaler_files = glob.glob(scaler_pattern)
                
                if scaler_files:
                    scaler_path = scaler_files[0]
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"✅ Scaler alternativo trovato: {scaler_path}")
                else:
                    logger.error("❌ Nessuno scaler trovato!")
                    logger.warning("Predizioni potrebbero essere inaccurate")
            
            # Estrai metadati
            if len(self.model.input_shape) == 3:
                if self.sequence_length is None:
                    self.sequence_length = self.model.input_shape[1]
                self.n_features = self.model.input_shape[2]
            
            # Carica metadata
            metadata_path = model_path.replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                
                # Estrai feature names se disponibili
                if not self.feature_names and 'feature_names' in self.model_metadata:
                    self.feature_names = self.model_metadata['feature_names']
            
            logger.info("✅ Modello caricato con successo")
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento modello: {e}")
            raise
    
    def prepare_input_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Prepara dati per predizione INCLUSO timestamp features."""
        # Converti in DataFrame se necessario
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 3:
                return data
            df = pd.DataFrame(data)
        else:
            raise TypeError(f"Tipo dati non supportato: {type(data)}")
        
        # Verifica lunghezza
        if len(df) < self.sequence_length:
            raise ValueError(f"Servono {self.sequence_length} righe, ricevute: {len(df)}")
        
        # Prendi ultime N righe
        if len(df) > self.sequence_length:
            df = df.iloc[-self.sequence_length:]
        
        # 🔥 PASSO 1: Gestisci timestamp se presente
        timestamp_data = None
        if 'timestamp' in df.columns:
            timestamp_data = df['timestamp']
            
            # Converti timestamp in features numeriche
            try:
                # Inizializza extractor se non esiste
                if not hasattr(self, 'timestamp_extractor'):
                    from timestamp_features import TimestampFeatureExtractor
                    self.timestamp_extractor = TimestampFeatureExtractor()
                
                # Estrai features temporali
                time_features = self.timestamp_extractor.extract_features(timestamp_data)
                
                # Combina con dati numerici
                numeric_data = df.select_dtypes(include=[np.number])
                
                # Rimuovi timestamp dalla parte numerica se presente
                if 'timestamp' in numeric_data.columns:
                    numeric_data = numeric_data.drop(columns=['timestamp'])
                
                # Unisci features numeriche e temporali
                df_combined = pd.concat([numeric_data, time_features], axis=1)
                
                logger.info(f"✅ Aggiunte {len(time_features.columns)} features temporali")
                logger.info(f"Features temporali: {time_features.columns.tolist()}")
                
            except Exception as e:
                logger.warning(f"Errore estrazione features temporali: {e}, uso solo dati numerici")
                df_combined = df.select_dtypes(include=[np.number])
                if 'timestamp' in df_combined.columns:
                    df_combined = df_combined.drop(columns=['timestamp'])
        else:
            # Solo dati numerici
            df_combined = df.select_dtypes(include=[np.number])
        
        # 🔥 PASSO 2: Verifica compatibilità con scaler
        if self.scaler is not None:
            try:
                # Controlla se lo scaler ha il numero corretto di features
                expected_features = getattr(self.scaler, 'n_features_in_', len(df_combined.columns))
                
                if len(df_combined.columns) != expected_features:
                    logger.warning(f"Mismatch features: dati={len(df_combined.columns)}, scaler={expected_features}")
                    
                    # Se abbiamo più features (timestamp aggiunte), aggiusta
                    if len(df_combined.columns) > expected_features:
                        # Mantieni solo prime expected_features colonne
                        df_combined = df_combined.iloc[:, :expected_features]
                        logger.info(f"Ridotte features a {expected_features} per compatibilità scaler")
                
                # Normalizza
                normalized_data = self.scaler.transform(df_combined.values)
                
            except Exception as e:
                logger.error(f"Errore normalizzazione: {e}")
                raise ValueError(f"Impossibile normalizzare dati: {e}")
        else:
            logger.warning("⚠️  Nessuno scaler disponibile, uso dati grezzi")
            normalized_data = df_combined.values
        
        # 🔥 PASSO 3: Shape per LSTM
        # [batch_size, sequence_length, n_features]
        input_array = normalized_data.reshape(1, self.sequence_length, -1)
        
        logger.info(f"✅ Input shape finale: {input_array.shape}")
        
        return input_array
    
    def predict_single(self, data: Union[pd.DataFrame, np.ndarray],
                      return_confidence: bool = False) -> Dict[str, Any]:
        """Effettua singola predizione."""
        # Prepara dati
        input_data = self.prepare_input_data(data)
        
        # Predizione
        prediction_raw = self.model.predict(input_data, verbose=0)
        prediction_value = float(prediction_raw.flatten()[0])
        
        # De-normalizza se possibile
        if self.scaler is not None:
            try:
                dummy_array = np.zeros((1, self.n_features))
                dummy_array[0, 0] = prediction_value
                denorm_array = self.scaler.inverse_transform(dummy_array)
                prediction_denorm = float(denorm_array[0, 0])
            except:
                prediction_denorm = prediction_value
        else:
            prediction_denorm = prediction_value
        
        # Risultato base
        result = {
            'prediction': prediction_denorm,
            'prediction_raw': prediction_value,
            'timestamp': datetime.now().isoformat(),
            'sequence_length_used': self.sequence_length
        }
        
        # Confidence interval
        if return_confidence:
            ci_result = self.calculate_confidence_interval(input_data)
            result.update(ci_result)
        
        # Segnale di trading
        signal_result = self.generate_trading_signal(prediction_denorm)
        result.update(signal_result)
        
        logger.info(f"✅ Predizione: {prediction_denorm:.6f}")
        
        return result
    
    def calculate_confidence_interval(self, input_data: np.ndarray,
                                     n_bootstrap: int = 50) -> Dict[str, Any]:
        """Calcola intervallo di confidenza."""
        try:
            bootstrap_predictions = []
            
            for _ in range(n_bootstrap):
                noise = np.random.normal(0, 0.01, size=input_data.shape)
                noisy_input = input_data + noise
                pred = self.model.predict(noisy_input, verbose=0).flatten()[0]
                
                # De-normalizza
                if self.scaler is not None:
                    try:
                        dummy_array = np.zeros((1, self.n_features))
                        dummy_array[0, 0] = pred
                        denorm_array = self.scaler.inverse_transform(dummy_array)
                        pred_denorm = float(denorm_array[0, 0])
                    except:
                        pred_denorm = pred
                else:
                    pred_denorm = pred
                
                bootstrap_predictions.append(pred_denorm)
            
            # Calcola statistiche
            bootstrap_array = np.array(bootstrap_predictions)
            mean_pred = np.mean(bootstrap_array)
            std_pred = np.std(bootstrap_array)
            
            # Intervallo di confidenza
            z_score = 1.96  # 95% CI
            margin_of_error = z_score * std_pred
            ci_lower = mean_pred - margin_of_error
            ci_upper = mean_pred + margin_of_error
            
            return {
                'confidence_interval': {
                    'lower': float(ci_lower),
                    'upper': float(ci_upper),
                    'mean': float(mean_pred),
                    'std': float(std_pred)
                },
                'confidence_level': self.confidence_level
            }
            
        except Exception as e:
            logger.warning(f"Impossibile calcolare CI: {e}")
            return {
                'confidence_interval': None
            }
    
    def generate_trading_signal(self, prediction: float, 
                               timestamp: pd.Series = None) -> Dict[str, Any]:
        """Genera segnale con adattamento temporale."""
        
        # Calcola TP/SL base
        tp_sl_result = self.calculate_tp_sl(prediction)
        
        # 🔥 ADATTA in base all'ora se timestamp disponibile
        if timestamp is not None and len(timestamp) > 0:
            last_timestamp = timestamp.iloc[-1]
            time_scaling = self.timestamp_extractor.get_time_based_scaling(
                pd.Series([last_timestamp])
            )
            
            # Modifica TP/SL in base alla volatilità attesa
            volatility_factor = time_scaling['volatility_factor']
            session = time_scaling['session']
            
            # Aggiusta pips in base alla sessione
            tp_sl_result['tp_pips'] = int(tp_sl_result['tp_pips'] * volatility_factor)
            tp_sl_result['sl_pips'] = int(tp_sl_result['sl_pips'] * volatility_factor)
            
            # Aggiorna prezzi
            current_price = tp_sl_result['entry_price']
            pip_value = 0.0001
            
            if tp_sl_result['operation'] == "BUY":
                tp_sl_result['take_profit'] = current_price + (tp_sl_result['tp_pips'] * pip_value)
                tp_sl_result['stop_loss'] = current_price - (tp_sl_result['sl_pips'] * pip_value)
            else:
                tp_sl_result['take_profit'] = current_price - (tp_sl_result['tp_pips'] * pip_value)
                tp_sl_result['stop_loss'] = current_price + (tp_sl_result['sl_pips'] * pip_value)
            
            # Aggiungi info sessione
            tp_sl_result['trading_session'] = session
            tp_sl_result['volatility_factor'] = volatility_factor
        
        # Determina segnale
        percent_change = tp_sl_result['predicted_move_pct']
        
        if percent_change > 2.0:
            signal = 'BUY'
            strength = min(1.0, percent_change / 5.0)
        elif percent_change < -2.0:
            signal = 'SELL'
            strength = min(1.0, abs(percent_change) / 5.0)
        else:
            signal = 'NEUTRAL'
            strength = 0.0
        
        # Raccomandazioni
        recommendations = self._generate_recommendations(signal, strength, tp_sl_result)
        
        return {
            'trading_signal': signal,
            'signal_strength': float(strength),
            'signal_confidence': tp_sl_result['confidence'],
            'predicted_change_percent': float(percent_change),
            'operation_details': tp_sl_result,
            'recommendations': recommendations
        }
    
    def calculate_tp_sl(self, prediction: float) -> Dict[str, Any]:
        """Calcola Take Profit e Stop Loss con R/R = 4:1."""
        # Per forex, assumiamo che prediction sia in pips
        predicted_pips = prediction * 50  # Scala
        
        # Limita range
        predicted_pips = max(min(predicted_pips, 100), -100)
        
        # Determina operazione
        if predicted_pips > 0:
            operation = "BUY"
            tp_pips = 80    # 4R = 80 pips (80/20 = 4:1)
            sl_pips = 20    # 1R = 20 pips
            risk_reward = 4.0
        else:
            operation = "SELL"
            tp_pips = 80    # 4R = 80 pips
            sl_pips = 20    # 1R = 20 pips  
            risk_reward = 4.0
        
        # Prezzo corrente di esempio (dovrebbe essere estratto dai dati)
        current_price = 1.10000
        
        # Calcola prezzi
        pip_value = 0.0001
        
        if operation == "BUY":
            tp_price = current_price + (tp_pips * pip_value)
            sl_price = current_price - (sl_pips * pip_value)
        else:
            tp_price = current_price - (tp_pips * pip_value)
            sl_price = current_price + (sl_pips * pip_value)
        
        # Round per forex
        decimals = 5
        tp_price = round(tp_price, decimals)
        sl_price = round(sl_price, decimals)
        current_price = round(current_price, decimals)
        
        # Position sizing con 4% rischio
        position_size = self._calculate_position_size(current_price, sl_pips, risk_percent=4.0)
        
        return {
            'operation': operation,
            'entry_price': current_price,
            'take_profit': tp_price,
            'stop_loss': sl_price,
            'tp_pips': tp_pips,
            'sl_pips': sl_pips,
            'risk_reward_ratio': risk_reward,
            'predicted_move_pct': round(predicted_pips * pip_value / current_price * 100, 4),
            'position_size': position_size,
            'confidence': min(0.95, 0.5 + abs(prediction) / 2)
        }
    
    def _calculate_position_size(self, entry_price: float, sl_pips: int) -> Dict[str, Any]:
        """Calcola dimensionamento posizione."""
        RISK_PER_TRADE = 0.02  # 2%
        ACCOUNT_BALANCE = 10000.0
        LOT_SIZE = 100000
        
        risk_amount = ACCOUNT_BALANCE * RISK_PER_TRADE
        
        if sl_pips > 0:
            risk_per_pip = risk_amount / sl_pips
            pip_per_lot = 10.0  # Per EUR/USD
            lots = risk_per_pip / pip_per_lot
        else:
            lots = 0.01
        
        # Limita
        lots = max(0.01, min(lots, 10.0))
        
        position_value = lots * LOT_SIZE * entry_price
        
        return {
            'lots': round(lots, 2),
            'position_value_usd': round(position_value, 2),
            'risk_amount_usd': round(risk_amount, 2),
            'risk_percent': RISK_PER_TRADE * 100
        }
    
    def _generate_recommendations(self, signal: str, strength: float, 
                                 tp_sl_result: Dict) -> List[str]:
        """Genera raccomandazioni."""
        recommendations = []
        
        op = tp_sl_result['operation']
        tp = tp_sl_result['take_profit']
        sl = tp_sl_result['stop_loss']
        rr = tp_sl_result['risk_reward_ratio']
        
        recommendations.append(f"{op} a {tp_sl_result['entry_price']}")
        recommendations.append(f"TP: {tp} ({tp_sl_result['tp_pips']} pips)")
        recommendations.append(f"SL: {sl} ({tp_sl_result['sl_pips']} pips)")
        recommendations.append(f"R/R: {rr}:1")
        
        if rr < 1:
            recommendations.append("⚠️  R/R basso, considera setup migliore")
        elif rr > 2:
            recommendations.append("✅ Ottimo R/R ratio")
        
        if strength > 0.7:
            recommendations.append("🎯 Segnale forte - size normale")
        elif strength > 0.4:
            recommendations.append("📊 Segnale moderato - size ridotta")
        else:
            recommendations.append("⚡ Segnale debole - size piccola")
        
        recommendations.append("📌 Imposta TP/SL subito dopo ingresso")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Restituisce info modello."""
        if self.model is None:
            return {'error': 'Modello non caricato'}
        
        info = {
            'model_loaded': True,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'scaler_loaded': self.scaler is not None,
            'feature_names': self.feature_names
        }
        
        info.update(self.model_metadata)
        return info

def create_predictor_from_training(model_dir: str = 'models') -> TradingPredictor:
    """
    Crea predictor dall'ultimo modello addestrato.
    """
    import glob
    
    # Cerca file modello
    model_files = glob.glob(os.path.join(model_dir, "*.h5"))
    
    if not model_files:
        raise FileNotFoundError(f"Nessun modello trovato in {model_dir}")
    
    # Ordina per data (più recente primo)
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model = model_files[0]
    
    # Cerca scaler
    model_base = os.path.basename(latest_model).replace('.h5', '')
    scaler_pattern = os.path.join(model_dir, f"{model_base}*scaler*.pkl")
    scaler_files = glob.glob(scaler_pattern)
    
    if scaler_files:
        scaler_file = scaler_files[0]
    else:
        scaler_file = None
        logger.warning(f"Nessuno scaler trovato per {model_base}")
    
    # Crea predictor
    predictor = TradingPredictor(
        model_path=latest_model,
        scaler_path=scaler_file
    )
    
    logger.info(f"✅ Predictor creato da: {os.path.basename(latest_model)}")
    
    return predictor

if __name__ == "__main__":
    # Test
    print("Test predictor.py")
    try:
        predictor = create_predictor_from_training("models")
        print(f"✅ Predictor creato")
        
        # Dati di test
        test_data = pd.DataFrame({
            'price': np.random.randn(20).cumsum() + 1.1,
            'BB_UPPER': np.random.randn(20) + 1.101,
            'BB_MIDDLE': np.random.randn(20) + 1.1,
            'BB_LOWER': np.random.randn(20) + 1.099,
            'MACD_MAIN': np.random.randn(20),
            'MACD_SIGNAL': np.random.randn(20) * 0.8,
            'MOMENTUM': np.random.randn(20) * 100,
            'RSI': np.random.uniform(30, 70, 20),
            'BULLPOWER': np.random.randn(20),
            'BEARPOWER': np.random.randn(20)
        })
        
        result = predictor.predict_single(test_data, return_confidence=True)
        print(f"Predizione: {result['prediction']:.6f}")
        print(f"Segnale: {result['trading_signal']}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")