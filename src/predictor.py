"""
predictor.py
Modulo per effettuare previsioni con il modello di trading addestrato.
Gestisce predizioni in tempo reale, calcolo di confidence interval e generazione di segnali.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingPredictor:
    """
    Classe per effettuare previsioni con il modello di trading.
    Gestisce predizioni singole, batch e in tempo reale.
    """
    
    def __init__(self, model_path: str = None, 
                 scaler_path: str = None,
                 sequence_length: int = None,
                 feature_names: List[str] = None,
                 confidence_level: float = 0.95):
        """
        Inizializza il predictor.
        
        Args:
            model_path: Percorso del modello salvato
            scaler_path: Percorso dello scaler salvato
            sequence_length: Lunghezza sequenze (se None, inferita dal modello)
            feature_names: Nomi delle features
            confidence_level: Livello di confidenza per intervalli (default: 0.95)
        """
        self.model = None
        self.scaler = None
        self.sequence_length = sequence_length
        self.feature_names = feature_names
        self.confidence_level = confidence_level
        self.model_metadata = {}
        
        # Carica modello e scaler se forniti
        if model_path:
            self.load_model(model_path, scaler_path)
        
        # Cache per performance
        self.prediction_cache = {}
        self.cache_max_size = 1000
        
        logger.info(f"TradingPredictor inizializzato (confidence: {confidence_level})")
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """
        Carica modello e scaler da file.
        
        Args:
            model_path: Percorso del modello
            scaler_path: Percorso dello scaler
        """
        try:
            logger.info(f"Caricamento modello da: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            if scaler_path:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler caricato da: {scaler_path}")
            
            # Estrai metadati dal modello
            if len(self.model.input_shape) == 3:
                if self.sequence_length is None:
                    self.sequence_length = self.model.input_shape[1]
                self.n_features = self.model.input_shape[2]
                
                logger.info(f"Modello caricato: sequenze di {self.sequence_length} tick")
                logger.info(f"Numero features: {self.n_features}")
            
            # Carica metadata se disponibile
            metadata_path = model_path.replace('.h5', '_metadata.json')
            try:
                import json
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Metadata caricati da: {metadata_path}")
            except:
                logger.warning("Metadata non trovati o non leggibili")
            
            logger.info("Modello caricato con successo!")
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {str(e)}")
            raise
    
    def prepare_input_data(self, data: Union[pd.DataFrame, np.ndarray, List],
                          is_normalized: bool = False) -> np.ndarray:
        """
        Prepara i dati di input per la predizione.
        
        Args:
            data: Dati in vari formati (DataFrame, array, lista)
            is_normalized: Se True, i dati sono già normalizzati
            
        Returns:
            Array numpy pronto per la predizione
        """
        logger.debug("Preparazione dati input per predizione...")
        
        # Converti in DataFrame se necessario
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            # Se è già nella forma corretta
            if len(data.shape) == 3:
                # Controlla dimensioni
                if data.shape[1] != self.sequence_length:
                    raise ValueError(
                        f"Input deve avere {self.sequence_length} timestep. "
                        f"Ricevuto: {data.shape[1]}"
                    )
                return data
            
            # Altrimenti, crea DataFrame
            if self.feature_names and len(data.shape) == 2:
                df = pd.DataFrame(data, columns=self.feature_names[:data.shape[1]])
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, list):
            data_array = np.array(data)
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(1, -1)
            df = pd.DataFrame(data_array)
        else:
            raise TypeError(f"Tipo dati non supportato: {type(data)}")
        
        # Verifica che abbiamo abbastanza dati
        if len(df) < self.sequence_length:
            raise ValueError(
                f"Sono necessari almeno {self.sequence_length} tick per una predizione. "
                f"Ricevuti: {len(df)}"
            )
        
        # Prendi gli ultimi N tick
        if len(df) > self.sequence_length:
            logger.debug(f"Prendendo ultimi {self.sequence_length} tick da {len(df)} disponibili")
            df = df.iloc[-self.sequence_length:]
        
        # Normalizza se necessario
        if not is_normalized and self.scaler is not None:
            logger.debug("Normalizzazione dati...")
            
            # Seleziona colonne features
            if self.feature_names is not None:
                # Filtra colonne disponibili
                available_features = [f for f in self.feature_names if f in df.columns]
                if not available_features:
                    available_features = df.columns.tolist()
                
                # Assicurati che l'ordine sia corretto
                df_selected = df[available_features].copy()
            else:
                df_selected = df.copy()
            
            # Normalizza
            df_normalized = pd.DataFrame(
                self.scaler.transform(df_selected),
                columns=df_selected.columns,
                index=df_selected.index
            )
            
            # Mantieni eventuali colonne non normalizzate
            if len(df_normalized.columns) < len(df.columns):
                other_cols = [col for col in df.columns if col not in df_normalized.columns]
                df_normalized = pd.concat([df_normalized, df[other_cols]], axis=1)
            
            df = df_normalized
        
        # Converti in array per il modello
        features_array = df.values
        
        # Reshape per il modello: (1, sequence_length, n_features)
        # Controlla numero di features
        if features_array.shape[1] != self.n_features:
            logger.warning(
                f"Numero features ({features_array.shape[1]}) diverso da "
                f"quelle del modello ({self.n_features}). Potrebbero esserci problemi."
            )
        
        input_array = features_array.reshape(1, self.sequence_length, -1)
        
        return input_array
    
    def predict_single(self, data: Union[pd.DataFrame, np.ndarray, List],
                      return_confidence: bool = False,
                      use_cache: bool = True) -> Dict[str, Any]:
        """
        Effettua una singola predizione.
        
        Args:
            data: Dati di input (ultimi N tick)
            return_confidence: Se True, calcola intervallo di confidenza
            use_cache: Se True, usa cache per performance
            
        Returns:
            Dizionario con risultati della predizione
        """
        # Genera cache key
        cache_key = None
        if use_cache:
            try:
                cache_key = str(hash(str(data)))
                if cache_key in self.prediction_cache:
                    logger.debug("Predizione recuperata da cache")
                    return self.prediction_cache[cache_key].copy()
            except:
                pass
        
        # Prepara dati
        input_data = self.prepare_input_data(data)
        
        # Effettua predizione
        start_time = datetime.now()
        prediction_raw = self.model.predict(input_data, verbose=0)
        prediction_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # De-normalizza se possibile
        prediction_value = float(prediction_raw.flatten()[0])
        if self.scaler is not None:
            try:
                # De-normalizza usando lo scaler
                dummy_array = np.zeros((1, self.n_features))
                dummy_array[0, 0] = prediction_value  # Assume price è la prima feature
                denorm_array = self.scaler.inverse_transform(dummy_array)
                prediction_denorm = float(denorm_array[0, 0])
            except:
                prediction_denorm = prediction_value
                logger.warning("Impossibile de-normalizzare la predizione")
        else:
            prediction_denorm = prediction_value
        
        # Risultato base
        result = {
            'prediction': prediction_denorm,
            'prediction_raw': prediction_value,
            'timestamp': datetime.now().isoformat(),
            'prediction_time_ms': prediction_time,
            'sequence_length_used': self.sequence_length,
            'model_used': self.model_metadata.get('model_name', 'unknown')
        }
        
        # Calcola confidence interval se richiesto
        if return_confidence:
            confidence_result = self.calculate_confidence_interval(input_data)
            result.update(confidence_result)
        
        # Calcola segnale di trading se possibile
        signal_result = self.generate_trading_signal(prediction_denorm, input_data)
        result.update(signal_result)
        
        # Salva in cache
        if use_cache and cache_key:
            self._update_cache(cache_key, result)
        
        logger.info(f"Predizione completata: {prediction_denorm:.6f} (time: {prediction_time:.1f}ms)")
        
        return result
    
    def predict_batch(self, data_list: List[Union[pd.DataFrame, np.ndarray, List]],
                     return_confidence: bool = False,
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Effettua predizioni batch per multiple sequenze.
        
        Args:
            data_list: Lista di dati di input
            return_confidence: Se True, calcola intervalli di confidenza
            batch_size: Dimensione batch per predizioni
            
        Returns:
            Lista di dizionari con risultati
        """
        logger.info(f"Avvio predizioni batch su {len(data_list)} sequenze...")
        
        results = []
        batch_inputs = []
        
        for i, data in enumerate(data_list):
            try:
                # Prepara input
                input_data = self.prepare_input_data(data)
                batch_inputs.append(input_data)
                
                # Processa batch quando raggiunge batch_size
                if len(batch_inputs) >= batch_size or i == len(data_list) - 1:
                    # Concatena batch
                    batch_array = np.concatenate(batch_inputs, axis=0)
                    
                    # Predizioni batch
                    batch_predictions = self.model.predict(batch_array, verbose=0)
                    
                    # Processa risultati batch
                    for j, pred in enumerate(batch_predictions):
                        idx = i - len(batch_inputs) + j + 1
                        
                        # De-normalizza se possibile
                        pred_value = float(pred.flatten()[0])
                        if self.scaler is not None:
                            try:
                                dummy_array = np.zeros((1, self.n_features))
                                dummy_array[0, 0] = pred_value
                                denorm_array = self.scaler.inverse_transform(dummy_array)
                                pred_denorm = float(denorm_array[0, 0])
                            except:
                                pred_denorm = pred_value
                        else:
                            pred_denorm = pred_value
                        
                        # Risultato base
                        result = {
                            'prediction': pred_denorm,
                            'prediction_raw': pred_value,
                            'sequence_index': idx,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Confidence interval se richiesto
                        if return_confidence:
                            conf_result = self.calculate_confidence_interval(batch_inputs[j])
                            result.update(conf_result)
                        
                        # Segnale di trading
                        signal_result = self.generate_trading_signal(pred_denorm, batch_inputs[j])
                        result.update(signal_result)
                        
                        results.append(result)
                    
                    # Reset batch
                    batch_inputs = []
                    
            except Exception as e:
                logger.error(f"Errore nella predizione {i}: {str(e)}")
                results.append({
                    'error': str(e),
                    'sequence_index': i,
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"Batch predictions completate: {len(results)} risultati")
        return results
    
    def calculate_confidence_interval(self, input_data: np.ndarray,
                                     n_bootstrap: int = 100) -> Dict[str, Any]:
        """
        Calcola intervallo di confidenza usando bootstrap.
        
        Args:
            input_data: Dati di input
            n_bootstrap: Numero di campioni bootstrap
            
        Returns:
            Dizionario con intervallo di confidenza
        """
        if n_bootstrap < 10:
            logger.warning(f"n_bootstrap troppo piccolo ({n_bootstrap}). Usando 10.")
            n_bootstrap = 10
        
        try:
            logger.debug(f"Calcolo confidence interval con {n_bootstrap} bootstrap samples...")
            
            # Bootstrap predictions
            bootstrap_predictions = []
            
            for _ in range(n_bootstrap):
                # Aggiungi rumore gaussiano piccolo per variabilità
                noise = np.random.normal(0, 0.01, size=input_data.shape)
                noisy_input = input_data + noise
                
                # Predizione
                pred = self.model.predict(noisy_input, verbose=0).flatten()[0]
                
                # De-normalizza se possibile
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
            
            # Calcola intervallo di confidenza
            if self.confidence_level == 0.95:
                z_score = 1.96  # Per 95% CI
            elif self.confidence_level == 0.99:
                z_score = 2.576
            else:
                # Approssimazione per altri livelli
                from scipy import stats
                z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            
            margin_of_error = z_score * std_pred
            ci_lower = mean_pred - margin_of_error
            ci_upper = mean_pred + margin_of_error
            
            # Calcola percentuali per segnali
            current_pred = bootstrap_predictions[0]  # Prima predizione (senza rumore)
            pct_from_lower = ((current_pred - ci_lower) / (ci_upper - ci_lower) * 100 
                            if ci_upper > ci_lower else 50)
            
            return {
                'confidence_interval': {
                    'lower': float(ci_lower),
                    'upper': float(ci_upper),
                    'mean': float(mean_pred),
                    'std': float(std_pred),
                    'margin_of_error': float(margin_of_error)
                },
                'confidence_level': self.confidence_level,
                'n_bootstrap_samples': n_bootstrap,
                'position_in_ci': float(pct_from_lower)  # Percentuale nell'intervallo
            }
            
        except Exception as e:
            logger.warning(f"Impossibile calcolare confidence interval: {str(e)}")
            return {
                'confidence_interval': None,
                'confidence_error': str(e)
            }
    
    def generate_trading_signal(self, prediction: float, 
                               input_data: np.ndarray = None) -> Dict[str, Any]:
        """
        Genera segnale di trading basato sulla predizione.
        
        Args:
            prediction: Valore predetto
            input_data: Dati di input (opzionale, per contesto)
            
        Returns:
            Dizionario con segnale di trading
        """
        # Soglie di default (possono essere personalizzate)
        BUY_THRESHOLD = 0.02    # 2% sopra
        SELL_THRESHOLD = -0.02  # 2% sotto
        STRONG_BUY = 0.05       # 5% sopra
        STRONG_SELL = -0.05     # 5% sotto
        
        # Se abbiamo input_data, calcoliamo riferimento dal prezzo attuale
        reference_price = None
        if input_data is not None and self.scaler is not None:
            try:
                # Estrai ultimo prezzo dalla sequenza
                last_price_normalized = input_data[0, -1, 0]  # Assume price è prima feature
                
                # De-normalizza
                dummy_array = np.zeros((1, self.n_features))
                dummy_array[0, 0] = last_price_normalized
                denorm_array = self.scaler.inverse_transform(dummy_array)
                reference_price = float(denorm_array[0, 0])
            except:
                pass
        
        # Se non abbiamo reference_price, usiamo una logica semplificata
        if reference_price is None:
            # Logica basata su soglie assolute (semplificata)
            if prediction > STRONG_BUY:
                signal = 'STRONG_BUY'
                strength = 1.0
            elif prediction > BUY_THRESHOLD:
                signal = 'BUY'
                strength = 0.7
            elif prediction < STRONG_SELL:
                signal = 'STRONG_SELL'
                strength = 1.0
            elif prediction < SELL_THRESHOLD:
                signal = 'SELL'
                strength = 0.7
            else:
                signal = 'NEUTRAL'
                strength = 0.0
            
            percent_change = prediction  # In questo caso, prediction è già il cambio %
            
        else:
            # Calcola cambio percentuale
            percent_change = ((prediction - reference_price) / reference_price * 100 
                            if reference_price != 0 else 0)
            
            # Genera segnale basato su percent_change
            if percent_change > STRONG_BUY * 100:  # Converti in percentuale
                signal = 'STRONG_BUY'
                strength = min(1.0, percent_change / (STRONG_BUY * 100 * 2))
            elif percent_change > BUY_THRESHOLD * 100:
                signal = 'BUY'
                strength = min(0.7, (percent_change - BUY_THRESHOLD * 100) / 
                             ((STRONG_BUY - BUY_THRESHOLD) * 100))
            elif percent_change < STRONG_SELL * 100:
                signal = 'STRONG_SELL'
                strength = min(1.0, abs(percent_change) / (abs(STRONG_SELL) * 100 * 2))
            elif percent_change < SELL_THRESHOLD * 100:
                signal = 'SELL'
                strength = min(0.7, (abs(percent_change) - abs(SELL_THRESHOLD) * 100) / 
                             ((abs(STRONG_SELL) - abs(SELL_THRESHOLD)) * 100))
            else:
                signal = 'NEUTRAL'
                strength = 0.0
        
        # Calcola confidence del segnale (semplificato)
        signal_confidence = min(0.95, 0.5 + abs(percent_change) / 10)
        
        # Aggiungi raccomandazioni
        recommendations = self._generate_trading_recommendations(signal, strength, percent_change)
        
        return {
            'trading_signal': signal,
            'signal_strength': float(strength),
            'signal_confidence': float(signal_confidence),
            'predicted_change_percent': float(percent_change),
            'reference_price': reference_price,
            'recommendations': recommendations,
            'signal_timestamp': datetime.now().isoformat()
        }
    
    def _generate_trading_recommendations(self, signal: str, strength: float, 
                                         percent_change: float) -> List[str]:
        """Genera raccomandazioni di trading basate sul segnale."""
        recommendations = []
        
        if signal in ['STRONG_BUY', 'BUY']:
            if strength > 0.8:
                recommendations.append("Considera posizione long significativa")
            elif strength > 0.5:
                recommendations.append("Considera posizione long moderata")
            else:
                recommendations.append("Considera piccolo ingresso long")
            
            recommendations.append("Imposta stop-loss al di sotto del supporto recente")
            
            if percent_change > 5:
                recommendations.append("Alto potenziale - considera trailing stop")
        
        elif signal in ['STRONG_SELL', 'SELL']:
            if strength > 0.8:
                recommendations.append("Considera posizione short significativa")
            elif strength > 0.5:
                recommendations.append("Considera posizione short moderata")
            else:
                recommendations.append("Considera piccolo ingresso short")
            
            recommendations.append("Imposta stop-loss al di sopra della resistenza recente")
            
            if percent_change < -5:
                recommendations.append("Forte downtrend - considera posizione conservativa")
        
        else:  # NEUTRAL
            recommendations.append("Mantieni posizioni esistenti")
            recommendations.append("Aspetta segnale più chiaro per nuovi ingressi")
            recommendations.append("Considera riduzione esposizione se volatilità alta")
        
        # Raccomandazioni generali
        recommendations.append("Mai rischiare più del 2% del capitale per trade")
        recommendations.append("Verifica con analisi fondamentale e tecnica")
        recommendations.append("Monitora volume e liquidità")
        
        return recommendations
    
    def predict_sequence(self, historical_data: pd.DataFrame,
                        future_steps: int = 5,
                        return_confidence: bool = False) -> pd.DataFrame:
        """
        Predice multiple steps futuri usando predizioni ricorsive.
        
        Args:
            historical_data: Dati storici
            future_steps: Numero di steps da predire
            return_confidence: Se True, calcola intervalli di confidenza
            
        Returns:
            DataFrame con predizioni future
        """
        logger.info(f"Avvio predizione ricorsiva per {future_steps} steps...")
        
        predictions = []
        current_data = historical_data.copy()
        
        for step in range(future_steps):
            try:
                # Prepara dati per predizione corrente
                if len(current_data) < self.sequence_length:
                    # Riutilizza dati se non abbiamo abbastanza
                    padding_needed = self.sequence_length - len(current_data)
                    padding_data = current_data.iloc[-padding_needed:].copy()
                    current_data = pd.concat([padding_data, current_data], ignore_index=True)
                
                # Prendi ultimi N tick
                recent_data = current_data.iloc[-self.sequence_length:]
                
                # Effettua predizione
                prediction_result = self.predict_single(
                    recent_data, 
                    return_confidence=return_confidence,
                    use_cache=False
                )
                
                # Estrai predizione
                pred_value = prediction_result['prediction']
                
                # Crea nuova riga per il futuro
                # (semplificato - nella realtà dovresti predire anche altre features)
                new_row = recent_data.iloc[-1:].copy()
                
                # Aggiorna prezzo predetto
                if 'price' in new_row.columns:
                    new_row['price'] = pred_value
                elif len(new_row.columns) > 0:
                    # Assume la prima colonna è il prezzo
                    new_row.iloc[0, 0] = pred_value
                
                # Aggiungi timestamp predetto
                if 'timestamp' in new_row.columns:
                    last_timestamp = pd.to_datetime(new_row['timestamp'].iloc[0])
                    new_timestamp = last_timestamp + timedelta(
                        minutes=1  # Assume intervallo di 1 minuto
                    )
                    new_row['timestamp'] = new_timestamp
                
                # Aggiungi colonna di predizione
                new_row['is_prediction'] = True
                new_row['prediction_step'] = step + 1
                
                # Aggiungi alla lista
                predictions.append({
                    'step': step + 1,
                    'prediction': pred_value,
                    'timestamp': new_row['timestamp'].iloc[0] if 'timestamp' in new_row else None,
                    'confidence_interval': prediction_result.get('confidence_interval'),
                    'trading_signal': prediction_result.get('trading_signal')
                })
                
                # Aggiungi ai dati per predizione ricorsiva
                current_data = pd.concat([current_data, new_row], ignore_index=True)
                
                logger.debug(f"Step {step + 1}: {pred_value:.4f}")
                
            except Exception as e:
                logger.error(f"Errore allo step {step}: {str(e)}")
                predictions.append({
                    'step': step + 1,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                break
        
        # Converti in DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        logger.info(f"Predizione ricorsiva completata: {len(predictions_df)} steps")
        return predictions_df
    
    def _update_cache(self, key: str, value: Dict):
        """Aggiorna cache delle predizioni."""
        try:
            if len(self.prediction_cache) >= self.cache_max_size:
                # Rimuovi la chiave più vecchia
                oldest_key = next(iter(self.prediction_cache))
                del self.prediction_cache[oldest_key]
            
            self.prediction_cache[key] = value
        except Exception as e:
            logger.warning(f"Errore nell'aggiornamento cache: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Restituisce informazioni sul modello caricato."""
        if self.model is None:
            return {'error': 'Modello non caricato'}
        
        info = {
            'model_loaded': True,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'model_input_shape': self.model.input_shape,
            'model_output_shape': self.model.output_shape,
            'model_layers': len(self.model.layers),
            'model_trainable_params': self.model.count_params(),
            'scaler_loaded': self.scaler is not None,
            'feature_names': self.feature_names,
            'confidence_level': self.confidence_level
        }
        
        info.update(self.model_metadata)
        return info
    
    def save_prediction_log(self, predictions: List[Dict], 
                           filepath: str = 'predictions_log.csv'):
        """
        Salva log delle predizioni in CSV.
        
        Args:
            predictions: Lista di predizioni
            filepath: Percorso del file CSV
        """
        try:
            if not predictions:
                logger.warning("Nessuna predizione da salvare")
                return
            
            # Converti in DataFrame
            df = pd.DataFrame(predictions)
            
            # Salva
            df.to_csv(filepath, index=False)
            logger.info(f"Log predizioni salvato in: {filepath}")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio del log: {str(e)}")


# Funzioni di utilità
def create_predictor_from_training(model_dir: str = 'models',
                                  model_prefix: str = 'trading_model') -> TradingPredictor:
    """
    Crea un predictor cercando automaticamente l'ultimo modello addestrato.
    
    Args:
        model_dir: Directory dei modelli
        model_prefix: Prefisso dei file modello
        
    Returns:
        TradingPredictor configurato
    """
    import os
    import glob
    
    # Cerca file modello
    model_files = glob.glob(os.path.join(model_dir, f'{model_prefix}*.h5'))
    
    if not model_files:
        raise FileNotFoundError(f"Nessun modello trovato in {model_dir} con prefisso {model_prefix}")
    
    # Prendi l'ultimo modello (per timestamp)
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model = model_files[0]
    
    # Cerca scaler corrispondente
    model_name = os.path.basename(latest_model).replace('.h5', '')
    scaler_file = os.path.join(model_dir, f'{model_name}_scaler.pkl')
    
    if not os.path.exists(scaler_file):
        logger.warning(f"Scaler non trovato: {scaler_file}")
        scaler_file = None
    
    # Cerca metadata
    metadata_file = os.path.join(model_dir, f'{model_name}_metadata.json')
    feature_names = None
    
    if os.path.exists(metadata_file):
        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Estrai feature names se disponibili
            if 'feature_names' in metadata:
                feature_names = metadata['feature_names']
        except:
            pass
    
    # Crea predictor
    predictor = TradingPredictor(
        model_path=latest_model,
        scaler_path=scaler_file,
        feature_names=feature_names
    )
    
    logger.info(f"Predictor creato dal modello: {os.path.basename(latest_model)}")
    return predictor


def predict_live_data(predictor: TradingPredictor, 
                     live_data_source: Any,
                     prediction_interval: int = 60) -> Dict[str, Any]:
    """
    Simula predizioni su dati live.
    
    Args:
        predictor: Predictor configurato
        live_data_source: Fonte dati live (simulata o reale)
        prediction_interval: Intervallo tra predizioni in secondi
        
    Returns:
        Dizionario con risultati live
    """
    # Questa è una funzione di esempio per dati live
    # In pratica, dovresti integrarla con la tua fonte dati reali
    
    logger.info(f"Avvio predizioni live (intervallo: {prediction_interval}s)...")
    
    # Esempio: simula dati live
    import time
    from datetime import datetime
    
    predictions_log = []
    
    try:
        for i in range(5):  # Esempio: 5 predizioni
            # Simula ottenimento dati live
            logger.info(f"Predizione live #{i+1}...")
            
            # Genera dati di esempio
            np.random.seed(42 + i)
            sample_data = pd.DataFrame({
                'price': np.random.randn(predictor.sequence_length).cumsum() + 100,
                'indicator1': np.random.randn(predictor.sequence_length),
                'indicator2': np.random.randn(predictor.sequence_length),
                'volume': np.random.randint(100, 10000, predictor.sequence_length)
            })
            
            # Effettua predizione
            result = predictor.predict_single(
                sample_data,
                return_confidence=True
            )
            
            # Aggiungi a log
            result['live_sequence'] = i + 1
            predictions_log.append(result)
            
            logger.info(f"  Segnale: {result.get('trading_signal')}, "
                       f"Predizione: {result.get('prediction'):.4f}")
            
            # Attendi per prossima predizione
            if i < 4:  # Non aspettare dopo l'ultima
                time.sleep(prediction_interval)
    
    except KeyboardInterrupt:
        logger.info("Predizioni live interrotte dall'utente")
    except Exception as e:
        logger.error(f"Errore nelle predizioni live: {str(e)}")
    
    # Salva log
    if predictions_log:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'live_predictions_{timestamp}.csv'
        predictor.save_prediction_log(predictions_log, log_file)
    
    return {
        'total_predictions': len(predictions_log),
        'predictions_log': predictions_log,
        'last_signal': predictions_log[-1].get('trading_signal') if predictions_log else None
    }


if __name__ == "__main__":
    # Test del predictor
    print("Test del predictor.py")
    print("=" * 50)
    
    try:
        # Crea un modello di esempio per test
        import os
        os.makedirs('test_models', exist_ok=True)
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        # Modello semplice
        model_test = Sequential([
            LSTM(16, input_shape=(10, 5), return_sequences=False),
            Dense(1)
        ])
        
        model_test.compile(optimizer='adam', loss='mse')
        
        # Salva modello
        model_path = 'test_models/test_predictor_model.h5'
        model_test.save(model_path)
        
        # Crea scaler di esempio
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 5))
        scaler_path = 'test_models/test_predictor_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        
        print(f"\nModello e scaler di test creati")
        
        # Test predictor
        predictor = TradingPredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            sequence_length=10,
            feature_names=['price', 'ind1', 'ind2', 'ind3', 'volume'],
            confidence_level=0.95
        )
        
        # Test get_model_info
        print("\nInformazioni modello:")
        model_info = predictor.get_model_info()
        for key, value in model_info.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"  {key}: {value}")
        
        # Test predizione singola
        print("\nTest predizione singola...")
        
        # Crea dati di esempio
        test_data = pd.DataFrame({
            'price': np.random.randn(10).cumsum() + 100,
            'ind1': np.random.randn(10),
            'ind2': np.random.randn(10),
            'ind3': np.random.randn(10),
            'volume': np.random.randint(100, 10000, 10)
        })
        
        result = predictor.predict_single(
            test_data,
            return_confidence=True
        )
        
        print(f"  Predizione: {result['prediction']:.4f}")
        print(f"  Segnale: {result['trading_signal']}")
        print(f"  Confidence interval: [{result['confidence_interval']['lower']:.4f}, "
              f"{result['confidence_interval']['upper']:.4f}]")
        
        # Test predizioni batch
        print("\nTest predizioni batch...")
        
        batch_data = [test_data] * 3  # 3 sequenze identiche
        batch_results = predictor.predict_batch(
            batch_data,
            return_confidence=False
        )
        
        print(f"  Predizioni batch completate: {len(batch_results)}")
        for i, res in enumerate(batch_results[:2]):  # Mostra prime 2
            print(f"    Batch {i+1}: {res['prediction']:.4f} ({res['trading_signal']})")
        
        # Test predizione ricorsiva
        print("\nTest predizione ricorsiva...")
        
        # Crea dati storici
        historical_data = pd.DataFrame({
            'price': np.random.randn(20).cumsum() + 100,
            'timestamp': pd.date_range(start='2024-01-01', periods=20, freq='1min')
        })
        
        future_predictions = predictor.predict_sequence(
            historical_data,
            future_steps=3,
            return_confidence=False
        )
        
        print(f"  Predizioni future:")
        for _, row in future_predictions.iterrows():
            if 'prediction' in row:
                print(f"    Step {row['step']}: {row['prediction']:.4f} ({row.get('trading_signal', 'N/A')})")
        
        print("\n" + "=" * 50)
        print("TEST COMPLETATO CON SUCCESSO!")
        print("=" * 50)
        
        # Pulisci file test
        import shutil
        if os.path.exists('test_models'):
            shutil.rmtree('test_models')
            print("\nFile di test puliti")
        
    except Exception as e:
        print(f"\nErrore durante il test: {str(e)}")
        import traceback
        traceback.print_exc()