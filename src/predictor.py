"""
predictor.py
Predizione con TP/SL - Versione semplificata ma completa
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf  # 🔥 USA tensorflow.keras
from typing import Dict, List, Any, Union
import joblib
import logging
from datetime import datetime
import warnings
from timestamp_features import TimestampFeatureExtractor
warnings.filterwarnings('ignore')
from typing import Dict, List, Any, Union, Optional

# 🔥 AGGIUNGI QUESTI IMPORT PER load_model
import json
import glob
import h5py

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
        """Carica modello e scaler con gestione COMPLETA compatibilità Keras."""
        try:
            logger.info(f"🔄 Caricamento modello: {os.path.basename(model_path)}")
            
            import h5py
            import json
            
            # 🔥 PASSO 1: VERIFICA FORMATO E STRATEGIA DI CARICAMENTO
            file_ext = os.path.splitext(model_path)[1].lower()
            
            # Strategie di caricamento in base al formato
            if file_ext == '.keras':
                # FORMATO MODERNO KERAS - carica normalmente
                logger.info("📦 Formato .keras rilevato (moderno)")
                self.model = tf.keras.models.load_model(model_path)
                logger.info("✅ Modello .keras caricato normalmente")
                
            elif file_ext == '.h5':
                # FORMATO .h5 - gestione avanzata
                logger.info("📦 Formato .h5 rilevato (legacy)")
                self.model = self._load_h5_model_with_compatibility(model_path)
                
            else:
                raise ValueError(f"Formato file non supportato: {file_ext}")
            
            # 🔥 PASSO 2: VERIFICA E RICOMPILAZIONE SE NECESSARIA
            self._ensure_model_is_compiled(model_path)
            
            # 🔥 PASSO 3: CARICA SCALER CON RICERCA INTELLIGENTE
            self._load_scaler_with_intelligence(model_path, scaler_path)
            
            # 🔥 PASSO 4: ESTRAI INFO DIMENSIONALI
            self._extract_model_dimensions()
            
            # 🔥 PASSO 5: CARICA METADATA
            self._load_model_metadata(model_path)
            
            logger.info("✅ Modello e scaler caricati con successo!")
            
        except Exception as e:
            logger.error(f"❌ Errore critico nel caricamento: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_h5_model_with_compatibility(self, model_path: str) -> tf.keras.Model:
        """Carica modello .h5 con gestione avanzata compatibilità."""
        import h5py
        
        # PRIMA: Verifica se è un modello salvato in formato separato
        try:
            with h5py.File(model_path, 'r') as f:
                if 'saved_format' in f.attrs and f.attrs['saved_format'] == 'separated_components':
                    logger.info("📦 Modello in formato separato rilevato")
                    return self._load_separated_model(model_path, f)
        except:
            pass  # Non è formato separato, procedi normalmente
        
        # SECONDA: Prova a caricare con diverse strategie
        load_strategies = [
            self._try_load_h5_normal,
            self._try_load_h5_with_custom_objects,
            self._try_load_h5_compile_false
        ]
        
        last_error = None
        for strategy in load_strategies:
            try:
                model = strategy(model_path)
                if model:
                    return model
            except Exception as e:
                last_error = e
                continue
        
        # Se tutte le strategie falliscono
        raise RuntimeError(f"Tutte le strategie di caricamento .h5 fallite. Ultimo errore: {last_error}")

    def _load_separated_model(self, model_path: str, h5_file) -> tf.keras.Model:
        """Carica modello in formato separato (pesi + architettura)."""
        import json
        
        model_dir = os.path.dirname(model_path)
        
        # Carica architettura
        arch_file = h5_file.attrs['architecture_file']
        arch_path = os.path.join(model_dir, arch_file)
        
        with open(arch_path, 'r') as f:
            model_json = f.read()
        
        # Crea modello dall'architettura
        model = tf.keras.models.model_from_json(model_json)
        
        # Carica pesi
        weights_file = h5_file.attrs['weights_file']
        weights_path = os.path.join(model_dir, weights_file)
        model.load_weights(weights_path)
        
        logger.info(f"✅ Modello caricato da formato separato")
        logger.info(f"   Architettura: {arch_file}")
        logger.info(f"   Pesi: {weights_file}")
        
        return model

    def _try_load_h5_normal(self, model_path: str) -> tf.keras.Model:
        """Prima strategia: carica normalmente."""
        return tf.keras.models.load_model(model_path)

    def _try_load_h5_with_custom_objects(self, model_path: str) -> tf.keras.Model:
        """Seconda strategia: carica con custom_objects."""
        custom_objects = {
            # Metriche come stringhe
            'mse': 'mse',
            'mae': 'mae',
            'accuracy': 'accuracy',
            'binary_accuracy': 'binary_accuracy',
            'auc': 'auc',
            'categorical_accuracy': 'categorical_accuracy',
            
            # Layer
            'BatchNormalization': tf.keras.layers.BatchNormalization,
            'Dropout': tf.keras.layers.Dropout,
            'Bidirectional': tf.keras.layers.Bidirectional,
            'LSTM': tf.keras.layers.LSTM,
            'Dense': tf.keras.layers.Dense,
            'Input': tf.keras.layers.Input,
        }
        
        return tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects
        )

    def _try_load_h5_compile_false(self, model_path: str) -> tf.keras.Model:
        """Terza strategia: carica senza compilare."""
        return tf.keras.models.load_model(
            model_path, 
            compile=False
        )

    def _ensure_model_is_compiled(self, model_path: str):
        """Garantisce che il modello sia compilato."""
        if self.model.compiled_loss is None:
            logger.info("🔧 Modello non compilato, ricompilo...")
            
            # Determina tipo di problema
            problem_type = self._detect_problem_type(model_path)
            
            # Configura in base al tipo
            if problem_type == 'binary_classification':
                self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("✅ Ricompilato per classificazione binaria")
                
            elif problem_type == 'multi_classification':
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("✅ Ricompilato per classificazione multi-classe")
                
            else:  # regressione di default
                self.model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
                logger.info("✅ Ricompilato per regressione")

    def _get_current_price_from_data(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Optional[float]:
        """
        Estrae il prezzo corrente dai dati se disponibili.
        
        Returns:
            float: Prezzo corrente, o None se non disponibile
        """
        # Se abbiamo dati recenti nel predictor
        if hasattr(self, 'last_data') and self.last_data is not None:
            try:
                if isinstance(self.last_data, pd.DataFrame):
                    # Cerca colonna 'price' o 'close' o la prima colonna numerica
                    for col in ['price', 'close', 'Price', 'Close']:
                        if col in self.last_data.columns:
                            return float(self.last_data[col].iloc[-1])
                    
                    # Se non trova, cerca la prima colonna numerica
                    numeric_cols = self.last_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        return float(self.last_data[numeric_cols[0]].iloc[-1])
                
                elif isinstance(self.last_data, np.ndarray):
                    # Assumi che il prezzo sia nella prima colonna
                    if len(self.last_data.shape) > 1:
                        return float(self.last_data[-1, 0])
                    else:
                        return float(self.last_data[-1])
            except Exception as e:
                logger.warning(f"Impossibile estrarre prezzo dai dati: {e}")
        
        return None

    def _extract_current_price_from_data(self, data: Union[pd.DataFrame, np.ndarray]) -> float:
        """
        Estrae il prezzo corrente dai dati di input.
        
        Args:
            data: Dati di input (DataFrame o array)
        
        Returns:
            float: Prezzo corrente (default 1.10000 se non trovato)
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Cerca colonne prezzo
                price_columns = ['price', 'Price', 'PRICE', 'close', 'Close', 'CLOSE']
                
                for col in price_columns:
                    if col in data.columns:
                        price = data[col].iloc[-1]
                        logger.info(f"📊 Prezzo corrente estratto da colonna '{col}': {price:.5f}")
                        return float(price)
                
                # Se non trova, cerca la prima colonna numerica
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price = data[numeric_cols[0]].iloc[-1]
                    logger.info(f"📊 Prezzo corrente dalla prima colonna numerica: {price:.5f}")
                    return float(price)
            
            elif isinstance(data, np.ndarray):
                # Array numpy - assumi che il prezzo sia il primo elemento
                if len(data.shape) == 1:
                    price = data[-1]
                else:
                    price = data[-1, 0]
                
                logger.info(f"📊 Prezzo corrente da array numpy: {price:.5f}")
                return float(price)
        
        except Exception as e:
            logger.warning(f"⚠️  Impossibile estrarre prezzo corrente: {e}")
        
        # Fallback
        logger.warning("⚠️  Prezzo corrente non trovato, uso default 1.10000")
        return 1.10000

    def _detect_problem_type(self, model_path: str) -> str:
        """Rileva tipo di problema dal modello o metadata."""
        
        # 1. Prova dai metadata
        metadata_path = model_path.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if 'problem_type' in metadata:
                    return metadata['problem_type']
            except:
                pass
        
        # 2. Rileva dall'architettura del modello
        try:
            last_layer = self.model.layers[-1]
            
            if hasattr(last_layer, 'activation'):
                activation = last_layer.activation
                
                # Ottieni nome attivazione
                if hasattr(activation, '__name__'):
                    act_name = activation.__name__
                elif isinstance(activation, str):
                    act_name = activation
                else:
                    act_name = str(activation)
                
                # Determina tipo
                if 'sigmoid' in act_name.lower() and self.model.output_shape[-1] == 1:
                    return 'binary_classification'
                elif 'softmax' in act_name.lower():
                    return 'multi_classification'
        except:
            pass
        
        # Default
        return 'regression'

    def _load_scaler_with_intelligence(self, model_path: str, scaler_path: str = None):
        """Carica scaler con ricerca intelligente."""
        import glob
        
        # Se scaler_path è fornito e esiste
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler caricato dal path fornito: {os.path.basename(scaler_path)}")
            return
        
        # Ricerca intelligente
        model_dir = os.path.dirname(model_path)
        model_base = os.path.basename(model_path)
        
        # Rimuovi estensioni
        model_base_no_ext = model_base.replace('.h5', '').replace('.keras', '')
        
        # Pattern di ricerca in ordine di priorità
        search_patterns = [
            # Pattern esatto
            os.path.join(model_dir, f"{model_base_no_ext}_scaler.pkl"),
            # Pattern con wildcard
            os.path.join(model_dir, f"{model_base_no_ext}*scaler*.pkl"),
            # Pattern più generico
            os.path.join(model_dir, f"{model_base_no_ext.split('_classifier')[0]}*scaler*.pkl"),
            # Tutti gli scaler
            os.path.join(model_dir, "*scaler*.pkl")
        ]
        
        for pattern in search_patterns:
            scaler_files = glob.glob(pattern)
            if scaler_files:
                # Prendi il più recente
                scaler_files.sort(key=os.path.getmtime, reverse=True)
                selected_scaler = scaler_files[0]
                
                self.scaler = joblib.load(selected_scaler)
                logger.info(f"✅ Scaler trovato automaticamente: {os.path.basename(selected_scaler)}")
                return
        
        # Se non trovato
        logger.error("❌ Nessuno scaler trovato!")
        logger.error("   Pattern cercati:")
        for pattern in search_patterns[:3]:
            logger.error(f"   - {os.path.basename(pattern)}")
        
        raise FileNotFoundError(f"Nessuno scaler trovato per {model_base}")

    def _extract_model_dimensions(self):
        """Estrai sequence_length e n_features dal modello."""
        if self.model is None:
            return
        
        if len(self.model.input_shape) == 3:
            if self.sequence_length is None:
                self.sequence_length = int(self.model.input_shape[1])
            
            self.n_features = int(self.model.input_shape[2])
            
            logger.info(f"📐 Dimensioni modello:")
            logger.info(f"   Sequence length: {self.sequence_length}")
            logger.info(f"   Numero features: {self.n_features}")

    def _load_model_metadata(self, model_path: str):
        """Carica metadata del modello."""
        import json
        
        metadata_path = model_path.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                
                logger.info(f"📄 Metadata caricati: {os.path.basename(metadata_path)}")
                
                # Estrai feature_names se disponibili
                if not self.feature_names and 'feature_names' in self.model_metadata:
                    self.feature_names = self.model_metadata['feature_names']
                    logger.info(f"   Feature names: {len(self.feature_names)} features")
                    
            except Exception as e:
                logger.warning(f"⚠️  Impossibile caricare metadata: {e}")
                self.model_metadata = {}
        else:
            logger.warning("⚠️  File metadata non trovato")
            self.model_metadata = {}
    
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
        
        # 🔥 PASSO 0: VALIDA COMPATIBILITÀ SCALER
        if self.scaler is not None:
            df = self.validate_scaler_compatibility(df)
        
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
    
    def validate_scaler_compatibility(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Verifica e corregge la compatibilità tra dati e scaler.
        Restituisce DataFrame compatibile.
        """
        if self.scaler is None:
            logger.warning("⚠️  Nessuno scaler disponibile")
            return data_df
        
        # 🔥 1. Verifica se lo scaler ha attributo n_features_in_
        expected_features = getattr(self.scaler, 'n_features_in_', None)
        if expected_features is None:
            logger.warning("⚠️  Scaler senza n_features_in_, continuo senza validazione")
            return data_df
        
        # 🔥 2. Conta features numeriche nei dati
        numeric_columns = data_df.select_dtypes(include=[np.number]).columns
        actual_features = len(numeric_columns)
        
        logger.info(f"🔍 Validazione scaler:")
        logger.info(f"   Features dati: {actual_features}")
        logger.info(f"   Features scaler: {expected_features}")
        
        # 🔥 3. Se match perfetto, tutto OK
        if actual_features == expected_features:
            logger.info("✅ Match perfetto features")
            return data_df
        
        # 🔥 4. Se più features nei dati, riduci
        if actual_features > expected_features:
            logger.warning(f"⚠️  Troppe features: {actual_features} > {expected_features}")
            logger.info(f"   Mantengo prime {expected_features} features")
            
            # Mantieni solo prime N colonne numeriche
            selected_columns = list(numeric_columns[:expected_features])
            
            # Mantieni anche timestamp se presente
            if 'timestamp' in data_df.columns:
                selected_columns = ['timestamp'] + selected_columns
            
            return data_df[selected_columns]
        
        # 🔥 5. Se meno features nei dati, aggiungi dummy
        if actual_features < expected_features:
            logger.warning(f"⚠️  Poche features: {actual_features} < {expected_features}")
            logger.info(f"   Aggiungo {expected_features - actual_features} colonne dummy")
            
            # Crea DataFrame di zeri per le features mancanti
            for i in range(actual_features, expected_features):
                col_name = f"dummy_feature_{i}"
                data_df[col_name] = 0.0
            
            logger.info(f"   Dati ora hanno {expected_features} features")
        
        return data_df

    def predict_single(self, data: Union[pd.DataFrame, np.ndarray],
                  return_confidence: bool = False) -> Dict[str, Any]:
        """Effettua singola predizione."""
        # 🔥 SALVA I DATI PER USO SUCCESSIVO
        self.last_data = data.copy() if isinstance(data, pd.DataFrame) else data
        
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
        
        # 🔥 OTTIENI PREZZO CORRENTE DAI DATI
        current_price = self._extract_current_price_from_data(data)
        
        # Risultato base
        result = {
            'prediction': prediction_denorm,
            'prediction_raw': prediction_value,
            'current_price': current_price,  # 🔥 AGGIUNTO
            'timestamp': datetime.now().isoformat(),
            'sequence_length_used': self.sequence_length
        }
        
        # Confidence interval
        if return_confidence:
            ci_result = self.calculate_confidence_interval(input_data)
            result.update(ci_result)
        
        # 🔥 PASSA IL PREZZO CORRENTE A calculate_tp_sl
        signal_result = self.generate_trading_signal(prediction_denorm, current_price)
        result.update(signal_result)
        
        logger.info(f"✅ Predizione: {prediction_denorm:.6f}")
        logger.info(f"   Prezzo corrente: {current_price:.5f}")
        
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
                           current_price: float = None,
                           timestamp: pd.Series = None) -> Dict[str, Any]:
        """Genera segnale con adattamento temporale."""
        
        # 🔥 CALCOLA TP/SL CON PREZZO CORRENTE
        if current_price is None:
            current_price = 1.10000
            logger.warning("⚠️  Nessun prezzo fornito, uso default 1.10000")
        
        tp_sl_result = self.calculate_tp_sl(prediction, current_price)
        
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
            
            # 🔥 RICALCOLA PREZZI CON NUOVI PIPS
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
    
    def calculate_tp_sl(self, prediction: float, current_price: float = None) -> Dict[str, Any]:
        """
        Calcola Take Profit e Stop Loss con R/R = 4:1.
        
        Args:
            prediction: Valore predetto dal modello
            current_price: Prezzo corrente REALE (se None, usa 1.10000 come fallback)
        """
        # Se non viene fornito un prezzo corrente, prova a calcolarlo dai dati
        if current_price is None:
            current_price = self._get_current_price_from_data()
            if current_price is None:
                logger.warning("⚠️  Prezzo corrente non disponibile, uso default 1.10000")
                current_price = 1.10000
        
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
        
        # 🔥 CALCOLA PREZZI REALI
        pip_value = 0.0001
        
        if operation == "BUY":
            tp_price = current_price + (tp_pips * pip_value)
            sl_price = current_price - (sl_pips * pip_value)
        else:
            tp_price = current_price - (tp_pips * pip_value)
            sl_price = current_price + (sl_pips * pip_value)
        
        # Round per forex (5 decimali per EUR/USD)
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
    
    def _calculate_position_size(self, entry_price: float, sl_pips: int, risk_percent: float = 4.0) -> Dict[str, Any]:
        """Calcola dimensionamento posizione con rischio personalizzabile."""
        ACCOUNT_BALANCE = 10000.0
        LOT_SIZE = 100000
        
        risk_amount = ACCOUNT_BALANCE * (risk_percent / 100.0)
        
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
            'risk_percent': risk_percent
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

    # In predictor.py, aggiungi questa funzione:
    def ensure_scaler_compatibility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Garantisce che i dati abbiano stesso numero di features dello scaler."""
        if self.scaler is None:
            return data
        
        expected_features = getattr(self.scaler, 'n_features_in_', None)
        if expected_features and len(data.columns) != expected_features:
            logger.warning(f"Mismatch features: dati={len(data.columns)}, scaler={expected_features}")
            # Mantieni solo le prime N features
            return data.iloc[:, :expected_features]
        return data
    
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

