"""
model_trainer.py
Addestramento modello LSTM con salvataggio FORZATO dello scaler .pkl
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime
import os
from tensorflow.keras import metrics

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed per riproducibilità
np.random.seed(42)
tf.random.set_seed(42)

class LSTMTradingModel:
    """Classe per addestrare modelli LSTM con gestione scaler robusta."""
    
    def __init__(self, model_type: str = 'bidirectional', 
                 model_name: str = None,
                 data_loader = None):  # Aggiungi data_loader
        self.model_type = model_type
        self.model_name = model_name or f"trading_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = None
        self.history = None
        self.scaler = None
        self.data_loader = data_loader  # Salva riferimento al data_loader
        self.sequence_length = None
        self.n_features = None
        
        logger.info(f"LSTMTradingModel inizializzato. Tipo: {model_type}")
    
    # In model_trainer.py - build_model()
    def build_model(self, input_shape: Tuple[int, int], 
                output_units: int = 1, 
                problem_type: str = 'classification') -> Model:
        """
        Costruisce modello per classificazione TP/SL.
        """
        sequence_length, n_features = input_shape
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        logger.info(f"Costruzione modello con {n_features} features")
        
        inputs = Input(shape=(sequence_length, n_features))
        x = inputs
        
        # Architettura
        if self.model_type == 'bidirectional':
            x = Bidirectional(LSTM(128, return_sequences=True))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = Bidirectional(LSTM(64))(x)
        else:
            x = LSTM(128, return_sequences=True)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = LSTM(64)(x)
        
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # 🔥 OUTPUT PER CLASSIFICAZIONE
        if problem_type == 'classification':
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics_list = ['accuracy', 'AUC']
        else:
            outputs = Dense(output_units, activation='linear')(x)
            loss = 'mse'
            metrics_list = ['mae', 'mse']
        
        # 🔥 CORREZIONE CRITICA: Assegna il modello a self.model
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=loss,
            metrics=metrics_list
        )
        
        self.model = model  # 🔥 QUESTA RIGA MANCAVA!
        
        logger.info(f"✅ Modello costruito: {model.count_params()} parametri")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32) -> Dict[str, Any]:
        """Addestra il modello."""
        
        # 🔥 VERIFICA CHE IL MODELLO ESISTA
        if self.model is None:
            raise ValueError("Modello non costruito! Chiama build_model() prima di train().")
        
        logger.info(f"Training: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples")
        
        # Callbacks
        callbacks = self._get_default_callbacks()
        
        # Training
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        # Valutazione
        train_metrics = self.evaluate(X_train, y_train, prefix='train_')
        val_metrics = self.evaluate(X_val, y_val, prefix='val_')
        
        metrics_dict = {**train_metrics, **val_metrics}
        
        logger.info("✅ Training completato!")
        
        return {
            'history': self.history.history,
            'metrics': metrics_dict,
            'model': self.model
        }
    
    def _get_default_callbacks(self) -> List:
        """Restituisce callbacks di default."""
        os.makedirs('models/checkpoints', exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'models/checkpoints/{self.model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return callbacks
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                 prefix: str = '') -> Dict[str, float]:
        """Valuta il modello."""
        if self.model is None:
            raise ValueError("Modello non addestrato!")
        
        y_pred = self.model.predict(X, verbose=0)
        
        # Appiattisci
        y_flat = y.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calcola metriche
        mse = mean_squared_error(y_flat, y_pred_flat)
        mae = mean_absolute_error(y_flat, y_pred_flat)
        
        # R² con gestione errori
        try:
            ss_res = np.sum((y_flat - y_pred_flat) ** 2)
            ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
            
            if ss_tot == 0:
                r2 = 1.0
            else:
                r2 = 1 - (ss_res / ss_tot)
        except:
            r2 = -np.inf
        
        return {
            f'{prefix}mse': mse,
            f'{prefix}mae': mae,
            f'{prefix}r2': r2
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Effettua predizioni."""
        if self.model is None:
            raise ValueError("Modello non addestrato!")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, model_path: str = None, 
                  scaler_path: str = None,
                  metadata: Dict = None,
                  force_scaler_save: bool = True):
        """
        Salva modello e scaler.
        IMPORTANTE: Salva SEMPRE lo scaler, anche se deve crearne uno di default.
        """
        # Crea directory se non esiste
        os.makedirs('models', exist_ok=True)
        
        # Path di default
        if model_path is None:
            model_path = f'models/{self.model_name}.h5'
        
        if scaler_path is None:
            scaler_path = f'models/{self.model_name}_scaler.pkl'
        
        # 1. Salva modello
        self.model.save(model_path)
        logger.info(f"✅ Modello salvato: {model_path}")
        
        # 2. CRITICO: Salva SEMPRE lo scaler
        scaler_to_save = None
        
        # Prova a ottenere lo scaler dal data_loader
        if self.data_loader is not None and hasattr(self.data_loader, 'scaler'):
            scaler_to_save = self.data_loader.scaler
            logger.info("Usando scaler dal data_loader")
        
        # Se non disponibile, prova a crearne uno
        if scaler_to_save is None and force_scaler_save:
            logger.warning("Scaler non disponibile, creo scaler di default...")
            scaler_to_save = self._create_default_scaler()
        
        # Salva lo scaler
        if scaler_to_save is not None:
            joblib.dump(scaler_to_save, scaler_path)
            logger.info(f"✅ Scaler salvato: {scaler_path}")
            
            # Salva anche parametri dello scaler per debug
            try:
                scaler_params = {
                    'mean_': scaler_to_save.mean_.tolist() if hasattr(scaler_to_save, 'mean_') else [],
                    'scale_': scaler_to_save.scale_.tolist() if hasattr(scaler_to_save, 'scale_') else [],
                    'n_features_in_': scaler_to_save.n_features_in_ if hasattr(scaler_to_save, 'n_features_in_') else 0,
                    'type': str(type(scaler_to_save))
                }
                
                params_path = scaler_path.replace('.pkl', '_params.json')
                import json
                with open(params_path, 'w') as f:
                    json.dump(scaler_params, f, indent=2)
                logger.debug(f"Parametri scaler salvati: {params_path}")
            except Exception as e:
                logger.warning(f"Impossibile salvare parametri scaler: {e}")
        else:
            logger.error("❌ IMPOSSIBILE SALVARE SCALER!")
        
        # 3. Salva metadata CON TIMESTAMP INFO
        metadata_path = f'models/{self.model_name}_metadata.json'
        
        if metadata is None:
            metadata = {}
        
        # 🔥 AGGIUNGI INFO TIMESTAMP FEATURES
        if self.data_loader is not None:
            # Feature names dal data_loader
            if hasattr(self.data_loader, 'feature_names'):
                metadata['feature_names'] = self.data_loader.feature_names
            
            # Identifica timestamp features
            if hasattr(self.data_loader, 'feature_names'):
                timestamp_features = [f for f in self.data_loader.feature_names 
                                    if any(keyword in f.lower() for keyword in 
                                        ['hour', 'day', 'week', 'session', 'sin', 'cos'])]
                if timestamp_features:
                    metadata['timestamp_features'] = timestamp_features
        
        metadata_to_save = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'training_date': datetime.now().isoformat(),
            'scaler_saved': scaler_to_save is not None,
            'scaler_path': os.path.basename(scaler_path) if scaler_to_save is not None else None,
            **metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2)
        
        logger.info(f"✅ Metadata salvati: {metadata_path}")
        
        # 🔥 LOG SPECIFICO PER TIMESTAMP FEATURES
        if metadata_to_save.get('timestamp_features'):
            logger.info("📅 TIMESTAMP FEATURES SALVATE:")
            for feat in metadata_to_save['timestamp_features']:
                logger.info(f"   • {feat}")
        
        # 4. Log di verifica
        logger.info("📁 VERIFICA FILE CREATI:")
        logger.info(f"   • {os.path.basename(model_path)}")
        logger.info(f"   • {os.path.basename(scaler_path)}")
        logger.info(f"   • {os.path.basename(metadata_path)}")
    
    def _create_default_scaler(self):
        """Crea uno scaler di default per emergenze."""
        from sklearn.preprocessing import StandardScaler
        
        logger.warning("Creazione scaler di default...")
        
        # Crea scaler
        scaler = StandardScaler()
        
        # Fit con dati dummy basati su forex tipico
        # Supponiamo 9 features come nel tuo caso
        dummy_data = np.array([
            # EUR/USD tipico con indicatori
            [1.1010, 1.1000, 1.0990, 0.0010, 0.0008, 50.0, 50.0, 0.0010, -0.0010],
            [1.1020, 1.1010, 1.1000, 0.0020, 0.0015, 60.0, 55.0, 0.0020, -0.0020],
            [1.1000, 1.0990, 1.0980, -0.0010, -0.0005, 40.0, 45.0, -0.0010, 0.0010],
            [1.1030, 1.1020, 1.1010, 0.0030, 0.0020, 70.0, 60.0, 0.0030, -0.0030],
            [1.0990, 1.0980, 1.0970, -0.0020, -0.0010, 30.0, 40.0, -0.0020, 0.0020]
        ])
        
        # Ripeti per avere più dati
        dummy_data = np.tile(dummy_data, (20, 1))
        
        scaler.fit(dummy_data)
        
        logger.warning("⚠️  Scaler di default creato (non ottimale per predizioni reali)")
        
        return scaler
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """Carica modello e scaler."""
        logger.info(f"Caricamento modello: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler caricato: {scaler_path}")
        
        # Estrai shape
        if len(self.model.input_shape) == 3:
            self.sequence_length = self.model.input_shape[1]
            self.n_features = self.model.input_shape[2]
        
        logger.info("Modello caricato con successo!")

def train_model_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         sequence_length: int, n_features: int,
                         model_type: str = 'bidirectional',
                         epochs: int = 100,
                         batch_size: int = 32,
                         model_name: str = None,
                         data_loader = None,
                         problem_type: str = 'classification') -> Tuple[LSTMTradingModel, Dict]:
    """
    Pipeline completa di addestramento per sistema profittevole.
    """
    logger.info("=" * 60)
    logger.info(f"PIPELINE ADDESTRAMENTO - {problem_type.upper()}")
    logger.info("=" * 60)
    
    logger.info(f"🔢 Informazioni features:")
    logger.info(f"   Sequence length: {sequence_length}")
    logger.info(f"   Numero features: {n_features}")
    
    # Timestamp features detection
    timestamp_features_count = 0
    if n_features > 9:
        timestamp_features_count = n_features - 9
        logger.info(f"   📅 Timestamp features rilevate: {timestamp_features_count}")
    
    # Analisi dataset per trading profittevole
    if problem_type == 'classification':
        tp_count = np.sum(y_train == 1)
        sl_count = np.sum(y_train == 0)
        total_samples = len(y_train)
        
        logger.info(f"🎯 ANALISI DATASET PER TRADING PROFITTEVOLE:")
        logger.info(f"   TP (1): {tp_count} ({tp_count/total_samples:.1%})")
        logger.info(f"   SL (0): {sl_count} ({sl_count/total_samples:.1%})")
        logger.info(f"   R/R configurato: 4:1 (TP=80 pips, SL=20 pips)")
        
        # Calcola accuracy minima richiesta per profitto
        accuracy_required = 20 / (80 + 20)  # SL / (TP + SL)
        current_accuracy = tp_count / total_samples if total_samples > 0 else 0
        
        logger.info(f"   Accuracy richiesta: {accuracy_required:.1%}")
        logger.info(f"   Accuracy dataset: {current_accuracy:.1%}")
        logger.info(f"   Edge potenziale: {current_accuracy - accuracy_required:+.1%}p")
        
        # Calcola Profit Factor potenziale
        if sl_count > 0:
            expected_profit_per_trade = (tp_count * 80) - (sl_count * 20)
            profit_factor = (tp_count * 80) / (sl_count * 20) if sl_count > 0 else float('inf')
            logger.info(f"   Profit Factor atteso: {profit_factor:.2f}")
            logger.info(f"   Profitto atteso/trade: {expected_profit_per_trade/total_samples:.1f} pips")
        else:
            logger.info(f"   Profit Factor atteso: ∞ (nessun SL nel training)")
        
        if current_accuracy < accuracy_required:
            logger.warning(f"⚠️  Dataset non profittevole! Accuracy {current_accuracy:.1%} < {accuracy_required:.1%}")
            logger.warning("   Considera: 1. Più feature 2. Miglior preprocessing 3. Più dati")
        else:
            logger.info(f"✅ Dataset potenzialmente profittevole!")
    
    # Adatta epochs per più features
    if timestamp_features_count > 0:
        epochs = int(epochs * 1.2)
        logger.info(f"   🔧 Epochs adattati: {epochs}")
    
    # Inizializza trainer
    trainer = LSTMTradingModel(
        model_type=model_type, 
        model_name=model_name,
        data_loader=data_loader
    )
    
    # 🔥 CORREZIONE: Chiama build_model con tutti i parametri
    trainer.build_model(
        input_shape=(sequence_length, n_features),
        output_units=1,
        problem_type=problem_type
    )
    
    # Verifica che il modello sia stato creato
    if trainer.model is None:
        raise ValueError("Modello non costruito! build_model() non ha creato il modello.")
    
    # Addestra
    results = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Valutazione
    test_metrics = trainer.evaluate(X_test, y_test, prefix='final_test_')
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALUTAZIONE FINALE:")
    
    # Gestione correttamente i valori che potrebbero essere stringhe
    loss_val = test_metrics.get('final_test_loss')
    if loss_val is not None and isinstance(loss_val, (int, float)):
        logger.info(f"   Loss: {loss_val:.6f}")
    else:
        logger.info(f"   Loss: {loss_val}")
    
    if problem_type == 'classification':
        accuracy = test_metrics.get('final_test_accuracy')
        if accuracy is not None and isinstance(accuracy, (int, float)):
            logger.info(f"   Accuracy: {accuracy:.4f}")
            
            # Valutazione trading-specifica
            y_pred_proba = trainer.model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            if len(cm) == 2:
                tp = cm[1, 1]  # True Positives
                fp = cm[0, 1]  # False Positives
                tn = cm[0, 0]  # True Negatives
                fn = cm[1, 0]  # False Negatives
                
                win_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                loss_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                # Calcola metriche trading con R/R = 4:1
                expected_profit = (tp * 80) - (fp * 20)
                total_trades = len(y_test)
                avg_profit_per_trade = expected_profit / total_trades if total_trades > 0 else 0
                profit_factor = (tp * 80) / (fp * 20) if fp > 0 else float('inf')
                
                logger.info(f"🎯 METRICHE TRADING (R/R 4:1):")
                logger.info(f"   Win Rate: {win_rate:.2%}")
                logger.info(f"   Loss Rate: {loss_rate:.2%}")
                logger.info(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
                logger.info(f"   Profitto atteso: {expected_profit:.0f} pips")
                logger.info(f"   Profitto medio/trade: {avg_profit_per_trade:.1f} pips")
                logger.info(f"   Profit Factor: {profit_factor:.2f}")
                
                # Valuta se il sistema è profittevole
                if win_rate > 0.2:  # > 20% win rate per R/R 4:1
                    logger.info(f"✅ SISTEMA POTENZIALMENTE PROFITTEVOLE!")
                    logger.info(f"   Edge: {(win_rate - 0.2):.1%}")
                else:
                    logger.warning(f"⚠️  Sistema non profittevole!")
                    logger.warning(f"   Win rate {win_rate:.1%} < 20% richiesto")
        else:
            logger.info(f"   Accuracy: {accuracy}")
    else:
        mse = test_metrics.get('final_test_mse')
        r2 = test_metrics.get('final_test_r2')
        
        if mse is not None and isinstance(mse, (int, float)):
            logger.info(f"   MSE: {mse:.6f}")
        else:
            logger.info(f"   MSE: {mse}")
        
        if r2 is not None and isinstance(r2, (int, float)):
            logger.info(f"   R²: {r2:.4f}")
        else:
            logger.info(f"   R²: {r2}")
    
    # Salva modello
    try:
        # Aggiungi metriche trading al metadata
        metadata_extras = {}
        if problem_type == 'classification' and 'cm' in locals():
            metadata_extras['trading_metrics'] = {
                'win_rate': float(win_rate),
                'loss_rate': float(loss_rate),
                'expected_profit_pips': float(expected_profit),
                'avg_profit_per_trade': float(avg_profit_per_trade),
                'profit_factor': float(profit_factor) if profit_factor != float('inf') else 'inf',
                'tp_count': int(tp),
                'fp_count': int(fp),
                'tn_count': int(tn),
                'fn_count': int(fn)
            }
        
        trainer.save_model(
            metadata={
                'sequence_length': sequence_length,
                'n_features': n_features,
                'model_type': model_type,
                'problem_type': problem_type,
                'test_metrics': test_metrics,
                'timestamp_features_count': timestamp_features_count,
                'trading_rr_ratio': '4:1',
                'tp_pips': 80,
                'sl_pips': 20,
                'required_win_rate': 0.20,
                **metadata_extras
            }
        )
    except Exception as e:
        logger.error(f"Errore salvataggio modello: {e}")
        raise
    
    return trainer, {**results, 'test_metrics': test_metrics}

if __name__ == "__main__":
    # Test
    print("Test model_trainer.py")
    X = np.random.randn(100, 20, 9)
    y = np.random.randn(100, 1)
    
    trainer, _ = train_model_pipeline(
        X_train=X[:80], y_train=y[:80],
        X_test=X[80:], y_test=y[80:],
        sequence_length=20,
        n_features=9,
        epochs=5,
        model_name="test_model"
    )
    print("✅ Test completato")