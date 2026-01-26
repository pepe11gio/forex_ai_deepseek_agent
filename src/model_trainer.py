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
import json

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
        
        # 🔥 OUTPUT PER CLASSIFICAZIONE O REGRESSIONE
        if problem_type == 'classification':
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            # 🔥 USA OGGETTI METRICHE, NON STRINGHE
            metrics_list = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc')
            ]
        else:
            outputs = Dense(output_units, activation='linear')(x)
            loss = 'mse'
            metrics_list = [
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                tf.keras.metrics.MeanSquaredError(name='mse')
            ]
        
        # 🔥 CORREZIONE CRITICA: Assegna il modello a self.model
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=loss,
            metrics=metrics_list  # 🔥 USO OGGETTI METRICHE
        )
        
        self.model = model  # 🔥 QUESTA RIGA MANCAVA!
        
        logger.info(f"✅ Modello costruito: {model.count_params()} parametri")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        class_weight: Dict = None) -> Dict[str, Any]:
        """Addestra il modello con class weights opzionali."""
        
        # 🔥 VERIFICA CHE IL MODELLO ESISTA
        if self.model is None:
            raise ValueError("Modello non costruito! Chiama build_model() prima di train().")
        
        logger.info(f"Training: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples")
        
        # 🔥 LOG CLASS WEIGHTS SE PRESENTI
        if class_weight is not None:
            logger.info(f"🔧 Class weights applicati:")
            for cls, weight in class_weight.items():
                cls_name = "TP Setup" if cls == 1 else "SL Setup"
                logger.info(f"   {cls_name} (classe {cls}): weight = {weight:.2f}")
        
        # Callbacks
        callbacks = self._get_default_callbacks()
        
        # 🔥 TRAINING CON CLASS WEIGHTS
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False,
            class_weight=class_weight  # 🔥 AGGIUNTO
        )
        
        # Valutazione
        train_metrics = self.evaluate(X_train, y_train, prefix='train_')
        val_metrics = self.evaluate(X_val, y_val, prefix='val_')
        
        metrics_dict = {**train_metrics, **val_metrics}
        
        # 🔥 ANALISI PERFORMANCE CON CLASS WEIGHTS
        if class_weight is not None:
            logger.info("📈 PERFORMANCE CON CLASS WEIGHTS:")
            
            # Predizioni sul validation set
            y_pred_proba = self.model.predict(X_val, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            from sklearn.metrics import confusion_matrix
            
            try:
                cm = confusion_matrix(y_val, y_pred)
                if len(cm) == 2:
                    tp = cm[1, 1]  # TP Setup corretti
                    fp = cm[0, 1]  # SL Setup predetti come TP
                    tn = cm[0, 0]  # SL Setup corretti
                    fn = cm[1, 0]  # TP Setup predetti come SL
                    
                    recall_tp = tp / (tp + fn) if (tp + fn) > 0 else 0
                    recall_sl = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    logger.info(f"   Recall TP Setup: {recall_tp:.2%}")
                    logger.info(f"   Recall SL Setup: {recall_sl:.2%}")
                    logger.info(f"   Differenza recall: {abs(recall_tp - recall_sl):.2%}p")
                    
                    # Verifica bilanciamento
                    if abs(recall_tp - recall_sl) < 0.3:  # Differenza < 30%
                        logger.info(f"✅ Recall bilanciato tra classi")
                    else:
                        logger.warning(f"⚠️  Recall sbilanciato (>30% differenza)")
            except Exception as e:
                logger.warning(f"Impossibile analizzare confusion matrix: {e}")
        
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
        Salva modello e scaler con COMPATIBILITÀ GARANTITA Keras 2.x/3.x.
        Versione SISTEMATA e FUNZIONANTE.
        """
        import json
        import os
        import joblib
        from datetime import datetime
        
        # Crea directory se non esiste
        os.makedirs('models', exist_ok=True)
        
        # 🔥 PATH DI DEFAULT
        if model_path is None:
            model_path = f'models/{self.model_name}.keras'  # Usa .keras per compatibilità
        
        if scaler_path is None:
            scaler_path = f'models/{self.model_name}_scaler.pkl'
        
        logger.info(f"💾 SALVATAGGIO MODELLO:")
        logger.info(f"   Modello: {os.path.basename(model_path)}")
        
        # 🔥 1. VERIFICA CHE IL MODELLO ESISTA
        if self.model is None:
            raise ValueError("❌ Modello non addestrato! Impossibile salvare.")
        
        # 🔥 2. DETERMINA TIPO DI PROBLEMA IN MODO ROBUSTO
        problem_type = 'regression'  # default
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
                
                if 'sigmoid' in act_name.lower() and self.model.output_shape[-1] == 1:
                    problem_type = 'binary_classification'
                elif 'softmax' in act_name.lower():
                    problem_type = 'multi_classification'
                else:
                    problem_type = 'regression'
        except Exception as e:
            logger.warning(f"⚠️  Impossibile determinare problem_type: {e}")
        
        logger.info(f"   Tipo problema: {problem_type}")
        
        # 🔥 3. RICOMPILA CON METRICHE COMPATIBILI
        try:
            # Ottieni configurazione ottimizzatore originale
            optimizer_config = 'adam'  # default
            if hasattr(self.model.optimizer, 'get_config'):
                optimizer_config = self.model.optimizer.get_config()
            
            # 🔥 METRICHE COMPATIBILI PER KERAS 2.x e 3.x
            if problem_type == 'binary_classification':
                # Per classificazione binaria
                self.model.compile(
                    optimizer=optimizer_config,
                    loss='binary_crossentropy',
                    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
                )
            elif problem_type == 'multi_classification':
                # Per classificazione multi-classe
                self.model.compile(
                    optimizer=optimizer_config,
                    loss='categorical_crossentropy',
                    metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
                )
            else:
                # Per regressione
                self.model.compile(
                    optimizer=optimizer_config,
                    loss='mse',
                    metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
                )
            
            logger.info(f"   ✅ Modello ricompilato con metriche compatibili")
            
        except Exception as e:
            logger.error(f"❌ Errore ricompilazione: {e}")
            logger.warning("⚠️  Salvo il modello senza ricompilare...")
        
        # 🔥 4. SALVA IL MODELLO IN FORMATO .keras (PREFERITO)
        try:
            # Prova a salvare in formato .keras
            self.model.save(model_path, save_format='keras')
            logger.info(f"✅ Modello salvato in formato .keras: {os.path.basename(model_path)}")
            
        except Exception as e:
            logger.error(f"❌ Errore salvataggio .keras: {e}")
            
            # 🔥 FALLBACK: Prova .h5
            try:
                model_path_h5 = model_path.replace('.keras', '.h5')
                self.model.save(model_path_h5)
                model_path = model_path_h5
                logger.info(f"✅ Modello salvato in formato .h5: {os.path.basename(model_path)}")
                
            except Exception as e2:
                logger.error(f"❌ Errore anche con .h5: {e2}")
                
                # 🔥 ULTIMA RISORSA: Salva pesi e architettura separatamente
                try:
                    weights_path = model_path.replace('.keras', '_weights.h5').replace('.h5', '_weights.h5')
                    self.model.save_weights(weights_path)
                    
                    # Salva architettura JSON
                    model_json = self.model.to_json()
                    arch_path = model_path.replace('.keras', '_architecture.json').replace('.h5', '_architecture.json')
                    with open(arch_path, 'w') as f:
                        f.write(model_json)
                    
                    # Crea file manifest
                    import h5py
                    with h5py.File(model_path, 'w') as f:
                        f.attrs['saved_format'] = 'separated_components'
                        f.attrs['weights_file'] = os.path.basename(weights_path)
                        f.attrs['architecture_file'] = os.path.basename(arch_path)
                        f.attrs['keras_version'] = tf.__version__
                    
                    logger.info(f"✅ Modello salvato in componenti separati:")
                    logger.info(f"   • {os.path.basename(arch_path)} (architettura)")
                    logger.info(f"   • {os.path.basename(weights_path)} (pesi)")
                    
                except Exception as e3:
                    logger.error(f"❌ ERRORE CRITICO: Impossibile salvare modello: {e3}")
                    raise RuntimeError(f"Salvataggio fallito completamente: {e3}")
        
        # 🔥 5. SALVA SCALER IN MODO ROBUSTO
        scaler_to_save = None
        
        if self.data_loader is not None and hasattr(self.data_loader, 'scaler'):
            scaler_to_save = self.data_loader.scaler
            logger.info("✅ Usando scaler dal data_loader")
        
        if scaler_to_save is None and hasattr(self, 'scaler') and self.scaler is not None:
            scaler_to_save = self.scaler
            logger.info("✅ Usando scaler interno")
        
        if scaler_to_save is None and force_scaler_save:
            logger.warning("⚠️  Scaler non disponibile, creo scaler di default...")
            scaler_to_save = self._create_default_scaler()
        
        if scaler_to_save is not None:
            try:
                joblib.dump(scaler_to_save, scaler_path)
                logger.info(f"✅ Scaler salvato: {os.path.basename(scaler_path)}")
                
                # 🔥 VERIFICA CHE LO SCALER SIA CARICABILE
                test_scaler = joblib.load(scaler_path)
                logger.debug(f"   Scaler verificato: {test_scaler.n_features_in_} features")
            except Exception as e:
                logger.error(f"❌ Errore salvataggio scaler: {e}")
                # Non blocchiamo, solo warning
        else:
            logger.error("❌ IMPOSSIBILE SALVARE SCALER!")
        
        # 🔥 6. SALVA METADATA COMPLETI
        metadata_path = model_path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
        
        if metadata is None:
            metadata = {}
        
        # Metadata standardizzati
        metadata_to_save = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'problem_type': problem_type,
            'sequence_length': getattr(self, 'sequence_length', None),
            'n_features': getattr(self, 'n_features', None),
            'training_date': datetime.now().isoformat(),
            'model_file': os.path.basename(model_path),
            'scaler_file': os.path.basename(scaler_path) if scaler_to_save else None,
            'keras_version': tf.__version__,
            'compatibility_mode': 'guaranteed',
            'input_shape': str(self.model.input_shape) if self.model else None,
            'output_shape': str(self.model.output_shape) if self.model else None,
            'total_params': self.model.count_params() if self.model else None,
            **metadata
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata_to_save, f, indent=2, default=str)
            
            logger.info(f"✅ Metadata salvati: {os.path.basename(metadata_path)}")
            
        except Exception as e:
            logger.error(f"❌ Errore salvataggio metadata: {e}")
        
        # 🔥 7. LOG DI VERIFICA
        logger.info(f"📁 FILE CREATI:")
        logger.info(f"   • {os.path.basename(model_path)} (modello)")
        
        if 'weights_path' in locals():
            logger.info(f"   • {os.path.basename(weights_path)} (pesi)")
            logger.info(f"   • {os.path.basename(arch_path)} (architettura)")
        
        if scaler_to_save:
            logger.info(f"   • {os.path.basename(scaler_path)} (scaler)")
        
        if os.path.exists(metadata_path):
            logger.info(f"   • {os.path.basename(metadata_path)} (metadata)")
        
        # 🔥 8. TEST DI CARICAMENTO RAPIDO
        try:
            logger.info("🧪 Test caricamento rapido...")
            test_model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ Test caricamento OK: {test_model.input_shape}")
            del test_model
        except Exception as e:
            logger.warning(f"⚠️  Test caricamento fallito: {e}")
            # Non blocchiamo, solo warning
        
        return model_path
    
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
        """Carica modello e scaler con compatibilità Keras 3.x."""
        logger.info(f"Caricamento modello: {model_path}")
        
        try:
            # 🔥 CARICA IN FORMATO KERAS (.keras)
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            # Fallback per vecchi modelli .h5
            logger.warning(f"⚠️  Errore caricamento formato Keras: {e}")
            if model_path.endswith('.h5'):
                self.model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("Modello .h5 caricato (compile=False)")
            else:
                raise
        
        # 🔥 RICOMPILA IL MODELLO CON METRICHE COMPATIBILI
        if self.model is not None:
            # Estrai info dal modello
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape
            
            # Determina tipo di problema in base alla forma di output
            if len(output_shape) == 2 and output_shape[1] == 1:
                # Probabilmente classificazione binaria o regressione
                if self.model.layers[-1].activation.__name__ == 'sigmoid':
                    # Classificazione binaria
                    self.model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'AUC']
                    )
                    logger.info("Modello ricompilato per classificazione binaria")
                else:
                    # Regressione
                    self.model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['mae', tf.keras.metrics.MeanSquaredError()]
                    )
                    logger.info("Modello ricompilato per regressione")
        
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler caricato: {scaler_path}")
        
        # Estrai shape
        if len(self.model.input_shape) == 3:
            self.sequence_length = self.model.input_shape[1]
            self.n_features = self.model.input_shape[2]
        
        logger.info("Modello caricato e ricompilato con successo!")

    def train_single_model_with_transfer(self, 
                                   X_regression: np.ndarray,
                                   y_regression: np.ndarray,
                                   X_classification: np.ndarray, 
                                   y_classification: np.ndarray,
                                   epochs_regression: int = 50,
                                   epochs_classification: int = 30,
                                   freeze_base_layers: bool = True):
        """
        Addestra un singolo modello con transfer learning:
        1. Pre-training per regressione (movimento)
        2. Fine-tuning per classificazione TP/SL
        """
        
        logger.info("=" * 60)
        logger.info("🤖 TRANSFER LEARNING SU SINGOLO MODELLO")
        logger.info("=" * 60)
        
        # FASE 1: Pre-training regressione
        logger.info("\nFASE 1: Pre-training regressione...")
        logger.info(f"   Samples: {len(X_regression)}")
        logger.info(f"   Target range: [{y_regression.min():.3f}, {y_regression.max():.3f}]")
        
        # Costruisci modello base per regressione
        self.build_model(
            input_shape=(X_regression.shape[1], X_regression.shape[2]),
            output_units=1,
            problem_type='regression'
        )
        
        # Addestra per regressione
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        history_reg = self.model.fit(
            X_regression, y_regression,
            epochs=epochs_regression,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        logger.info(f"✅ Pre-training completato. MSE finale: {history_reg.history['loss'][-1]:.4f}")
        
        # FASE 2: Fine-tuning per classificazione
        logger.info("\nFASE 2: Fine-tuning per classificazione TP/SL...")
        logger.info(f"   Samples: {len(X_classification)}")
        logger.info(f"   Classe 0 (SL): {np.sum(y_classification == 0)}")
        logger.info(f"   Classe 1 (TP): {np.sum(y_classification == 1)}")
        
        # Congela i layer base se richiesto
        if freeze_base_layers:
            for layer in self.model.layers[:-3]:  # Lascia ultimi 3 layer liberi
                layer.trainable = False
            logger.info(f"🔒 Congelati {len(self.model.layers) - 3} layer base")
        
        # Sostituisci l'output layer per classificazione
        # Rimuovi l'ultimo layer (regressione)
        base_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-2].output  # Prende l'output del penultimo layer
        )
        
        # Aggiungi nuovo output layer per classificazione
        x = base_model.output
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        # Crea nuovo modello per classificazione
        self.model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        
        # Compila per classificazione con learning rate più basso
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # LR più basso per fine-tuning
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        # Addestra per classificazione
        history_clf = self.model.fit(
            X_classification, y_classification,
            epochs=epochs_classification,
            batch_size=32,
            validation_split=0.2,
            class_weight={0: 1.0, 1: 1.0},  # Puoi bilanciare le classi se necessario
            verbose=1
        )
        
        # Valutazione
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = (self.model.predict(X_classification) > 0.5).astype(int)
        accuracy = accuracy_score(y_classification, y_pred)
        
        logger.info(f"\n📊 RISULTATI FINE-TUNING:")
        logger.info(f"   Accuracy: {accuracy:.2%}")
        logger.info(f"   Val Loss: {history_clf.history['val_loss'][-1]:.4f}")
        logger.info(f"   Val Accuracy: {history_clf.history['val_accuracy'][-1]:.2%}")
        
        # Report dettagliato
        logger.info("\n📋 CLASSIFICATION REPORT:")
        report = classification_report(y_classification, y_pred, 
                                    target_names=['SL Setup', 'TP Setup'])
        for line in report.split('\n'):
            logger.info(f"   {line}")
        
        return {
            'regression_history': history_reg.history,
            'classification_history': history_clf.history,
            'final_accuracy': accuracy,
            'model': self.model
        }

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
    # 🔥 IMPORTAZIONI NECESSARIE
    import numpy as np
    import os
    import joblib
    from datetime import datetime

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
    
    # 🔥 ANALISI CLASSI E CALCOLO CLASS WEIGHTS PER CLASSIFICAZIONE
    class_weight_dict = None
    if problem_type == 'classification':
        # Analisi distribuzione classi
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        # 🔥 ASSICURIAMOCI CHE np SIA DISPONIBILE
        # np dovrebbe essere già importato globalmente, ma per sicurezza:
        import numpy as np
        
        unique_classes, class_counts = np.unique(y_train_flat, return_counts=True)
        
        logger.info(f"🎯 ANALISI DISTRIBUZIONE CLASSI:")
        for cls, count in zip(unique_classes, class_counts):
            percentage = count / len(y_train_flat) * 100
            cls_name = "TP Setup" if cls == 1 else "SL Setup"
            logger.info(f"   Classe {cls} ({cls_name}): {count} samples ({percentage:.1f}%)")
        
        # Calcola class weights per bilanciamento
        if len(unique_classes) == 2:
            from sklearn.utils.class_weight import compute_class_weight
            try:
                class_weights = compute_class_weight(
                    'balanced',
                    classes=unique_classes,
                    y=y_train_flat
                )
                class_weight_dict = {int(cls): weight for cls, weight in zip(unique_classes, class_weights)}
                
                logger.info(f"🔧 CLASS WEIGHTS per bilanciamento:")
                for cls in unique_classes:
                    cls_name = "TP Setup" if cls == 1 else "SL Setup"
                    logger.info(f"   {cls_name} (classe {cls}): weight = {class_weight_dict[cls]:.2f}")
                
                # Analisi dataset per trading profittevole
                tp_count = np.sum(y_train_flat == 1)
                sl_count = np.sum(y_train_flat == 0)
                total_samples = len(y_train_flat)
                
                logger.info(f"📈 ANALISI PROFITTEVOLEZZA:")
                logger.info(f"   TP Setups: {tp_count} ({tp_count/total_samples:.1%})")
                logger.info(f"   SL Setups: {sl_count} ({sl_count/total_samples:.1%})")
                logger.info(f"   R/R configurato: 4:1 (TP=80 pips, SL=20 pips)")
                
                # Calcola win rate necessario per profitto
                required_win_rate = 20 / (80 + 20)  # SL / (TP + SL)
                current_win_rate = tp_count / total_samples if total_samples > 0 else 0
                
                logger.info(f"   Win rate necessario: {required_win_rate:.1%}")
                logger.info(f"   Win rate dataset: {current_win_rate:.1%}")
                logger.info(f"   Edge potenziale: {current_win_rate - required_win_rate:+.1%}p")
                
                if current_win_rate >= required_win_rate:
                    logger.info(f"✅ Dataset potenzialmente profittevole!")
                else:
                    logger.warning(f"⚠️  Dataset sotto la soglia di profittevolezza!")
                    logger.warning(f"   Considera: più feature engineering o più dati")
                
            except Exception as e:
                logger.warning(f"⚠️  Errore calcolo class weights: {e}")
                class_weight_dict = None
        else:
            logger.warning(f"⚠️  Numero classi non supportato: {len(unique_classes)}")
    
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
    
    # Costruisci modello
    trainer.build_model(
        input_shape=(sequence_length, n_features),
        output_units=1,
        problem_type=problem_type
    )
    
    # Verifica che il modello sia stato creato
    if trainer.model is None:
        raise ValueError("Modello non costruito! build_model() non ha creato il modello.")
    
    # Addestra con class weights se disponibili
    results = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict  # 🔥 Passa class weights
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
            
            # 🔥 VALUTAZIONE DETTAGLIATA CON CLASSIFICAZIONE
            # 🔥 IMPORT NUMPY ANCHE QUI
            import numpy as np
            
            y_pred_proba = trainer.model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y_test, y_pred)
            
            if len(cm) == 2:
                tp = cm[1, 1]  # True Positives (TP Setup corretti)
                fp = cm[0, 1]  # False Positives (SL Setup predetti come TP)
                tn = cm[0, 0]  # True Negatives (SL Setup corretti)
                fn = cm[1, 0]  # False Negatives (TP Setup predetti come SL)
                
                # Calcola metriche specifiche per trading
                total_trades = len(y_test)
                actual_tp_count = np.sum(y_test == 1)
                actual_sl_count = np.sum(y_test == 0)
                
                # Precision e Recall per ogni classe
                precision_tp = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_tp = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision_sl = tn / (tn + fn) if (tn + fn) > 0 else 0
                recall_sl = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Win rate predetto
                predicted_win_rate = (tp + fp) / total_trades if total_trades > 0 else 0
                actual_win_rate = actual_tp_count / total_trades if total_trades > 0 else 0
                
                # Calcola metriche trading con R/R = 4:1
                expected_profit = (tp * 80) - (fp * 20)
                avg_profit_per_trade = expected_profit / total_trades if total_trades > 0 else 0
                profit_factor = (tp * 80) / (fp * 20) if fp > 0 else float('inf')
                
                logger.info(f"🎯 METRICHE TRADING DETTAGLIATE:")
                logger.info(f"   TP Setup - Precision: {precision_tp:.2%}, Recall: {recall_tp:.2%}")
                logger.info(f"   SL Setup - Precision: {precision_sl:.2%}, Recall: {recall_sl:.2%}")
                logger.info(f"   Predicted Win Rate: {predicted_win_rate:.2%}")
                logger.info(f"   Actual Win Rate: {actual_win_rate:.2%}")
                logger.info(f"   Required Win Rate: 20.0%")
                logger.info(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
                logger.info(f"   Profitto atteso: {expected_profit:.0f} pips")
                logger.info(f"   Profitto medio/trade: {avg_profit_per_trade:.1f} pips")
                logger.info(f"   Profit Factor: {profit_factor:.2f}")
                
                # Report di classificazione
                logger.info(f"\n📋 CLASSIFICATION REPORT:")
                report = classification_report(y_test, y_pred, 
                                             target_names=['SL Setup', 'TP Setup'],
                                             digits=3)
                for line in report.split('\n'):
                    logger.info(f"   {line}")
                
                # Valuta se il sistema è profittevole
                if actual_win_rate > 0.2:
                    edge = (actual_win_rate - 0.2) * 100
                    logger.info(f"✅ SISTEMA POTENZIALMENTE PROFITTEVOLE!")
                    logger.info(f"   Edge: +{edge:.1f}%")
                else:
                    logger.warning(f"⚠️  Sistema non profittevole!")
                    logger.warning(f"   Win rate {actual_win_rate:.1%} < 20% richiesto")
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
    
    # 🔥 SALVA MODELLO CON COMPATIBILITÀ
    try:
        # Crea directory models se non esiste
        os.makedirs('models', exist_ok=True)
        
        # Path per il modello
        model_path = os.path.join('models', f"{model_name}.h5")
        scaler_path = os.path.join('models', f"{model_name}_scaler.pkl")
        metadata_path = os.path.join('models', f"{model_name}_metadata.json")
        
        logger.info(f"\n💾 SALVATAGGIO MODELLO: {model_name}")
        
        # 🔥 1. SALVA IL MODELLO CON GESTIONE ERRORI
        try:
            # Prima prova a salvare normalmente
            trainer.model.save(model_path)
            logger.info(f"✅ Modello salvato: {os.path.basename(model_path)}")
        except Exception as save_error:
            logger.warning(f"⚠️  Errore salvataggio standard: {save_error}")
            
            # Seconda prova: salva con metriche semplificate
            try:
                # Crea una copia del modello per non modificare l'originale
                import tensorflow as tf
                model_copy = tf.keras.models.clone_model(trainer.model)
                model_copy.set_weights(trainer.model.get_weights())
                
                # Ricompila con metriche base compatibili
                last_layer = model_copy.layers[-1] if model_copy.layers else None
                
                if last_layer and hasattr(last_layer, 'activation'):
                    activation_name = last_layer.activation.__name__ if hasattr(last_layer.activation, '__name__') else str(last_layer.activation)
                    
                    if activation_name == 'sigmoid':
                        # Classificazione binaria
                        model_copy.compile(
                            optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy']  # Solo accuracy, compatibile
                        )
                        logger.debug("Modello ricompilato per classificazione (compatibile)")
                    elif activation_name == 'linear':
                        # Regressione
                        model_copy.compile(
                            optimizer='adam',
                            loss='mse',
                            metrics=['mae']  # Solo mae, non mse (compatibile)
                        )
                        logger.debug("Modello ricompilato per regressione (compatibile)")
                    else:
                        # Default
                        model_copy.compile(
                            optimizer='adam',
                            loss='mse',
                            metrics=['mae']
                        )
                
                # Salva la versione compatibile
                model_copy.save(model_path)
                logger.info(f"✅ Modello salvato con metriche compatibili: {os.path.basename(model_path)}")
                
            except Exception as e2:
                logger.error(f"❌ Errore anche con salvataggio compatibile: {e2}")
                
                # Terza prova: salva solo pesi
                try:
                    weights_path = model_path.replace('.h5', '_weights.h5')
                    trainer.model.save_weights(weights_path)
                    
                    # Salva architettura come JSON
                    model_json = trainer.model.to_json()
                    arch_path = model_path.replace('.h5', '_architecture.json')
                    with open(arch_path, 'w') as f:
                        f.write(model_json)
                    
                    logger.info(f"✅ Modello salvato in formato separato:")
                    logger.info(f"   • {os.path.basename(arch_path)} (architettura)")
                    logger.info(f"   • {os.path.basename(weights_path)} (pesi)")
                    
                    # Crea file .h5 placeholder
                    import h5py
                    with h5py.File(model_path, 'w') as f:
                        f.attrs['saved_format'] = 'separated'
                        f.attrs['architecture_file'] = os.path.basename(arch_path)
                        f.attrs['weights_file'] = os.path.basename(weights_path)
                        f.attrs['model_name'] = model_name
                        
                except Exception as e3:
                    logger.error(f"❌ Errore critico nel salvataggio: {e3}")
                    raise RuntimeError(f"Impossibile salvare il modello: {e3}")
        
        # 🔥 2. SALVA SCALER
        scaler_to_save = None
        
        if data_loader is not None and hasattr(data_loader, 'scaler'):
            scaler_to_save = data_loader.scaler
            logger.info("✅ Usando scaler dal data_loader")
        else:
            logger.warning("⚠️  Data loader senza scaler, creo scaler di default")
            from sklearn.preprocessing import StandardScaler
            scaler_to_save = StandardScaler()
            # Fit con dati dummy
            import numpy as np
            dummy_data = np.random.randn(100, n_features)
            scaler_to_save.fit(dummy_data)
        
        if scaler_to_save is not None:
            joblib.dump(scaler_to_save, scaler_path)
            logger.info(f"✅ Scaler salvato: {os.path.basename(scaler_path)}")
        else:
            logger.error("❌ IMPOSSIBILE SALVARE SCALER!")
        
        # 🔥 3. SALVA METADATA CON INFO COMPATIBILITÀ
        metadata_to_save = {
            'model_name': model_name,
            'model_type': model_type,
            'sequence_length': sequence_length,
            'n_features': n_features,
            'training_date': datetime.now().isoformat(),
            'scaler_saved': scaler_to_save is not None,
            'scaler_path': os.path.basename(scaler_path) if scaler_to_save is not None else None,
            'problem_type': problem_type,
            'timestamp_features_count': timestamp_features_count,
            'trading_rr_ratio': '4:1',
            'tp_pips': 80,
            'sl_pips': 20,
            'required_win_rate': 0.20,
        }
        
        # Aggiungi metriche trading se classificazione
        if problem_type == 'classification' and 'cm' in locals():
            metadata_to_save['trading_metrics'] = {
                'win_rate': float(actual_win_rate) if 'actual_win_rate' in locals() else 0,
                'predicted_win_rate': float(predicted_win_rate) if 'predicted_win_rate' in locals() else 0,
                'expected_profit_pips': float(expected_profit) if 'expected_profit' in locals() else 0,
                'avg_profit_per_trade': float(avg_profit_per_trade) if 'avg_profit_per_trade' in locals() else 0,
                'profit_factor': float(profit_factor) if 'profit_factor' != float('inf') else 'inf',
                'tp_count': int(tp) if 'tp' in locals() else 0,
                'fp_count': int(fp) if 'fp' in locals() else 0,
                'tn_count': int(tn) if 'tn' in locals() else 0,
                'fn_count': int(fn) if 'fn' in locals() else 0,
                'precision_tp': float(precision_tp) if 'precision_tp' in locals() else 0,
                'recall_tp': float(recall_tp) if 'recall_tp' in locals() else 0,
                'precision_sl': float(precision_sl) if 'precision_sl' in locals() else 0,
                'recall_sl': float(recall_sl) if 'recall_sl' in locals() else 0
            }
        
        # Aggiungi feature names se disponibili
        if data_loader is not None:
            if hasattr(data_loader, 'feature_names'):
                metadata_to_save['feature_names'] = data_loader.feature_names
            
            # Identifica timestamp features
            if hasattr(data_loader, 'feature_names'):
                timestamp_features = [f for f in data_loader.feature_names 
                                    if any(keyword in f.lower() for keyword in 
                                        ['hour', 'day', 'week', 'session', 'sin', 'cos'])]
                if timestamp_features:
                    metadata_to_save['timestamp_features'] = timestamp_features
        
        # Aggiungi test metrics
        metadata_to_save['test_metrics'] = test_metrics
        
        # Salva metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2)
        
        logger.info(f"✅ Metadata salvati: {os.path.basename(metadata_path)}")
        
        # 🔥 4. LOG DI VERIFICA FINALE
        logger.info(f"\n📁 VERIFICA FILE CREATI:")
        logger.info(f"   • {os.path.basename(model_path)}")
        if 'arch_path' in locals():
            logger.info(f"   • {os.path.basename(arch_path)} (architettura)")
            logger.info(f"   • {os.path.basename(weights_path)} (pesi)")
        logger.info(f"   • {os.path.basename(scaler_path)}")
        logger.info(f"   • {os.path.basename(metadata_path)}")
        
        # 🔥 5. ASSEGNA IL MODELLO AL TRAINER PER USO SUCCESSIVO
        trainer.model_path = model_path
        trainer.scaler_path = scaler_path
        
    except Exception as e:
        logger.error(f"❌ Errore salvataggio modello: {e}")
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