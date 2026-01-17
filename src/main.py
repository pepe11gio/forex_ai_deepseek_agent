"""
main.py
Orchestratore principale del sistema di trading AI.
Versione COMPLETA con self-learning training per classificazione TP/SL.
AGGIORNATA: Supporta fine-tuning di modelli esistenti.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Import moduli del sistema
from data_loader import FinancialDataLoader, load_and_prepare_data
from model_trainer import LSTMTradingModel, train_model_pipeline
from pattern_analyzer import PatternAnalyzer, analyze_trading_model
from predictor import TradingPredictor, create_predictor_from_training
from deepseek_chat import DeepSeekChat, create_trading_chatbot
from trading_executor import TradingExecutor

# Configurazione
from config import CONFIG, create_directories

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'trading_system_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingAIOrchestrator:
    """Orchestratore principale con self-learning training e fine-tuning."""
    
    def __init__(self):
        """Inizializza l'orchestratore."""
        self.config = CONFIG
        self.data_loader = None
        self.predictor = None
        self.chatbot = None
        
        # Stato sistema
        self.system_state = {
            "initialized": False,
            "data_loaded": False,
            "model_trained": False,
            "predictor_ready": False,
            "chatbot_ready": False,
            "last_update": datetime.now().isoformat()
        }
        
        # Cache risultati
        self.results_cache = {}
        
        # Crea directory
        create_directories()
        
        logger.info("Trading AI Orchestrator inizializzato")
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        Carica TUTTI i file CSV dalla directory data/.
        """
        logger.info("=" * 60)
        logger.info("CARICAMENTO TUTTI I FILE DA DATA/")
        logger.info("=" * 60)
        
        try:
            data_dir = self.config["paths"]["data"]
            
            # Trova TUTTI i file CSV
            import glob
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            
            if not csv_files:
                raise FileNotFoundError(f"Nessun file CSV trovato in {data_dir}")
            
            logger.info(f"Trovati {len(csv_files)} file CSV")
            
            # Legge e combina TUTTI i file
            all_dataframes = []
            total_rows = 0
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    rows = len(df)
                    total_rows += rows
                    all_dataframes.append(df)
                    logger.info(f"  ✓ {os.path.basename(csv_file)}: {rows} righe")
                except Exception as e:
                    logger.error(f"  ✗ Errore {os.path.basename(csv_file)}: {e}")
                    continue
            
            if not all_dataframes:
                raise ValueError("Nessun file CSV caricato con successo")
            
            # Combina tutti i DataFrame
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Ordina per timestamp se presente
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp')
            
            logger.info(f"✅ Dati combinati: {total_rows} righe totali")
            
            # Salva temporaneamente
            import tempfile
            temp_file = os.path.join(tempfile.gettempdir(), 
                                   f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            combined_df.to_csv(temp_file, index=False)
            
            # Configura data loader
            data_config = self.config["data"]
            self.data_loader = FinancialDataLoader(
                sequence_length=data_config["sequence_length"],
                test_size=data_config["test_size"],
                random_state=42
            )
            
            # Processa dati combinati
            results = load_and_prepare_data(
                filepath=temp_file,
                sequence_length=data_config["sequence_length"],
                test_size=data_config["test_size"],
                lookahead=10
            )
            
            X_train, X_test, y_train, y_test, df_original, df_normalized, loader = results
            
            # Aggiorna stato
            self.system_state["data_loaded"] = True
            self.system_state["data_info"] = {
                "total_files": len(csv_files),
                "total_samples": total_rows,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "sequence_length": data_config["sequence_length"],
                "n_features": X_train.shape[2],
                "feature_names": loader.feature_names,
                "loaded_at": datetime.now().isoformat()
            }
            
            # Salva in cache
            self.results_cache["data"] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "loader": loader,
                "df_original": df_original,
                "df_normalized": df_normalized
            }
            
            # Pulisci file temporaneo
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            logger.info(f"✅ Dati caricati: {len(X_train)} train, {len(X_test)} test sequenze")
            
            return {
                "success": True,
                "data_info": self.system_state["data_info"]
            }
            
        except Exception as e:
            logger.error(f"Errore nel caricamento dati: {str(e)}")
            self.system_state["data_loaded"] = False
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Pipeline di training base per regressione.
        """
        logger.info("=" * 60)
        logger.info("TRAINING MODELLO REGRESSIONE")
        logger.info("=" * 60)
        
        try:
            # 1. Carica TUTTI i dati
            data_result = self.load_all_data()
            if not data_result["success"]:
                raise ValueError(f"Errore caricamento dati: {data_result.get('error')}")
            
            # 2. Recupera dati
            data = self.results_cache["data"]
            X_train, X_test = data["X_train"], data["X_test"]
            y_train, y_test = data["y_train"], data["y_test"]
            loader = data["loader"]
            
            # 3. Configura training
            model_config = self.config["model"]
            model_name = f"regression_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 4. Addestra modello regressione
            logger.info("Addestramento regressione in corso...")
            trainer, training_results = train_model_pipeline(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                sequence_length=self.config["data"]["sequence_length"],
                n_features=X_train.shape[2],
                model_type=model_config["type"],
                epochs=model_config["epochs"],
                batch_size=model_config["batch_size"],
                model_name=model_name,
                data_loader=loader,
                problem_type='regression'  # Regressione classica
            )
            
            # 5. Aggiorna stato
            self.system_state["model_trained"] = True
            self.system_state["model_info"] = {
                "model_name": model_name,
                "model_type": model_config["type"],
                "problem_type": "regression",
                "sequence_length": self.config["data"]["sequence_length"],
                "n_features": X_train.shape[2],
                "training_date": datetime.now().isoformat(),
                "performance": training_results.get("test_metrics", {})
            }
            
            # 6. Salva in cache
            self.results_cache["model"] = {
                "trainer": trainer,
                "training_results": training_results,
                "model_info": self.system_state["model_info"]
            }
            
            logger.info(f"✅ Modello regressione addestrato: {model_name}")
            
            return {
                "success": True,
                "model_info": self.system_state["model_info"],
                "performance": training_results.get("test_metrics", {})
            }
            
        except Exception as e:
            logger.error(f"Errore nel training regressione: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_self_learning_training(self, fine_tune_existing: bool = True) -> Dict[str, Any]:
        """
        Training evoluto CON FINE-TUNING se modello esistente disponibile.
        Se fine_tune_existing=True e trova un modello TP/SL, lo migliora.
        Altrimenti crea nuovo modello da zero.
        """
        logger.info("=" * 60)
        logger.info("🧠 TRAINING EVOLUTO - CLASSIFICAZIONE TP/SL")
        logger.info("=" * 60)
        
        # CERCA MODELLO ESISTENTE PER FINE-TUNING
        existing_model_path = None
        if fine_tune_existing:
            existing_model_path = self._find_latest_tp_sl_model()
            
            if existing_model_path:
                logger.info(f"🎯 TROVATO MODELLO ESISTENTE: {os.path.basename(existing_model_path)}")
                logger.info("🔧 Modalità: FINE-TUNING (miglioramento modello esistente)")
                return self._fine_tune_existing_model(existing_model_path)
            else:
                logger.info("🆕 Nessun modello TP/SL trovato")
                logger.info("🚀 Modalità: TRAINING DA ZERO")
        else:
            logger.info("🔄 Modalità: TRAINING DA ZERO (richiesto)")
        
        # SE NON TROVA MODELLO O FINE_TUNE_EXISTING=False, ALLENA DA ZERO
        return self._train_tp_sl_from_scratch()
    
    def _find_latest_tp_sl_model(self) -> Optional[str]:
        """
        Cerca l'ultimo modello TP/SL classifier nella directory models/.
        Ritorna il path al modello più recente, o None se non trovato.
        """
        import glob
        
        models_dir = self.config["paths"]["models"]
        
        # Cerca modelli TP/SL (nome contiene 'tp_sl_classifier')
        tp_sl_pattern = os.path.join(models_dir, "tp_sl_classifier_*.h5")
        tp_sl_models = glob.glob(tp_sl_pattern)
        
        if not tp_sl_models:
            # Cerca anche modelli migliorati
            improved_pattern = os.path.join(models_dir, "improved_tp_sl_classifier_*.h5")
            tp_sl_models = glob.glob(improved_pattern)
        
        if not tp_sl_models:
            logger.info("⚠️  Nessun modello TP/SL trovato in models/")
            return None
        
        # Ordina per data di modifica (più recente primo)
        tp_sl_models.sort(key=os.path.getmtime, reverse=True)
        latest_model = tp_sl_models[0]
        
        # Verifica che esistano anche scaler e metadata
        model_base = latest_model.replace('.h5', '')
        scaler_file = f"{model_base}_scaler.pkl"
        metadata_file = f"{model_base}_metadata.json"
        
        if not os.path.exists(scaler_file) or not os.path.exists(metadata_file):
            logger.warning(f"⚠️  Modello {os.path.basename(latest_model)} incompleto (mancano file)")
            return None
        
        logger.info(f"✅ Modello trovato: {os.path.basename(latest_model)}")
        return latest_model
    
    def _fine_tune_existing_model(self, model_path: str) -> Dict[str, Any]:
        """
        Carica modello TP/SL esistente e fa fine-tuning con dati recenti.
        """
        logger.info("=" * 60)
        logger.info("🎯 FINE-TUNING MODELLO TP/SL ESISTENTE")
        logger.info("=" * 60)
        
        try:
            # 🔥 PASSO 1: CARICA DATI RECENTI (ultimi 6 mesi)
            logger.info("1. Caricamento dati recenti per fine-tuning...")
            
            data_dir = self.config["paths"]["data"]
            sequence_length = self.config["data"]["sequence_length"]
            
            # Carica TUTTI i dati per estrarre i più recenti
            from data_loader import load_and_prepare_data
            results = load_and_prepare_data(
                filepath=data_dir,
                sequence_length=sequence_length,
                test_size=0.3,  # Più dati per test durante fine-tuning
                lookahead=10
            )
            
            X_train, X_test, y_train, y_test, df_original, df_normalized, loader = results
            
            # Salva nella cache
            self.results_cache["data"] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "loader": loader,
                "df_original": df_original,
                "df_normalized": df_normalized
            }
            
            # 🔥 PASSO 2: PREPARA DATI TP/SL CON DATI RECENTI
            logger.info("\n2. Preparazione dati TP/SL da dati recenti...")
            
            price_data = df_original['price'].values
            feature_columns = loader.feature_names if hasattr(loader, 'feature_names') else loader.feature_columns
            features_data = df_normalized[feature_columns].values
            
            # Crea sequenze TP/SL
            X_tp_sl, y_tp_sl = self._create_tp_sl_sequences_from_data(
                features_sequences=self._create_all_sequences(features_data, sequence_length),
                price_data=price_data,
                sequence_length=sequence_length,
                tp_pips=30,
                sl_pips=20,
                start_idx=0
            )
            
            if len(X_tp_sl) == 0:
                raise ValueError("Nessuna sequenza TP/SL valida creata")
            
            # Split train/test (80/20)
            split_idx = int(len(X_tp_sl) * 0.8)
            X_train_tp = X_tp_sl[:split_idx]
            X_test_tp = X_tp_sl[split_idx:]
            y_train_tp = y_tp_sl[:split_idx]
            y_test_tp = y_tp_sl[split_idx:]
            
            logger.info(f"📊 DATI PER FINE-TUNING:")
            logger.info(f"   Train: {len(X_train_tp)} samples")
            logger.info(f"   Test: {len(X_test_tp)} samples")
            logger.info(f"   TP ratio: {np.sum(y_tp_sl == 1) / len(y_tp_sl):.2%}")
            
            # 🔥 PASSO 3: CARICA MODELLO ESISTENTE
            logger.info("\n3. Caricamento modello esistente per fine-tuning...")
            
            model_base = model_path.replace('.h5', '')
            scaler_path = f"{model_base}_scaler.pkl"
            metadata_path = f"{model_base}_metadata.json"
            
            # Carica metadata per ottenere info sul modello
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Inizializza trainer con modello esistente
            trainer = LSTMTradingModel(
                model_type=metadata.get('model_type', 'bidirectional'),
                model_name=f"fine_tuned_{os.path.basename(model_base)}",
                data_loader=loader
            )
            
            # Carica il modello esistente
            trainer.load_model(model_path, scaler_path)
            
            logger.info(f"✅ Modello caricato: {os.path.basename(model_path)}")
            logger.info(f"   Accuracy originale: {metadata.get('performance', {}).get('accuracy', 'N/A')}")
            
            # 🔥 PASSO 4: FINE-TUNING CON LEARNING RATE BASSO
            logger.info("\n4. Fine-tuning in corso (learning rate basso)...")
            
            # Compila con learning rate più basso per fine-tuning
            from tensorflow.keras.optimizers import Adam
            trainer.model.compile(
                optimizer=Adam(learning_rate=0.0001),  # LR molto basso per fine-tuning
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC']
            )
            
            # Callbacks per fine-tuning
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
            fine_tune_callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Fine-tuning con pochi epochs
            history = trainer.model.fit(
                X_train_tp, y_train_tp,
                epochs=50,  # Meno epochs per fine-tuning
                batch_size=32,
                validation_data=(X_test_tp, y_test_tp),
                callbacks=fine_tune_callbacks,
                verbose=1,
                shuffle=True
            )
            
            # 🔥 PASSO 5: VALUTAZIONE DOPO FINE-TUNING
            logger.info("\n5. Valutazione dopo fine-tuning...")
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            y_pred_proba = trainer.model.predict(X_test_tp, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test_tp, y_pred)
            precision = precision_score(y_test_tp, y_pred, zero_division=0)
            recall = recall_score(y_test_tp, y_pred, zero_division=0)
            f1 = f1_score(y_test_tp, y_pred, zero_division=0)
            
            logger.info(f"📈 PERFORMANCE DOPO FINE-TUNING:")
            logger.info(f"   Accuracy: {accuracy:.2%}")
            logger.info(f"   Precision: {precision:.2%}")
            logger.info(f"   Recall: {recall:.2%}")
            logger.info(f"   F1-Score: {f1:.2%}")
            
            # 🔥 PASSO 6: SALVA MODELLO MIGLIORATO
            logger.info("\n6. Salvataggio modello fine-tuned...")
            
            # Nome nuovo per modello fine-tuned
            fine_tuned_name = f"tp_sl_classifier_fine_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trainer.model_name = fine_tuned_name
            
            trainer.save_model(
                metadata={
                    'original_model': os.path.basename(model_path),
                    'fine_tuning_date': datetime.now().isoformat(),
                    'fine_tuning_samples': len(X_train_tp),
                    'performance': {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1)
                    },
                    'training_type': 'fine_tuning',
                    **metadata  # Mantieni metadata originale
                }
            )
            
            # Aggiorna predictor
            model_full_path = os.path.join(self.config["paths"]["models"], f"{fine_tuned_name}.h5")
            scaler_full_path = os.path.join(self.config["paths"]["models"], f"{fine_tuned_name}_scaler.pkl")
            
            self.predictor = TradingPredictor(
                model_path=model_full_path,
                scaler_path=scaler_full_path,
                sequence_length=sequence_length,
                confidence_level=0.7
            )
            
            self.system_state["predictor_ready"] = True
            
            logger.info(f"✅ FINE-TUNING COMPLETATO!")
            logger.info(f"   Modello originale: {os.path.basename(model_path)}")
            logger.info(f"   Modello fine-tuned: {fine_tuned_name}")
            logger.info(f"   Accuracy: {accuracy:.2%}")
            
            return {
                'success': True,
                'training_type': 'fine_tuning',
                'original_model': os.path.basename(model_path),
                'fine_tuned_model': fine_tuned_name,
                'accuracy': accuracy,
                'samples_used': len(X_train_tp),
                'message': f'Fine-tuning completato. Accuracy: {accuracy:.2%}'
            }
            
        except Exception as e:
            logger.error(f"❌ Errore nel fine-tuning: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'training_type': 'fine_tuning',
                'error': str(e)
            }
    
    def _train_tp_sl_from_scratch(self) -> Dict[str, Any]:
        """
        Crea nuovo modello TP/SL da zero.
        """
        logger.info("=" * 60)
        logger.info("🚀 TRAINING DA ZERO - NUOVO MODELLO TP/SL")
        logger.info("=" * 60)
        
        try:
            # 🔥 PASSO 1: CARICA E PREPARA DATI
            logger.info("1. Caricamento e preparazione dati...")
            
            data_dir = self.config["paths"]["data"]
            sequence_length = self.config["data"]["sequence_length"]
            test_size = self.config["data"]["test_size"]
            
            from data_loader import load_and_prepare_data
            
            results = load_and_prepare_data(
                filepath=data_dir,
                sequence_length=sequence_length,
                test_size=test_size,
                lookahead=10
            )
            
            X_train, X_test, y_train, y_test, df_original, df_normalized, loader = results
            
            # Salva nella cache
            self.results_cache["data"] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "loader": loader,
                "df_original": df_original,
                "df_normalized": df_normalized
            }
            
            # 🔥 PASSO 2: PREPARA DATI TP/SL
            logger.info("\n2. Preparazione dati TP/SL...")
            
            price_data = df_original['price'].values
            feature_columns = loader.feature_names if hasattr(loader, 'feature_names') else loader.feature_columns
            features_data = df_normalized[feature_columns].values
            
            # Crea tutte le sequenze
            X_all_sequences = self._create_all_sequences(features_data, sequence_length)
            
            # Crea sequenze TP/SL
            X_tp_sl, y_tp_sl = self._create_tp_sl_sequences_from_data(
                features_sequences=X_all_sequences,
                price_data=price_data,
                sequence_length=sequence_length,
                tp_pips=30,
                sl_pips=20,
                start_idx=0
            )
            
            if len(X_tp_sl) == 0:
                raise ValueError("Nessuna sequenza TP/SL valida creata")
            
            # Split train/test
            train_ratio = len(X_train) / (len(X_train) + len(X_test))
            split_idx = int(len(X_tp_sl) * train_ratio)
            
            X_train_tp = X_tp_sl[:split_idx]
            X_test_tp = X_tp_sl[split_idx:]
            y_train_tp = y_tp_sl[:split_idx]
            y_test_tp = y_tp_sl[split_idx:]
            
            logger.info(f"📊 DATI PER TRAINING DA ZERO:")
            logger.info(f"   Train: {len(X_train_tp)} samples ({train_ratio:.1%})")
            logger.info(f"   Test: {len(X_test_tp)} samples ({(1-train_ratio):.1%})")
            logger.info(f"   TP ratio: {np.sum(y_tp_sl == 1) / len(y_tp_sl):.2%}")
            
            # 🔥 PASSO 3: TRAINING CLASSIFICAZIONE DA ZERO
            logger.info("\n3. Training modello classificazione da zero...")
            
            model_config = self.config["model"]
            model_name = f"tp_sl_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            trainer, training_results = train_model_pipeline(
                X_train=X_train_tp,
                y_train=y_train_tp,
                X_test=X_test_tp,
                y_test=y_test_tp,
                sequence_length=sequence_length,
                n_features=X_train_tp.shape[2],
                model_type=model_config["type"],
                epochs=150,
                batch_size=32,
                model_name=model_name,
                data_loader=loader,
                problem_type='classification'
            )
            
            # 🔥 PASSO 4: VALUTAZIONE DETTAGLIATA
            logger.info("\n4. Valutazione dettagliata del modello...")
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            y_pred_proba = trainer.model.predict(X_test_tp, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test_tp, y_pred)
            precision = precision_score(y_test_tp, y_pred, zero_division=0)
            recall = recall_score(y_test_tp, y_pred, zero_division=0)
            f1 = f1_score(y_test_tp, y_pred, zero_division=0)
            cm = confusion_matrix(y_test_tp, y_pred)
            
            logger.info(f"📊 PERFORMANCE CLASSIFICAZIONE:")
            logger.info(f"   Accuracy: {accuracy:.2%}")
            logger.info(f"   Precision: {precision:.2%}")
            logger.info(f"   Recall: {recall:.2%}")
            logger.info(f"   F1-Score: {f1:.2%}")
            
            # 🔥 PASSO 5: SALVATAGGIO REPORT
            logger.info("\n5. Salvataggio modello e report...")
            
            # Salva metadata
            trainer.save_model(
                metadata={
                    'sequence_length': sequence_length,
                    'n_features': X_train_tp.shape[2],
                    'model_type': model_config["type"],
                    'problem_type': 'binary_classification_tp_sl',
                    'performance': {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1)
                    },
                    'dataset_stats': {
                        'total_samples': len(X_tp_sl),
                        'train_samples': len(X_train_tp),
                        'test_samples': len(X_test_tp),
                        'tp_count': int(np.sum(y_tp_sl == 1)),
                        'sl_count': int(np.sum(y_tp_sl == 0)),
                        'tp_ratio': float(np.sum(y_tp_sl == 1) / len(y_tp_sl))
                    },
                    'training_type': 'from_scratch'
                }
            )
            
            # Aggiorna predictor
            model_full_path = os.path.join(self.config["paths"]["models"], f"{model_name}.h5")
            scaler_full_path = os.path.join(self.config["paths"]["models"], f"{model_name}_scaler.pkl")
            
            self.predictor = TradingPredictor(
                model_path=model_full_path,
                scaler_path=scaler_full_path,
                sequence_length=sequence_length,
                confidence_level=0.7
            )
            
            self.system_state["predictor_ready"] = True
            
            logger.info(f"✅ TRAINING DA ZERO COMPLETATO!")
            logger.info(f"   Modello: {model_name}")
            logger.info(f"   Accuracy: {accuracy:.2%}")
            
            return {
                'success': True,
                'training_type': 'from_scratch',
                'model_name': model_name,
                'accuracy': accuracy,
                'samples_used': len(X_tp_sl),
                'message': f'Training da zero completato. Accuracy: {accuracy:.2%}'
            }
            
        except Exception as e:
            logger.error(f"❌ Errore nel training da zero: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'training_type': 'from_scratch',
                'error': str(e)
            }
    
    def _create_all_sequences(self, features_data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Crea tutte le sequenze possibili dai dati features.
        """
        X_all_sequences = []
        n_total_sequences = len(features_data) - sequence_length + 1
        
        for i in range(n_total_sequences):
            sequence = features_data[i:i + sequence_length]
            X_all_sequences.append(sequence)
        
        return np.array(X_all_sequences)
    
    def _create_tp_sl_sequences_from_data(self, features_sequences: np.ndarray,
                                 price_data: np.ndarray,
                                 sequence_length: int,
                                 tp_pips: int = 80,    # MODIFICATO: 80 pips
                                 sl_pips: int = 20,    # MODIFICATO: 20 pips
                                 start_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea sequenze TP/SL dai dati REALI con R/R = 4:1.
        """
        X, y = [], []
        
        pip_value = 0.0001
        max_pips = max(tp_pips, sl_pips)
        buffer_ticks = max_pips * 15  # Buffer generoso per H4 timeframe
        
        logger.info(f"Creazione sequenze TP/SL da dati reali con R/R = 4:1...")
        logger.info(f"   TP: {tp_pips} pips, SL: {sl_pips} pips")
        logger.info(f"   R/R ratio: {tp_pips/sl_pips:.1f}:1")
        logger.info(f"   Buffer ticks: {buffer_ticks}")
        
        tp_count = 0
        sl_count = 0
        skipped = 0
        
        for i in range(len(features_sequences)):
            # Calcola l'indice corrispondente nel price_data
            price_start_idx = start_idx + i
            price_end_idx = price_start_idx + sequence_length
            
            # Verifica che abbiamo abbastanza dati futuri
            if price_end_idx + buffer_ticks >= len(price_data):
                skipped += 1
                continue
            
            # Features sequence
            sequence = features_sequences[i]
            
            # Current price: ultimo prezzo della sequenza
            current_price = price_data[price_end_idx - 1]
            
            # Future prices (dopo la fine della sequenza)
            future_start_idx = price_end_idx
            future_end_idx = future_start_idx + buffer_ticks
            future_prices = price_data[future_start_idx:future_end_idx]
            
            # Verifica che ci siano dati futuri
            if len(future_prices) == 0:
                skipped += 1
                continue
            
            tp_reached = False
            sl_reached = False
            
            # Check TP e SL con nuovo R/R
            for future_price in future_prices:
                # TP check (assume sempre long per training)
                if future_price >= current_price + (tp_pips * pip_value):
                    tp_reached = True
                    break
                
                # SL check
                if future_price <= current_price - (sl_pips * pip_value):
                    sl_reached = True
                    break
            
            # Target binario
            if tp_reached:
                X.append(sequence)
                y.append(1)
                tp_count += 1
            elif sl_reached:
                X.append(sequence)
                y.append(0)
                sl_count += 1
            else:
                # Se TP/SL non raggiunto nel buffer, skip
                skipped += 1
        
        # Calcola ratio TP/SL e accuracy minima richiesta
        total_trades = tp_count + sl_count
        if total_trades > 0:
            tp_ratio = tp_count / total_trades
            accuracy_required = sl_pips / (tp_pips + sl_pips)  # Per breakeven
            logger.info(f"✅ Sequenze TP/SL create (R/R = 4:1):")
            logger.info(f"   Totali: {total_trades}")
            logger.info(f"   TP: {tp_count} ({tp_ratio:.1%})")
            logger.info(f"   SL: {sl_count} ({(1-tp_ratio):.1%})")
            logger.info(f"   Accuracy richiesta per profitto: {accuracy_required:.1%}")
            logger.info(f"   Current accuracy: {tp_ratio:.1%}")
            logger.info(f"   Edge: {tp_ratio - accuracy_required:.1%}p")
            logger.info(f"   Skipped: {skipped} (buffer insufficiente)")
        else:
            logger.error("❌ Nessuna sequenza valida!")
        
        if len(X) == 0:
            logger.error("❌ Nessuna sequenza valida!")
            return np.array([]), np.array([])
        
        return np.array(X), np.array(y)
    
    # ... (le altre funzioni rimangono uguali) ...
    
    def setup_predictor(self, model_name: str = None) -> Dict[str, Any]:
        """
        Configura il predictor.
        """
        logger.info("=" * 60)
        logger.info("CONFIGURAZIONE PREDICTOR")
        logger.info("=" * 60)
        
        try:
            # Crea predictor dall'ultimo modello
            predictor = create_predictor_from_training(
                model_dir=self.config["paths"]["models"]
            )
            
            self.predictor = predictor
            
            # Aggiorna stato
            self.system_state["predictor_ready"] = True
            self.system_state["predictor_info"] = predictor.get_model_info()
            
            logger.info(f"✅ Predictor configurato")
            
            return {
                "success": True,
                "predictor_info": self.system_state["predictor_info"]
            }
            
        except Exception as e:
            logger.error(f"Errore configurazione predictor: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, generate_order: bool = True) -> Dict[str, Any]:
        """
        Effettua predizione con TP/SL.
        """
        logger.info("=" * 60)
        logger.info("PREDIZIONE CON TP/SL")
        logger.info("=" * 60)
        
        try:
            # Verifica predictor
            if not self.system_state["predictor_ready"]:
                self.setup_predictor()
            
            # 🔥 CARICAMENTO DATI SE NECESSARIO
            if not self.system_state["data_loaded"]:
                logger.info("Caricamento dati in corso...")
                data_result = self.load_all_data()
                if not data_result["success"]:
                    raise ValueError(f"Errore caricamento dati: {data_result.get('error')}")
            
            # Recupera dati
            if "data" not in self.results_cache:
                raise ValueError("Cache dati vuota")
            
            data = self.results_cache["data"]
            test_data = data["X_test"]
            
            if len(test_data) == 0:
                raise ValueError("Nessun dato di test disponibile")
            
            # Usa ultima sequenza
            last_sequence = test_data[-1:]
            
            # Predizione
            prediction_result = self.predictor.predict_single(
                last_sequence,
                return_confidence=True
            )
            
            # Genera ordine se richiesto
            if generate_order:
                from trading_executor import create_executor_from_prediction
                order_result = create_executor_from_prediction(prediction_result)
                prediction_result["order"] = order_result
            
            # Salva in cache
            if "predictions" not in self.results_cache:
                self.results_cache["predictions"] = []
            
            self.results_cache["predictions"].append({
                "result": prediction_result,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"✅ Predizione completata")
            
            return {
                "success": True,
                "prediction": prediction_result
            }
            
        except Exception as e:
            logger.error(f"Errore nella predizione: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_model(self) -> Dict[str, Any]:
        """
        Analizza il modello.
        """
        logger.info("=" * 60)
        logger.info("ANALISI MODELLO")
        logger.info("=" * 60)
        
        try:
            # Verifica dati
            if not self.system_state["data_loaded"]:
                data_result = self.load_all_data()
                if not data_result["success"]:
                    raise ValueError("Impossibile caricare dati per analisi")
            
            # Verifica modello
            if not self.system_state["model_trained"]:
                raise ValueError("Nessun modello addestrato disponibile")
            
            # Recupera dati
            data = self.results_cache["data"]
            model_data = self.results_cache["model"]
            
            X_train, X_test = data["X_train"], data["X_test"]
            y_train, y_test = data["y_train"], data["y_test"]
            loader = data["loader"]
            model_info = model_data["model_info"]
            
            # Configura percorsi
            analysis_dir = self.config["paths"]["analysis"]
            model_name = model_info["model_name"]
            output_dir = os.path.join(analysis_dir, model_name)
            
            # Esegui analisi
            analyzer = analyze_trading_model(
                model_path=os.path.join(self.config["paths"]["models"], f"{model_name}.h5"),
                scaler_path=os.path.join(self.config["paths"]["models"], f"{model_name}_scaler.pkl"),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=loader.feature_names,
                output_dir=output_dir
            )
            
            # Cerca report generato
            import glob
            report_files = glob.glob(os.path.join(output_dir, "model_analysis_report_*.json"))
            
            report_path = report_files[0] if report_files else None
            
            logger.info(f"✅ Analisi completata")
            
            return {
                "success": True,
                "report_path": report_path,
                "output_dir": output_dir
            }
            
        except Exception as e:
            logger.error(f"Errore nell'analisi: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def setup_chatbot(self, api_key: str = None) -> Dict[str, Any]:
        """
        Configura chatbot AI.
        """
        logger.info("=" * 60)
        logger.info("CONFIGURAZIONE CHATBOT AI")
        logger.info("=" * 60)
        
        try:
            # Determina API key
            if not api_key:
                api_key = os.getenv(self.config["chat"]["api_key_env_var"])
            
            # Configura chatbot
            chat_config = self.config["chat"]
            self.chatbot = DeepSeekChat(
                api_key=api_key,
                model=chat_config["model"],
                max_tokens=chat_config["max_tokens"],
                temperature=chat_config["temperature"]
            )
            
            # Aggiorna stato
            self.system_state["chatbot_ready"] = True
            
            logger.info(f"✅ Chatbot configurato")
            
            return {
                "success": True,
                "model": chat_config["model"]
            }
            
        except Exception as e:
            logger.error(f"Errore configurazione chatbot: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def interactive_chat(self):
        """
        Chat interattiva con AI.
        """
        logger.info("=" * 60)
        logger.info("CHAT INTERATTIVA")
        logger.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("💬 CHAT INTERATTIVA TRADING AI")
        print("=" * 60)
        
        # Verifica chatbot
        if not self.system_state["chatbot_ready"]:
            print("Configuro chatbot...")
            self.setup_chatbot()
        
        print("\nComandi speciali:")
        print("  /help     - Mostra comandi")
        print("  /status   - Stato sistema")
        print("  /clear    - Pulisce conversazione")
        print("  /exit     - Esci")
        print("\n" + "=" * 60)
        
        while True:
            try:
                user_input = input("\nTu: ").strip()
                
                if not user_input:
                    continue
                
                # Comandi speciali
                if user_input.lower() == "/exit":
                    print("Arrivederci!")
                    break
                
                elif user_input.lower() == "/help":
                    print("\nComandi:")
                    print("  /help     - Questo messaggio")
                    print("  /status   - Stato sistema")
                    print("  /predict  - Effettua predizione")
                    print("  /analyze  - Analizza ultima predizione")
                    print("  /clear    - Pulisce conversazione")
                    print("  /save     - Salva conversazione")
                    print("  /exit     - Esci")
                    continue
                
                elif user_input.lower() == "/status":
                    self._print_system_status()
                    continue
                
                elif user_input.lower() == "/clear":
                    self.chatbot.clear_conversation()
                    print("Conversazione pulita")
                    continue
                
                elif user_input.lower() == "/save":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"chat_conversation_{timestamp}.json"
                    self.chatbot.save_conversation(filepath)
                    print(f"Conversazione salvata: {filepath}")
                    continue
                
                # Chat normale
                print("AI: ", end="", flush=True)
                response = self.chatbot.chat(user_input)
                print(response["content"])
                
            except KeyboardInterrupt:
                print("\n\nInterrotto")
                break
            except Exception as e:
                print(f"\nErrore: {str(e)}")
    
    def predict_from_file(self, test_file: str, generate_order: bool = True) -> Dict[str, Any]:
        """
        Predizione su file specifico.
        """
        logger.info("=" * 60)
        logger.info(f"PREDIZIONE SU FILE: {os.path.basename(test_file)}")
        logger.info("=" * 60)
        
        try:
            # Verifica predictor
            if not self.system_state["predictor_ready"]:
                self.setup_predictor()
            
            # Carica file specifico
            if not os.path.exists(test_file):
                raise FileNotFoundError(f"File non trovato: {test_file}")
            
            df = pd.read_csv(test_file)
            
            # Verifica colonne necessarie
            required_cols = ['price', 'timestamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Colonne mancanti: {missing_cols}")
            
            # Prepara dati per predizione
            sequence_length = self.predictor.sequence_length
            
            if len(df) < sequence_length:
                raise ValueError(f"File troppo corto: {len(df)} < {sequence_length}")
            
            # Prendi ultime N righe
            recent_data = df.iloc[-sequence_length:]
            
            # Predizione
            prediction_result = self.predictor.predict_single(
                recent_data,
                return_confidence=True
            )
            
            # Genera ordine se richiesto
            if generate_order:
                from trading_executor import create_executor_from_prediction
                order_result = create_executor_from_prediction(prediction_result)
                prediction_result["order"] = order_result
            
            logger.info(f"✅ Predizione completata")
            
            return {
                "success": True,
                "prediction": prediction_result,
                "test_file": test_file
            }
            
        except Exception as e:
            logger.error(f"Errore predizione file: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "test_file": test_file
            }
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Pipeline completo: training → predizione → analisi.
        """
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETO")
        logger.info("=" * 60)
        
        try:
            # 1. Training
            print("\n1. Addestramento modello...")
            train_result = self.run_training_pipeline()
            
            if not train_result["success"]:
                raise ValueError(f"Training fallito: {train_result.get('error')}")
            
            # 2. Predizione
            print("\n2. Predizione con TP/SL...")
            self.setup_predictor()
            predict_result = self.predict(generate_order=True)
            
            if not predict_result["success"]:
                logger.warning(f"Predizione fallita: {predict_result.get('error')}")
            
            # 3. Analisi
            print("\n3. Analisi modello...")
            analyze_result = self.analyze_model()
            
            if not analyze_result["success"]:
                logger.warning(f"Analisi fallita: {analyze_result.get('error')}")
            
            # 4. Chatbot (opzionale)
            print("\n4. Configurazione chatbot...")
            chatbot_result = self.setup_chatbot()
            
            self.system_state["initialized"] = True
            self.system_state["last_update"] = datetime.now().isoformat()
            
            logger.info("\n✅ PIPELINE COMPLETATO")
            
            return {
                "success": True,
                "training": train_result,
                "prediction": predict_result,
                "analysis": analyze_result,
                "chatbot": chatbot_result
            }
            
        except Exception as e:
            logger.error(f"Errore pipeline: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _print_system_status(self):
        """Stampa stato sistema."""
        print("\n" + "=" * 60)
        print("STATO SISTEMA")
        print("=" * 60)
        
        for key, value in self.system_state.items():
            if key not in ["last_update"] and not isinstance(value, dict):
                status = "✅" if value else "❌" if isinstance(value, bool) else ""
                print(f"{key.replace('_', ' ').title():20} {status} {value}")
        
        print("\nDETTAGLI:")
        
        if self.system_state.get("data_loaded"):
            info = self.system_state["data_info"]
            print(f"📊 Dati: {info.get('total_samples', 0)} campioni")
            print(f"📁 File: {info.get('total_files', 0)} CSV")
        
        if self.system_state.get("model_trained"):
            info = self.system_state["model_info"]
            print(f"🤖 Modello: {info.get('model_name', 'N/A')}")
        
        if self.system_state.get("predictor_ready"):
            print(f"🎯 Predictor: ✅ Pronto")
        
        if self.system_state.get("chatbot_ready"):
            print(f"💬 Chatbot: ✅ Pronto")
        
        print("=" * 60)

def main():
    """Funzione principale."""
    orchestrator = TradingAIOrchestrator()
    orchestrator.run_full_pipeline()

if __name__ == "__main__":
    main()