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
        import json
        
        models_dir = self.config["paths"]["models"]
        
        # CERCA TUTTI i file .h5 e .keras
        all_model_files = []
        all_model_files.extend(glob.glob(os.path.join(models_dir, "*.h5")))
        all_model_files.extend(glob.glob(os.path.join(models_dir, "*.keras")))
        
        if not all_model_files:
            logger.info("⚠️  Nessun modello trovato in models/")
            return None
        
        # Filtra SOLO i modelli TP/SL
        tp_sl_keywords = [
            'tp_sl',           # tp_sl_... (generico)
            'tpsl',            # tpsl_... (senza underscore)
            '_tp_',            # ..._tp_... (contenente _tp_)
            '_sl_',            # ..._sl_... (contenente _sl_)
            'setup_classifier',# ...setup_classifier... (tuo modello)
            'take_profit',     # take_profit...
            'stop_loss',       # stop_loss...
            'classifier'       # ...classifier... (classificatori)
        ]
        
        tp_sl_models = []
        for model_file in all_model_files:
            filename = os.path.basename(model_file).lower()
            logger.debug(f"Verificando file: {filename}")
            
            # CRITERIO 1: Keyword nel nome file
            is_tp_sl_by_name = any(keyword in filename for keyword in tp_sl_keywords)
            
            # CRITERIO 2: Metadata se disponibili
            is_tp_sl_by_metadata = False
            metadata_file = model_file.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
            
            # Cerca anche metadata con pattern più flessibile
            if not os.path.exists(metadata_file):
                # Prova pattern alternativi per metadata
                metadata_patterns = [
                    model_file.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json'),
                    model_file.replace('.h5', '_meta.json').replace('.keras', '_meta.json'),
                    os.path.join(models_dir, f"{os.path.basename(model_file).replace('.h5', '').replace('.keras', '')}*metadata*.json"),
                    os.path.join(models_dir, f"{os.path.basename(model_file).replace('.h5', '').replace('.keras', '')}*meta*.json")
                ]
                
                for pattern in metadata_patterns:
                    if '*' in pattern:
                        matches = glob.glob(pattern)
                        if matches:
                            metadata_file = matches[0]
                            break
                    elif os.path.exists(pattern):
                        metadata_file = pattern
                        break
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Verifica nei metadata se è un modello TP/SL
                    problem_type = metadata.get('problem_type', '').lower()
                    training_type = metadata.get('training_type', '').lower()
                    model_type = metadata.get('model_type', '').lower()
                    
                    # Criteri per identificare modello TP/SL
                    if any(keyword in str(problem_type) for keyword in ['classification', 'trading', 'tp_sl', 'setup']):
                        is_tp_sl_by_metadata = True
                    elif any(keyword in str(training_type) for keyword in ['setup_classification', 'trading_setup', 'tp_sl']):
                        is_tp_sl_by_metadata = True
                    elif 'trading_params' in metadata or 'tp_pips' in metadata or 'sl_pips' in metadata:
                        is_tp_sl_by_metadata = True
                        
                    logger.debug(f"  Metadata {os.path.basename(metadata_file)}: problem_type={problem_type}, training_type={training_type}")
                        
                except Exception as e:
                    logger.debug(f"  Errore lettura metadata {metadata_file}: {e}")
            else:
                logger.debug(f"  Metadata non trovato per {os.path.basename(model_file)}")
            
            # Se soddisfa almeno uno dei criteri
            if is_tp_sl_by_name or is_tp_sl_by_metadata:
                tp_sl_models.append(model_file)
                logger.info(f"✅ Rilevato modello TP/SL: {os.path.basename(model_file)}")
                if is_tp_sl_by_name:
                    logger.debug(f"  - Riconosciuto per nome (keyword match)")
                if is_tp_sl_by_metadata:
                    logger.debug(f"  - Riconosciuto per metadata")
            else:
                logger.debug(f"❌ Scartato: {os.path.basename(model_file)} (non TP/SL)")
        
        if not tp_sl_models:
            logger.info("⚠️  Nessun modello TP/SL trovato in models/")
            
            # Mostra cosa c'è nella cartella
            logger.info("📁 Elenco file in models/:")
            for f in sorted(os.listdir(models_dir)):
                if f.endswith('.h5') or f.endswith('.keras'):
                    logger.info(f"  🤖 {f}")
                elif 'scaler' in f.lower() and f.endswith('.pkl'):
                    logger.info(f"  📊 {f} (scaler)")
                elif 'metadata' in f.lower() or 'meta' in f.lower():
                    logger.info(f"  📄 {f} (metadata)")
                    
            return None
        
        # Ordina per data di modifica (più recente primo)
        tp_sl_models.sort(key=os.path.getmtime, reverse=True)
        latest_model = tp_sl_models[0]
        
        # 🔥 CERCA SCALER CORRISPONDENTE
        scaler_found = False
        scaler_file = None
        
        model_base_name = os.path.basename(latest_model).replace('.h5', '').replace('.keras', '')
        
        # Pattern per cercare scaler
        scaler_patterns = [
            os.path.join(models_dir, f"{model_base_name}_scaler.pkl"),
            os.path.join(models_dir, f"{model_base_name}*scaler*.pkl"),
            os.path.join(models_dir, f"{model_base_name.split('_classifier')[0]}*scaler*.pkl"),  # Senza _classifier suffix
            os.path.join(models_dir, "*scaler*.pkl")  # Tutti gli scaler
        ]
        
        for pattern in scaler_patterns:
            scaler_files = glob.glob(pattern)
            if scaler_files:
                # Prendi lo scaler più recente per questo modello
                scaler_files.sort(key=os.path.getmtime, reverse=True)
                scaler_file = scaler_files[0]
                scaler_found = True
                break
        
        if not scaler_found:
            logger.error(f"❌ Nessuno scaler trovato per {os.path.basename(latest_model)}!")
            logger.error("   Pattern cercati:")
            for pattern in scaler_patterns[:3]:  # Mostra solo primi 3 pattern
                logger.error(f"   - {os.path.basename(pattern)}")
            return None
        
        # 🔥 CERCA METADATA
        metadata_found = False
        metadata_file = None
        
        metadata_patterns = [
            os.path.join(models_dir, f"{model_base_name}_metadata.json"),
            os.path.join(models_dir, f"{model_base_name}*metadata*.json"),
            os.path.join(models_dir, f"{model_base_name.split('_classifier')[0]}*metadata*.json"),
            os.path.join(models_dir, "*metadata*.json")  # Tutti i metadata
        ]
        
        for pattern in metadata_patterns:
            metadata_files = glob.glob(pattern)
            if metadata_files:
                metadata_files.sort(key=os.path.getmtime, reverse=True)
                metadata_file = metadata_files[0]
                metadata_found = True
                break
        
        logger.info(f"✅ Modello TP/SL trovato: {os.path.basename(latest_model)}")
        logger.info(f"   Scaler: {os.path.basename(scaler_file) if scaler_file else 'NON TROVATO'}")
        logger.info(f"   Metadata: {os.path.basename(metadata_file) if metadata_file else 'NON TROVATO'}")
        
        # Log dettagliato se trovato
        if metadata_found:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"   Tipo: {metadata.get('problem_type', 'N/A')}")
                logger.info(f"   Data: {metadata.get('training_date', 'N/A')}")
                if 'trading_params' in metadata:
                    logger.info(f"   TP/SL: {metadata['trading_params'].get('tp_pips', 'N/A')}/{metadata['trading_params'].get('sl_pips', 'N/A')} pips")
            except:
                pass
        
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
        Training TP/SL che USA il modello di pattern recognition (Fase 1).
        Se non trova modello pattern, usa approccio alternativo.
        """
        logger.info("=" * 60)
        logger.info("🚀 TRAINING TP/SL CON/SENZA PATTERN RECOGNITION")
        logger.info("=" * 60)
        
        try:
            pattern_model_path = None
            pattern_predictor = None
            
            # 🔥 PASSO 1: CERCA MODELLO PATTERN
            try:
                pattern_model_path = self._find_latest_regression_model()
                if pattern_model_path:
                    logger.info(f"1. Pattern model trovato: {os.path.basename(pattern_model_path)}")
                    
                    # Prova a caricare il pattern model
                    from predictor import TradingPredictor
                    scaler_path = pattern_model_path.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
                    
                    if not os.path.exists(scaler_path):
                        # Cerca scaler alternativo
                        import glob
                        model_dir = os.path.dirname(pattern_model_path)
                        model_base = os.path.basename(pattern_model_path).replace('.h5', '').replace('.keras', '')
                        scaler_pattern = os.path.join(model_dir, f"{model_base}*scaler*.pkl")
                        scaler_files = glob.glob(scaler_pattern)
                        if scaler_files:
                            scaler_path = scaler_files[0]
                    
                    pattern_predictor = TradingPredictor(
                        model_path=pattern_model_path,
                        scaler_path=scaler_path if os.path.exists(scaler_path) else None,
                        sequence_length=self.config["data"]["sequence_length"]
                    )
                    logger.info("✅ Pattern model caricato con successo")
                else:
                    logger.warning("1. ⚠️  Nessun modello pattern recognition trovato")
                    logger.info("   Userò approccio senza pattern model")
                    
            except Exception as e:
                logger.warning(f"1. ⚠️  Errore caricamento pattern model: {e}")
                logger.info("   Continuo senza pattern model")
                pattern_predictor = None
            
            # 🔥 PASSO 2: CARICA DATI
            logger.info("\n2. Caricamento dati per training TP/SL...")
            
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
            
            # 🔥 PASSO 3: CREA SETUP DI INGRESSO
            logger.info("\n3. Creazione setup di trading...")
            
            price_data = df_original['price'].values
            feature_columns = loader.feature_names
            
            if pattern_predictor:
                # Con pattern model
                X_setups, y_setups = self._create_trading_setups_with_patterns(
                    features_data=df_normalized[feature_columns].values,
                    price_data=price_data,
                    sequence_length=sequence_length,
                    pattern_predictor=pattern_predictor,
                    tp_pips=80,
                    sl_pips=20
                )
            else:
                # Senza pattern model - approccio semplificato
                logger.info("   Usando approccio semplificato (senza pattern model)")
                X_setups, y_setups = self._create_trading_setups_simple(
                    features_data=df_normalized[feature_columns].values,
                    price_data=price_data,
                    sequence_length=sequence_length,
                    tp_pips=80,
                    sl_pips=20
                )
            
            if len(X_setups) == 0:
                raise ValueError("Nessun setup di ingresso valido trovato!")
            
            # 🔥 PASSO 4: SPLIT E TRAINING
            logger.info("\n4. Training classificatore setup...")
            
            split_idx = int(len(X_setups) * 0.8)
            X_train_setup = X_setups[:split_idx]
            X_test_setup = X_setups[split_idx:]
            y_train_setup = y_setups[:split_idx]
            y_test_setup = y_setups[split_idx:]
                    
            logger.info(f"📊 DATASET SETUP TRADING:")
            logger.info(f"   Total setups: {len(X_setups)}")
            logger.info(f"   Train: {len(X_train_setup)} samples")
            logger.info(f"   Test: {len(X_test_setup)} samples")
            logger.info(f"   TP setups: {np.sum(y_setups == 1)} ({(np.sum(y_setups == 1)/len(y_setups)):.1%})")
            logger.info(f"   SL setups: {np.sum(y_setups == 0)} ({(np.sum(y_setups == 0)/len(y_setups)):.1%})")
            
            # 🔥 PASSO 5: TRAINING CLASSIFICATORE SETUP
            model_config = self.config["model"]
            model_name = f"tp_sl_setup_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            from model_trainer import train_model_pipeline
            trainer, training_results = train_model_pipeline(
                X_train=X_train_setup,
                y_train=y_train_setup,
                X_test=X_test_setup,
                y_test=y_test_setup,
                sequence_length=sequence_length,
                n_features=X_train_setup.shape[2],
                model_type=model_config["type"],
                epochs=150,
                batch_size=32,
                model_name=model_name,
                data_loader=loader,
                problem_type='classification'
            )
            
            # 🔥 PASSO 6: VALUTAZIONE E SALVATAGGIO
            logger.info("\n5. Valutazione classificatore setup...")
            
            from sklearn.metrics import accuracy_score, classification_report
            y_pred_proba = trainer.model.predict(X_test_setup, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test_setup, y_pred)
            
            logger.info(f"📊 PERFORMANCE CLASSIFICAZIONE SETUP:")
            logger.info(f"   Accuracy: {accuracy:.2%}")
            logger.info(classification_report(y_test_setup, y_pred, 
                                            target_names=['SL Setup', 'TP Setup']))
            
            # Salva modello
            trainer.save_model(
                metadata={
                    'sequence_length': sequence_length,
                    'n_features': X_train_setup.shape[2],
                    'model_type': model_config["type"],
                    'problem_type': 'trading_setup_classification',
                    'pattern_model': os.path.basename(pattern_model_path),
                    'performance': {'accuracy': float(accuracy)},
                    'trading_params': {
                        'rr_ratio': '4:1',
                        'tp_pips': 80,
                        'sl_pips': 20,
                        'required_win_rate': 0.20
                    }
                }
            )
            
            # Aggiorna predictor
            self.predictor = TradingPredictor(
                model_path=os.path.join(self.config["paths"]["models"], f"{model_name}.h5"),
                scaler_path=os.path.join(self.config["paths"]["models"], f"{model_name}_scaler.pkl"),
                sequence_length=sequence_length,
                confidence_level=0.7
            )
            
            self.system_state["predictor_ready"] = True
            
            logger.info(f"✅ TRAINING TP/SL COMPLETATO!")
            logger.info(f"   Modello: {model_name}")
            logger.info(f"   Accuracy setup: {accuracy:.2%}")
            logger.info(f"   Pattern model: {os.path.basename(pattern_model_path)}")
            
            return {
                'success': True,
                'training_type': 'setup_classification',
                'model_name': model_name,
                'accuracy': accuracy,
                'pattern_model': os.path.basename(pattern_model_path),
                'message': f'Training TP/SL completato. Accuracy: {accuracy:.2%}'
            }
            
        except Exception as e:
            logger.error(f"❌ Errore nel training TP/SL: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'training_type': 'setup_classification',
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
    
    def analyze_model(self, model_path: str = None) -> Dict[str, Any]:
        """
        Analizza il modello usando ESCLUSIVAMENTE la cache per i dati.
        """
        logger.info("=" * 60)
        logger.info("📊 ANALISI MODELLO (VERSIONE CACHE)")
        logger.info("=" * 60)
        
        try:
            # 🔥 1. CARICA DATI SE NECESSARIO
            if not self.system_state["data_loaded"]:
                logger.info("Caricamento dati per analisi...")
                data_result = self.load_all_data()
                if not data_result["success"]:
                    raise ValueError(f"Errore caricamento dati: {data_result.get('error')}")
            
            # 🔥 2. RECUPERA TUTTO DALLA CACHE
            if "data" not in self.results_cache:
                raise ValueError("Cache dati vuota. Esegui load_all_data() prima.")
            
            data = self.results_cache["data"]
            
            # Estrai tutto dalla cache
            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            loader = data["loader"]
            df_original = data["df_original"]  # 🔥 SEMPRE dalla cache
            df_normalized = data.get("df_normalized")  # Opzionale
            
            logger.info(f"📦 Dati dalla cache:")
            logger.info(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
            logger.info(f"   df_original shape: {df_original.shape}")
            logger.info(f"   Sequence length: {loader.sequence_length}")
            
            # 🔥 3. TROVA IL MODELLO
            models_dir = self.config["paths"]["models"]
            
            if model_path is None:
                import glob
                model_files = []
                model_files.extend(glob.glob(os.path.join(models_dir, "*.keras")))
                model_files.extend(glob.glob(os.path.join(models_dir, "*.h5")))
                
                if not model_files:
                    raise ValueError(f"Nessun modello trovato in {models_dir}")
                
                model_files.sort(key=os.path.getmtime, reverse=True)
                model_path = model_files[0]
            
            logger.info(f"🤖 Analisi modello: {os.path.basename(model_path)}")
            
            # 🔥 4. TROVA SCALER
            model_base = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
            scaler_pattern = os.path.join(models_dir, f"{model_base}*scaler*.pkl")
            import glob
            scaler_files = glob.glob(scaler_pattern)
            
            scaler_path = scaler_files[0] if scaler_files else None
            if scaler_path:
                logger.info(f"📊 Scaler: {os.path.basename(scaler_path)}")
            
            # 🔥 5. DETERMINA TIPO MODELLO
            is_classification = self._is_classification_model(model_path)
            
            if is_classification:
                logger.info("🎯 Modello di CLASSIFICAZIONE TP/SL rilevato")
                
                # 🔥 6a. PER CLASSIFICAZIONE: Crea target binari
                logger.info("   Creazione target binari per valutazione...")
                
                # Usa df_original dalla cache
                price_data = df_original['price'].values
                
                y_train_binary = self._create_binary_targets_from_data(
                    X_train, price_data, loader.sequence_length,
                    tp_pips=80, sl_pips=20
                )
                y_test_binary = self._create_binary_targets_from_data(
                    X_test, price_data, loader.sequence_length,
                    tp_pips=80, sl_pips=20
                )
                
                logger.info(f"   Train binary: TP={np.sum(y_train_binary==1)}, SL={np.sum(y_train_binary==0)}")
                logger.info(f"   Test binary: TP={np.sum(y_test_binary==1)}, SL={np.sum(y_test_binary==0)}")
                
                # Usa target binari per analisi
                y_train_analysis = y_train_binary
                y_test_analysis = y_test_binary
                
                # Info trading
                total_setups = len(y_test_binary)
                if total_setups > 0:
                    win_rate = np.sum(y_test_binary == 1) / total_setups
                    required_win_rate = 20 / (80 + 20)  # 0.20 per R/R 4:1
                    logger.info(f"   Win rate test: {win_rate:.1%}")
                    logger.info(f"   Win rate richiesto: {required_win_rate:.1%}")
                    logger.info(f"   Edge: {win_rate - required_win_rate:+.1%}p")
                    
            else:
                logger.info("📈 Modello di REGRESSIONE rilevato")
                y_train_analysis = y_train
                y_test_analysis = y_test
            
            # 🔥 7. CONFIGURA OUTPUT
            analysis_dir = self.config["paths"]["analysis"]
            model_name = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(analysis_dir, f"{model_name}_analysis_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"📁 Output directory: {output_dir}")
            
            # 🔥 8. ESEGUI ANALISI
            from pattern_analyzer import analyze_trading_model
            
            try:
                analyzer = analyze_trading_model(
                    model_path=model_path,
                    scaler_path=scaler_path,
                    X_train=X_train,
                    y_train=y_train_analysis,
                    X_test=X_test,
                    y_test=y_test_analysis,
                    feature_names=loader.feature_names,
                    output_dir=output_dir
                )
                
                # 🔥 9. CERCA REPORT
                import glob
                json_reports = glob.glob(os.path.join(output_dir, "model_analysis_*.json"))
                png_files = glob.glob(os.path.join(output_dir, "*.png"))
                
                report_path = json_reports[0] if json_reports else None
                
                logger.info(f"✅ Analisi completata!")
                logger.info(f"   File generati: {len(json_reports)} JSON, {len(png_files)} PNG")
                
                if report_path:
                    # Leggi e mostra sommario
                    import json
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    
                    logger.info(f"\n📊 PERFORMANCE RIEPILOGO:")
                    
                    perf = report.get('performance', {})
                    if is_classification:
                        if 'test_accuracy' in perf:
                            logger.info(f"   Accuracy: {perf['test_accuracy']:.2%}")
                        if 'test_precision' in perf:
                            logger.info(f"   Precision: {perf['test_precision']:.2%}")
                        if 'test_recall' in perf:
                            logger.info(f"   Recall: {perf['test_recall']:.2%}")
                    else:
                        if 'test_r2' in perf:
                            logger.info(f"   R²: {perf['test_r2']:.3f}")
                        if 'test_mse' in perf:
                            logger.info(f"   MSE: {perf['test_mse']:.6f}")
                    
                    insights = report.get('insights', [])
                    if insights:
                        logger.info(f"\n💡 INSIGHTS:")
                        for insight in insights[:5]:
                            logger.info(f"   • {insight}")
                
                return {
                    "success": True,
                    "report_path": report_path,
                    "output_dir": output_dir,
                    "model_type": "classification" if is_classification else "regression",
                    "model_file": os.path.basename(model_path),
                    "model_info": self._get_model_metadata(model_path)
                }
                
            except Exception as e:
                logger.error(f"❌ Errore durante analyze_trading_model: {e}")
                raise
            
        except Exception as e:
            logger.error(f"❌ Errore nell'analisi: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def _create_binary_targets_from_data(self, X_sequences: np.ndarray,
                                    price_data: np.ndarray,
                                    sequence_length: int,
                                    tp_pips: int = 80,
                                    sl_pips: int = 20,
                                    lookahead_multiplier: int = 10) -> np.ndarray:
        """
        Crea target binari per classificazione basati su dati reali.
        1 = TP raggiunto prima, 0 = SL raggiunto prima
        """
        y_binary = []
        pip_value = 0.0001
        buffer_ticks = max(tp_pips, sl_pips) * lookahead_multiplier
        
        logger.info(f"   Creazione target binari:")
        logger.info(f"   TP: {tp_pips} pips, SL: {sl_pips} pips")
        logger.info(f"   Buffer: {buffer_ticks} ticks")
        
        tp_count = 0
        sl_count = 0
        no_outcome = 0
        
        for i in range(len(X_sequences)):
            # Calcola indice corrispondente nel price_data
            # X_sequences sono state create con sliding window
            # L'ultimo prezzo della sequenza è all'indice: i + sequence_length - 1
            price_idx = i + sequence_length - 1
            
            if price_idx >= len(price_data):
                no_outcome += 1
                y_binary.append(0)  # Default
                continue
            
            if price_idx + buffer_ticks >= len(price_data):
                # Dati futuri insufficienti
                no_outcome += 1
                y_binary.append(0)  # Default
                continue
            
            current_price = price_data[price_idx]
            future_prices = price_data[price_idx + 1:price_idx + 1 + buffer_ticks]
            
            if len(future_prices) == 0:
                no_outcome += 1
                y_binary.append(0)
                continue
            
            tp_reached = False
            sl_reached = False
            
            # Verifica TP/SL
            for future_price in future_prices:
                # TP (assume sempre long per valutazione)
                if future_price >= current_price + (tp_pips * pip_value):
                    tp_reached = True
                    break
                
                # SL
                if future_price <= current_price - (sl_pips * pip_value):
                    sl_reached = True
                    break
            
            if tp_reached:
                y_binary.append(1)
                tp_count += 1
            elif sl_reached:
                y_binary.append(0)
                sl_count += 1
            else:
                # Nessuno raggiunto nel buffer
                y_binary.append(0)
                no_outcome += 1
        
        total = len(X_sequences)
        logger.info(f"   Risultati: TP={tp_count} ({tp_count/total:.1%}), "
                    f"SL={sl_count} ({sl_count/total:.1%}), "
                    f"No outcome={no_outcome} ({no_outcome/total:.1%})")
        
        return np.array(y_binary)

    def _get_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """Legge metadata del modello."""
        import json
        import os
        
        metadata_path = model_path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
        
        if not os.path.exists(metadata_path):
            # Cerca pattern alternativo
            import glob
            model_dir = os.path.dirname(model_path)
            model_base = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
            metadata_pattern = os.path.join(model_dir, f"{model_base}*metadata*.json")
            metadata_files = glob.glob(metadata_pattern)
            metadata_path = metadata_files[0] if metadata_files else None
        
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        
        return {}

    def _is_classification_model(self, model_path: str) -> bool:
        """Determina se un modello è di classificazione."""
        import json
        import os
        
        # 1. Prova a leggere metadata
        metadata_path = model_path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                problem_type = metadata.get('problem_type', '').lower()
                training_type = metadata.get('training_type', '').lower()
                
                if any(keyword in problem_type for keyword in ['classification', 'binary', 'tp_sl', 'setup']):
                    return True
                if any(keyword in training_type for keyword in ['classification', 'unified', 'tp_sl']):
                    return True
                    
            except Exception as e:
                logger.warning(f"Errore lettura metadata: {e}")
        
        # 2. Prova a caricare il modello e controllare l'ultimo layer
        try:
            import keras
            model = keras.models.load_model(model_path)
            last_layer = model.layers[-1]
            
            if hasattr(last_layer, 'activation'):
                activation_name = str(last_layer.activation).lower()
                if 'sigmoid' in activation_name:
                    return True  # Sigmoid = classificazione binaria
                elif 'softmax' in activation_name:
                    return True  # Softmax = classificazione multi-classe
            
            # Controlla output shape
            output_shape = model.output_shape
            if len(output_shape) == 2 and output_shape[1] == 1:
                # Output singolo - potrebbe essere regressione o classificazione binaria
                # In dubbio, assumiamo regressione
                return False
            
        except Exception as e:
            logger.warning(f"Errore analisi modello: {e}")
        
        # Default: assume regressione
        return False

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

    def _create_trading_setups_with_patterns(self, features_data: np.ndarray,
                                        price_data: np.ndarray,
                                        sequence_length: int,
                                        pattern_predictor,
                                        tp_pips: int = 80,
                                        sl_pips: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea setup di trading usando pattern recognition.
        
        Logica:
        1. Per ogni possibile punto di ingresso
        2. Usa pattern model per predire movimento
        3. Se predizione > soglia → Setup potenziale
        4. Verifica se TP o SL viene raggiunto prima
        5. Label: 1=TP Setup, 0=SL Setup
        """
        X_setups, y_setups = [], []
        
        pip_value = 0.0001
        buffer_ticks = max(tp_pips, sl_pips) * 15
        
        # 🔥 STATISTICHE DETTAGLIATE
        stats = {
            'total_sequences': 0,
            'sequences_analyzed': 0,
            'no_future_data': 0,
            'weak_prediction': 0,
            'setup_identified': 0,
            'tp_setups': 0,
            'sl_setups': 0,
            'no_outcome': 0,
            'avg_prediction_strength': 0.0,
            'strong_predictions': 0,
            'weak_predictions': 0
        }
        
        prediction_strengths = []
        
        logger.info(f"🔍 ANALISI SEQUENZE PER SETUP TRADING:")
        logger.info(f"   Sequenze totali possibili: {len(features_data) - sequence_length - buffer_ticks}")
        logger.info(f"   Soglia entrata: movimento predetto > {tp_pips * 0.5} pips")
        logger.info(f"   TP: {tp_pips} pips, SL: {sl_pips} pips, R/R: {tp_pips/sl_pips:.1f}:1")
        logger.info(f"   Buffer verifica outcome: {buffer_ticks} ticks")
        
        for i in range(len(features_data) - sequence_length - buffer_ticks):
            stats['total_sequences'] += 1
            
            # Prendi sequenza per pattern recognition
            sequence = features_data[i:i + sequence_length]
            sequence_reshaped = sequence.reshape(1, sequence_length, -1)
            
            # 🔥 USO PATTERN MODEL PER PREDIRE MOVIMENTO
            try:
                # Pattern model predice variazione % (regressione)
                pattern_prediction = pattern_predictor.model.predict(sequence_reshaped, verbose=0)
                predicted_move_pips = pattern_prediction[0][0] * 100  # Converti % in pips approx
                prediction_strengths.append(abs(predicted_move_pips))
                
                stats['sequences_analyzed'] += 1
                
                # 🔥 CONDIZIONE ENTRATA: Movimento predetto > soglia
                entry_threshold = tp_pips * 0.5  # Entra se predice almeno metà del TP
                
                if abs(predicted_move_pips) >= entry_threshold:
                    stats['strong_predictions'] += 1
                else:
                    stats['weak_predictions'] += 1
                
                if abs(predicted_move_pips) < entry_threshold:
                    stats['weak_prediction'] += 1
                    continue  # Setup non sufficientemente forte
                    
                # Determina direzione
                if predicted_move_pips > 0:
                    direction = "BUY"
                else:
                    direction = "SELL"
                
                # 🔥 VERIFICA OUTCOME REALE
                current_idx = i + sequence_length - 1
                current_price = price_data[current_idx]
                
                future_prices = price_data[current_idx + 1:current_idx + 1 + buffer_ticks]
                
                if len(future_prices) == 0:
                    stats['no_future_data'] += 1
                    continue
                
                tp_reached = False
                sl_reached = False
                
                for future_price in future_prices:
                    if direction == "BUY":
                        if future_price >= current_price + (tp_pips * pip_value):
                            tp_reached = True
                            break
                        elif future_price <= current_price - (sl_pips * pip_value):
                            sl_reached = True
                            break
                    else:  # SELL
                        if future_price <= current_price - (tp_pips * pip_value):
                            tp_reached = True
                            break
                        elif future_price >= current_price + (sl_pips * pip_value):
                            sl_reached = True
                            break
                
                # 🔥 AGGIUNGI SETUP AL DATASET
                if tp_reached:
                    X_setups.append(sequence)
                    y_setups.append(1)  # TP Setup
                    stats['tp_setups'] += 1
                    stats['setup_identified'] += 1
                elif sl_reached:
                    X_setups.append(sequence)
                    y_setups.append(0)  # SL Setup
                    stats['sl_setups'] += 1
                    stats['setup_identified'] += 1
                else:
                    stats['no_outcome'] += 1
                    
            except Exception as e:
                logger.debug(f"Errore analisi sequenza {i}: {e}")
                continue
        
        # 🔥 STATISTICHE FINALI DETTAGLIATE
        if stats['sequences_analyzed'] > 0:
            stats['avg_prediction_strength'] = np.mean(prediction_strengths) if prediction_strengths else 0
        
        logger.info(f"\n📊 STATISTICHE ANALISI SEQUENZE:")
        logger.info(f"   Sequenze totali: {stats['total_sequences']}")
        logger.info(f"   Sequenze analizzate: {stats['sequences_analyzed']} ({stats['sequences_analyzed']/stats['total_sequences']:.1%})")
        logger.info(f"   Setup identificati: {stats['setup_identified']} ({stats['setup_identified']/stats['sequences_analyzed']:.1%} delle analizzate)")
        
        if stats['setup_identified'] > 0:
            logger.info(f"   • TP Setups: {stats['tp_setups']} ({stats['tp_setups']/stats['setup_identified']:.1%})")
            logger.info(f"   • SL Setups: {stats['sl_setups']} ({stats['sl_setups']/stats['setup_identified']:.1%})")
        
        logger.info(f"\n🔎 FILTRI APPLICATI:")
        logger.info(f"   Predizione debole (<{tp_pips * 0.5} pips): {stats['weak_prediction']}")
        logger.info(f"   Dati futuri insufficienti: {stats['no_future_data']}")
        logger.info(f"   Nessun outcome nel buffer: {stats['no_outcome']}")
        
        if stats['sequences_analyzed'] > 0:
            logger.info(f"\n📈 STATISTICHE PREDIZIONI PATTERN MODEL:")
            logger.info(f"   Predizioni forti (≥{tp_pips * 0.5} pips): {stats['strong_predictions']} ({stats['strong_predictions']/stats['sequences_analyzed']:.1%})")
            logger.info(f"   Predizioni deboli: {stats['weak_predictions']} ({stats['weak_predictions']/stats['sequences_analyzed']:.1%})")
            logger.info(f"   Forza predizione media: {stats['avg_prediction_strength']:.1f} pips")
        
        # Calcola win rate e profittabilità
        total_setups = stats['tp_setups'] + stats['sl_setups']
        if total_setups > 0:
            win_rate = stats['tp_setups'] / total_setups
            required_win_rate = sl_pips / (tp_pips + sl_pips)  # 20/(80+20) = 0.20
            
            logger.info(f"\n🎯 ANALISI PROFITTEVOLEZZA:")
            logger.info(f"   Win rate reale: {win_rate:.1%}")
            logger.info(f"   Win rate richiesto (R/R {tp_pips/sl_pips:.1f}:1): {required_win_rate:.1%}")
            logger.info(f"   Edge: {win_rate - required_win_rate:+.1%}p")
            
            if win_rate > required_win_rate:
                logger.info(f"   ✅ SETUP POTENZIALMENTE PROFITTEVOLI!")
                
                # Calcola Profit Factor atteso
                profit_factor = (stats['tp_setups'] * tp_pips) / (stats['sl_setups'] * sl_pips) if stats['sl_setups'] > 0 else float('inf')
                expected_profit = (stats['tp_setups'] * tp_pips) - (stats['sl_setups'] * sl_pips)
                
                logger.info(f"   Profit Factor atteso: {profit_factor:.2f}")
                logger.info(f"   Profitto atteso totale: {expected_profit:.0f} pips")
                logger.info(f"   Profitto medio per setup: {expected_profit/total_setups:.1f} pips")
            else:
                logger.warning(f"   ⚠️  Setup non profittevoli!")
                logger.warning(f"   Win rate insufficiente per R/R {tp_pips/sl_pips:.1f}:1")
        else:
            logger.error("❌ Nessun setup identificato!")
        
        if len(X_setups) == 0:
            logger.error("❌ Dataset vuoto - nessun setup valido trovato!")
            return np.array([]), np.array([])
        
        logger.info(f"\n✅ DATASET CREATO CON SUCCESSO:")
        logger.info(f"   Sequenze totali nel dataset: {len(X_setups)}")
        logger.info(f"   TP/SL ratio: {stats['tp_setups']}:{stats['sl_setups']}")
        
        return np.array(X_setups), np.array(y_setups)
    
    def _find_latest_regression_model(self) -> Optional[str]:
        """
        Cerca l'ultimo modello regressione nella directory models/.
        Supporta sia .h5 che .keras.
        """
        import glob
        
        models_dir = self.config["paths"]["models"]
        
        # Cerca modelli regressione in entrambi i formati
        patterns = [
            os.path.join(models_dir, "regression_model_*.keras"),
            os.path.join(models_dir, "regression_model_*.h5")
        ]
        
        regression_models = []
        for pattern in patterns:
            regression_models.extend(glob.glob(pattern))
        
        if not regression_models:
            logger.warning("⚠️  Nessun modello regressione trovato")
            return None
        
        # Ordina per data di modifica (più recente primo)
        regression_models.sort(key=os.path.getmtime, reverse=True)
        latest_model = regression_models[0]
        
        logger.info(f"✅ Modello regressione trovato: {os.path.basename(latest_model)}")
        return latest_model

    def _create_trading_setups_simple(self, features_data: np.ndarray,
                                 price_data: np.ndarray,
                                 sequence_length: int,
                                 tp_pips: int = 80,
                                 sl_pips: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea setup di trading senza pattern model (approccio semplificato).
        """
        X_setups, y_setups = [], []
        
        pip_value = 0.0001
        buffer_ticks = max(tp_pips, sl_pips) * 15
        
        # 🔥 STATISTICHE DETTAGLIATE
        stats = {
            'total_sequences': 0,
            'sequences_analyzed': 0,
            'no_future_data': 0,
            'no_setup_conditions': 0,
            'setup_identified': 0,
            'tp_setups': 0,
            'sl_setups': 0,
            'no_outcome': 0,
            'setup_conditions_count': []
        }
        
        logger.info(f"🔍 ANALISI SEQUENZE PER SETUP SEMPLICI:")
        logger.info(f"   Sequenze totali possibili: {len(features_data) - sequence_length - buffer_ticks}")
        logger.info(f"   TP: {tp_pips} pips, SL: {sl_pips} pips, R/R: {tp_pips/sl_pips:.1f}:1")
        logger.info(f"   Buffer verifica outcome: {buffer_ticks} ticks")
        logger.info(f"   Condizioni setup richieste: ≥2 tra [RSI_EXTREME, MACD_CROSS, NEAR_BB]")
        
        for i in range(len(features_data) - sequence_length - buffer_ticks):
            stats['total_sequences'] += 1
            
            # Estrai sequenza
            sequence = features_data[i:i + sequence_length]
            
            # 🔥 REGOLE SEMPLICI PER IDENTIFICARE SETUP
            setup_conditions = []
            
            if sequence.shape[1] >= 9:
                stats['sequences_analyzed'] += 1
                
                # RSI (colonna 7 in features normalizzate)
                rsi_values = sequence[:, 6]
                last_rsi = rsi_values[-1]
                
                # MACD (colonne 4-5)
                macd_main = sequence[:, 3]
                macd_signal = sequence[:, 4]
                
                # Bollinger Bands (colonne 0-2)
                bb_upper = sequence[:, 0]
                bb_lower = sequence[:, 2]
                price_relative = sequence[:, 8]
                
                # 🔥 CONDIZIONI PER SETUP POTENZIALE
                # Condizione 1: RSI estremo
                if last_rsi < -0.5:
                    setup_conditions.append("RSI_OVERSOLD")
                elif last_rsi > 0.5:
                    setup_conditions.append("RSI_OVERBOUGHT")
                
                # Condizione 2: MACD crossover
                if len(macd_main) > 1 and len(macd_signal) > 1:
                    if macd_main[-1] > macd_signal[-1] and macd_main[-2] <= macd_signal[-2]:
                        setup_conditions.append("MACD_BULLISH_CROSS")
                    elif macd_main[-1] < macd_signal[-1] and macd_main[-2] >= macd_signal[-2]:
                        setup_conditions.append("MACD_BEARISH_CROSS")
                
                # Condizione 3: Prezzo vicino a Bollinger Band
                if len(price_relative) > 0 and len(bb_upper) > 0 and len(bb_lower) > 0:
                    last_price = price_relative[-1]
                    last_bb_upper = bb_upper[-1]
                    last_bb_lower = bb_lower[-1]
                    
                    bb_width = last_bb_upper - last_bb_lower
                    if bb_width > 0:
                        distance_to_upper = (last_bb_upper - last_price) / bb_width
                        distance_to_lower = (last_price - last_bb_lower) / bb_width
                        
                        if distance_to_upper < 0.1:
                            setup_conditions.append("NEAR_BB_UPPER")
                        elif distance_to_lower < 0.1:
                            setup_conditions.append("NEAR_BB_LOWER")
                
                stats['setup_conditions_count'].append(len(setup_conditions))
                
                # 🔥 SE ABBASTANZA CONDIZIONI → SETUP POTENZIALE
                if len(setup_conditions) >= 2:
                    # Determina direzione
                    if "RSI_OVERSOLD" in setup_conditions or "NEAR_BB_LOWER" in setup_conditions or "MACD_BULLISH_CROSS" in setup_conditions:
                        direction = "BUY"
                    elif "RSI_OVERBOUGHT" in setup_conditions or "NEAR_BB_UPPER" in setup_conditions or "MACD_BEARISH_CROSS" in setup_conditions:
                        direction = "SELL"
                    else:
                        direction = "BUY"
                    
                    # 🔥 VERIFICA OUTCOME REALE
                    current_idx = i + sequence_length - 1
                    current_price = price_data[current_idx]
                    
                    future_prices = price_data[current_idx + 1:current_idx + 1 + buffer_ticks]
                    
                    if len(future_prices) == 0:
                        stats['no_future_data'] += 1
                        continue
                    
                    tp_reached = False
                    sl_reached = False
                    
                    for future_price in future_prices:
                        if direction == "BUY":
                            if future_price >= current_price + (tp_pips * pip_value):
                                tp_reached = True
                                break
                            elif future_price <= current_price - (sl_pips * pip_value):
                                sl_reached = True
                                break
                        else:
                            if future_price <= current_price - (tp_pips * pip_value):
                                tp_reached = True
                                break
                            elif future_price >= current_price + (sl_pips * pip_value):
                                sl_reached = True
                                break
                    
                    # 🔥 AGGIUNGI AL DATASET
                    if tp_reached:
                        X_setups.append(sequence)
                        y_setups.append(1)
                        stats['tp_setups'] += 1
                        stats['setup_identified'] += 1
                    elif sl_reached:
                        X_setups.append(sequence)
                        y_setups.append(0)
                        stats['sl_setups'] += 1
                        stats['setup_identified'] += 1
                    else:
                        stats['no_outcome'] += 1
                else:
                    stats['no_setup_conditions'] += 1
        
        # 🔥 STATISTICHE FINALI DETTAGLIATE
        logger.info(f"\n📊 STATISTICHE ANALISI SEQUENZE SEMPLICI:")
        logger.info(f"   Sequenze totali: {stats['total_sequences']}")
        logger.info(f"   Sequenze analizzate: {stats['sequences_analyzed']} ({stats['sequences_analyzed']/stats['total_sequences']:.1%})")
        logger.info(f"   Setup identificati: {stats['setup_identified']} ({stats['setup_identified']/stats['sequences_analyzed']:.1%} delle analizzate)")
        
        if stats['setup_identified'] > 0:
            logger.info(f"   • TP Setups: {stats['tp_setups']} ({stats['tp_setups']/stats['setup_identified']:.1%})")
            logger.info(f"   • SL Setups: {stats['sl_setups']} ({stats['sl_setups']/stats['setup_identified']:.1%})")
        
        logger.info(f"\n🔎 FILTRI APPLICATI:")
        logger.info(f"   Condizioni setup insufficienti: {stats['no_setup_conditions']}")
        logger.info(f"   Dati futuri insufficienti: {stats['no_future_data']}")
        logger.info(f"   Nessun outcome nel buffer: {stats['no_outcome']}")
        
        if stats['setup_conditions_count']:
            avg_conditions = np.mean(stats['setup_conditions_count'])
            logger.info(f"   Condizioni medie per sequenza: {avg_conditions:.1f}")
        
        # Calcola win rate
        total_setups = stats['tp_setups'] + stats['sl_setups']
        if total_setups > 0:
            win_rate = stats['tp_setups'] / total_setups
            required_win_rate = sl_pips / (tp_pips + sl_pips)
            
            logger.info(f"\n🎯 ANALISI PROFITTEVOLEZZA:")
            logger.info(f"   Win rate: {win_rate:.1%}")
            logger.info(f"   Win rate richiesto: {required_win_rate:.1%}")
            logger.info(f"   Edge: {win_rate - required_win_rate:+.1%}p")
        
        if len(X_setups) == 0:
            logger.error("❌ Nessun setup identificato!")
        
        return np.array(X_setups), np.array(y_setups)
    
    def run_unified_training_pipeline(self):
        """
        Pipeline UNIFICATA con singolo modello:
        1. Genera dati regressione (movimento futuro)
        2. Genera dati classificazione TP/SL
        3. Addestra modello con transfer learning
        """
        logger.info("=" * 60)
        logger.info("🚀 PIPELINE UNIFICATA - SINGOLO MODELLO")
        logger.info("=" * 60)
        
        try:
            # 1. Carica tutti i dati
            logger.info("\n1. Caricamento dati...")
            data_result = self.load_all_data()
            if not data_result["success"]:
                raise ValueError(f"Errore caricamento dati: {data_result.get('error')}")
            
            data = self.results_cache["data"]
            X_all = data["X_train"]  # Usa training set per addestramento
            loader = data["loader"]
            df_original = data["df_original"]
            
            # 2. Prepara dati per REGRESSIONE (movimento futuro)
            logger.info("\n2. Preparazione dati regressione...")
            price_data = df_original['price'].values
            sequence_length = self.config["data"]["sequence_length"]
            
            # Crea target regressione: movimento % nei prossimi N periodi
            y_regression = self._create_regression_targets(X_all, price_data, sequence_length)
            
            logger.info(f"   Regression samples: {len(X_all)}")
            logger.info(f"   Target mean: {np.mean(y_regression):.4f}, std: {np.std(y_regression):.4f}")
            
            # 3. Prepara dati per CLASSIFICAZIONE TP/SL
            logger.info("\n3. Preparazione dati classificazione TP/SL...")
            
            # Crea setup TP/SL reali (basati su dati storici)
            X_classification, y_classification = self._create_tp_sl_classification_data(
                loader=loader,
                df_normalized=data["df_normalized"],  # Prendi dalla cache
                price_data=price_data,
                sequence_length=sequence_length,
                tp_pips=80,
                sl_pips=20
            )
            
            logger.info(f"   Classification samples: {len(X_classification)}")
            logger.info(f"   TP setups: {np.sum(y_classification == 1)} ({np.mean(y_classification == 1):.1%})")
            logger.info(f"   SL setups: {np.sum(y_classification == 0)} ({np.mean(y_classification == 0):.1%})")
            
            # 4. Addestra modello UNICO con transfer learning
            logger.info("\n4. Addestramento modello unificato...")
            
            from model_trainer import LSTMTradingModel
            trainer = LSTMTradingModel(
                model_type=self.config["model"]["type"],
                model_name=f"unified_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                data_loader=loader
            )
            
            # Usa la nuova funzione di transfer learning
            results = trainer.train_single_model_with_transfer(
                X_regression=X_all,
                y_regression=y_regression,
                X_classification=X_classification,
                y_classification=y_classification,
                epochs_regression=30,  # Meno epochs per pre-training
                epochs_classification=50,  # Più epochs per fine-tuning
                freeze_base_layers=True
            )
            
            # 5. Salva modello
            logger.info("\n5. Salvataggio modello...")
            
            metadata = {
                'training_type': 'unified_transfer_learning',
                'regression_samples': len(X_all),
                'classification_samples': len(X_classification),
                'tp_sl_ratio': f"{np.sum(y_classification == 1)}:{np.sum(y_classification == 0)}",
                'final_accuracy': results['final_accuracy'],
                'tp_pips': 80,
                'sl_pips': 20,
                'rr_ratio': '4:1'
            }
            
            model_path = trainer.save_model(metadata=metadata)
            
            # 6. Aggiorna predictor
            logger.info("\n6. Configurazione predictor...")
            self.predictor = TradingPredictor(
                model_path=model_path,
                scaler_path=model_path.replace('.keras', '_scaler.pkl').replace('.h5', '_scaler.pkl'),
                sequence_length=sequence_length
            )
            
            self.system_state["predictor_ready"] = True
            
            logger.info(f"\n✅ PIPELINE UNIFICATA COMPLETATA!")
            logger.info(f"   Modello: {os.path.basename(model_path)}")
            logger.info(f"   Accuracy TP/SL: {results['final_accuracy']:.2%}")
            
            return {
                'success': True,
                'model_path': model_path,
                'accuracy': results['final_accuracy'],
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Errore pipeline unificata: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def _create_regression_targets(self, X_sequences: np.ndarray, 
                                price_data: np.ndarray,
                                sequence_length: int,
                                lookahead: int = 10) -> np.ndarray:
        """
        Crea target per regressione: movimento % futuro.
        """
        y_regression = []
        
        for i in range(len(X_sequences)):
            # Indice corrispondente nel price_data
            price_idx = i + sequence_length - 1
            
            if price_idx + lookahead < len(price_data):
                current_price = price_data[price_idx]
                future_price = price_data[price_idx + lookahead]
                
                # Movimento % normalizzato
                movement = (future_price - current_price) / current_price
                
                # Normalizza ulteriormente (max ±10%)
                movement = np.clip(movement, -0.10, 0.10)
                
                y_regression.append(movement)
            else:
                y_regression.append(0.0)
        
        return np.array(y_regression)

    def _create_tp_sl_classification_data(self, 
                                     loader,  # DataLoader
                                     df_normalized,  # DataFrame normalizzato
                                     price_data: np.ndarray,
                                     sequence_length: int,
                                     tp_pips: int = 80,
                                     sl_pips: int = 20) -> tuple:
        """
        Crea dati per classificazione TP/SL.
        """
        X_setups, y_setups = [], []
        
        pip_value = 0.0001
        buffer_ticks = max(tp_pips, sl_pips) * 15  # Buffer più grande
        
        logger.info(f"   Cercando setup TP/SL con R/R {tp_pips/sl_pips:.1f}:1")
        logger.info(f"   Buffer ticks: {buffer_ticks}")
        
        # Ottieni features dai dati normalizzati
        feature_columns = loader.feature_names if hasattr(loader, 'feature_names') else loader.feature_columns
        features_array = df_normalized[feature_columns].values
        
        total_checked = 0
        tp_count = 0
        sl_count = 0
        
        for i in range(len(features_array) - sequence_length - buffer_ticks):
            total_checked += 1
            
            # Prendi sequenza
            sequence = features_array[i:i + sequence_length]
            
            # Prezzo corrente (dai dati originali, non normalizzati)
            current_idx = i + sequence_length - 1
            if current_idx >= len(price_data):
                continue
                
            current_price = price_data[current_idx]
            
            # Analizza futuro
            future_start = current_idx + 1
            future_end = future_start + buffer_ticks
            
            if future_end >= len(price_data):
                continue
                
            future_prices = price_data[future_start:future_end]
            
            # Verifica se TP o SL viene raggiunto per primo
            tp_reached = False
            sl_reached = False
            
            for future_price in future_prices:
                # TP (assume sempre long per training)
                if future_price >= current_price + (tp_pips * pip_value):
                    tp_reached = True
                    break
                
                # SL
                if future_price <= current_price - (sl_pips * pip_value):
                    sl_reached = True
                    break
            
            # Aggiungi al dataset
            if tp_reached:
                X_setups.append(sequence)
                y_setups.append(1)  # TP Setup
                tp_count += 1
            elif sl_reached:
                X_setups.append(sequence)
                y_setups.append(0)  # SL Setup
                sl_count += 1
        
        logger.info(f"   Sequenze analizzate: {total_checked}")
        logger.info(f"   Setup trovati: {len(X_setups)} ({len(X_setups)/total_checked:.1%})")
        logger.info(f"   TP setups: {tp_count} ({tp_count/max(1, len(X_setups)):.1%})")
        logger.info(f"   SL setups: {sl_count} ({sl_count/max(1, len(X_setups)):.1%})")
        
        if len(X_setups) == 0:
            logger.warning("⚠️  Nessun setup trovato! Riducono buffer_ticks o aumentare dati")
            # Fallback: crea dati dummy per test
            return self._create_dummy_classification_data(features_array, sequence_length)
        
        return np.array(X_setups), np.array(y_setups)

    def _create_dummy_classification_data(self, features_array: np.ndarray, 
                                        sequence_length: int) -> tuple:
        """
        Crea dati dummy per test quando non si trovano setup reali.
        """
        logger.warning("⚠️  Creazione dati dummy per test...")
        
        n_samples = min(1000, len(features_array) - sequence_length)
        X_dummy = []
        y_dummy = []
        
        for i in range(n_samples):
            sequence = features_array[i:i + sequence_length]
            X_dummy.append(sequence)
            # Alterna TP/SL per bilanciamento
            y_dummy.append(1 if i % 2 == 0 else 0)
        
        logger.info(f"   Dummy samples creati: {len(X_dummy)}")
        logger.info(f"   TP: {np.sum(y_dummy == 1)}, SL: {np.sum(y_dummy == 0)}")
        
        return np.array(X_dummy), np.array(y_dummy)

def main():
    """Funzione principale."""
    orchestrator = TradingAIOrchestrator()
    orchestrator.run_full_pipeline()

if __name__ == "__main__":
    main()