"""
main.py
Orchestratore principale del sistema di trading AI.
Coordina tutti i moduli: data loading, training, analysis, prediction e chat AI.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Import moduli del sistema
from data_loader import FinancialDataLoader, load_and_prepare_data
from model_trainer import LSTMTradingModel, train_model_pipeline
from pattern_analyzer import PatternAnalyzer, analyze_trading_model
from predictor import TradingPredictor, create_predictor_from_training
from deepseek_chat import DeepSeekChat, create_trading_chatbot, analyze_trading_system_integrated

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

try:
    from config import PATHS, create_directories
    # Crea directory all'avvio
    create_directories()
except ImportError:
    # Fallback se config.py non esiste
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'src':
        PROJECT_ROOT = os.path.dirname(current_dir)
    else:
        PROJECT_ROOT = current_dir
    
    PATHS = {
        'models': os.path.join(PROJECT_ROOT, 'models'),
        'data': os.path.join(PROJECT_ROOT, 'data'),
        'analysis': os.path.join(PROJECT_ROOT, 'analysis'),
        'logs': os.path.join(PROJECT_ROOT, 'logs'),
    }
    
    # Crea directory
    for dir_path in PATHS.values():
        os.makedirs(dir_path, exist_ok=True)

class TradingAIOrchestrator:
    """
    Orchestratore principale del sistema di trading AI.
    Gestisce il flusso completo: dati → training → analisi → predizione → chat.
    """
    
    def __init__(self, config_file: str = None):
        """
        Inizializza l'orchestratore.
        
        Args:
            config_file: Percorso file di configurazione JSON (opzionale)
        """
        self.config = self._load_config(config_file)
        self.data_loader = None
        self.model_trainer = None
        self.pattern_analyzer = None
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
        
        logger.info("Trading AI Orchestrator inizializzato")
    
    def _load_config(self, config_file: str = None) -> Dict:
        """
        Carica configurazione da file o usa default.
        
        Args:
            config_file: Percorso file configurazione
            
        Returns:
            Dizionario configurazione
        """
        default_config = {
            "data": {
                "sequence_length": 20,
                "test_size": 0.2,
                "target_column": "price",
                "exclude_columns": ["timestamp"]
            },
            "model": {
                "type": "bidirectional",  # standard, stacked, bidirectional
                "epochs": 100,
                "batch_size": 32,
                "model_name": f"trading_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "prediction": {
                "confidence_level": 0.95,
                "cache_predictions": True
            },
            "chat": {
                "api_key_env_var": "DEEPSEEK_API_KEY",
                "model": "deepseek-chat",
                "max_tokens": 2000,
                "temperature": 0.7
            },
            "paths": {
                "data_dir": "data",
                "models_dir": "models",
                "analysis_dir": "analysis",
                "logs_dir": "logs"
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Merge configurazioni
                import copy
                merged_config = copy.deepcopy(default_config)
                self._deep_update(merged_config, user_config)
                
                logger.info(f"Configurazione caricata da: {config_file}")
                return merged_config
                
            except Exception as e:
                logger.warning(f"Errore nel caricamento configurazione: {str(e)}. Usando default.")
                return default_config
        else:
            logger.info("Usando configurazione di default")
            return default_config
    
    def _deep_update(self, target: Dict, source: Dict):
        """Aggiorna ricorsivamente dizionario target con source."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def initialize_directories(self):
        """Inizializza le directory necessarie."""
        paths = self.config["paths"]
        
        for dir_name in paths.values():
            os.makedirs(dir_name, exist_ok=True)
        
        logger.info("Directory inizializzate")
    
    def load_data(self, data_file: str = None, data_dir: str = None) -> Dict[str, Any]:
        """
        Carica e prepara i dati finanziari.
        
        Args:
            data_file: Percorso file CSV specifico
            data_dir: Directory con file CSV (se data_file non specificato)
            
        Returns:
            Dizionario con dati preparati e informazioni
        """
        logger.info("=" * 60)
        logger.info("FASE 1: CARICAMENTO DATI")
        logger.info("=" * 60)
        
        try:
            # Determina file dati
            if data_file and os.path.exists(data_file):
                filepath = data_file
            elif data_dir:
                # Cerca file CSV nella directory
                csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                if not csv_files:
                    raise FileNotFoundError(f"Nessun file CSV trovato in {data_dir}")
                
                # Prendi il file più recente
                csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
                filepath = os.path.join(data_dir, csv_files[0])
                logger.info(f"Usando file più recente: {csv_files[0]}")
            else:
                # Usa directory di default
                default_dir = self.config["paths"]["data_dir"]
                csv_files = [f for f in os.listdir(default_dir) if f.endswith('.csv')]
                if not csv_files:
                    raise FileNotFoundError(f"Nessun file CSV trovato in {default_dir}")
                
                filepath = os.path.join(default_dir, csv_files[0])
                logger.info(f"Usando file: {csv_files[0]}")
            
            # Configura data loader
            data_config = self.config["data"]
            self.data_loader = FinancialDataLoader(
                sequence_length=data_config["sequence_length"],
                test_size=data_config["test_size"],
                random_state=42
            )
            
            # Carica e prepara dati
            logger.info(f"Caricamento dati da: {filepath}")
            results = load_and_prepare_data(
                filepath=filepath,
                sequence_length=data_config["sequence_length"],
                test_size=data_config["test_size"],
                lookahead=10
            )
            
            X_train, X_test, y_train, y_test, df_original, df_normalized, loader = results
            
            # Aggiorna stato
            self.system_state["data_loaded"] = True
            self.system_state["data_info"] = {
                "file": filepath,
                "original_samples": len(df_original),
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
                "df_original": df_original,
                "df_normalized": df_normalized,
                "loader": loader
            }
            
            logger.info(f"Dati caricati: {len(X_train)} train, {len(X_test)} test sequenze")
            logger.info(f"Feature: {loader.feature_names}")
            
            return {
                "success": True,
                "data_info": self.system_state["data_info"],
                "shapes": {
                    "X_train": X_train.shape,
                    "X_test": X_test.shape,
                    "y_train": y_train.shape,
                    "y_test": y_test.shape
                }
            }
            
        except Exception as e:
            logger.error(f"Errore nel caricamento dati: {str(e)}")
            self.system_state["data_loaded"] = False
            return {
                "success": False,
                "error": str(e)
            }
    
    def train_model(self, use_cached_data: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello di machine learning.
        
        Args:
            use_cached_data: Se True, usa dati in cache
            **kwargs: Parametri aggiuntivi per training
            
        Returns:
            Dizionario con risultati training
        """
        logger.info("=" * 60)
        logger.info("FASE 2: ADDESTRAMENTO MODELLO")
        logger.info("=" * 60)
        
        try:
            # Verifica dati disponibili
            if not self.system_state["data_loaded"] and not use_cached_data:
                raise ValueError("Dati non caricati. Esegui prima load_data()")
            
            # Recupera dati
            if use_cached_data and "data" in self.results_cache:
                data = self.results_cache["data"]
                X_train, X_test = data["X_train"], data["X_test"]
                y_train, y_test = data["y_train"], data["y_test"]
                loader = data["loader"]
            else:
                raise ValueError("Dati non disponibili in cache")
            
            # Configurazione modello
            model_config = self.config["model"]
            model_type = kwargs.get("model_type", model_config["type"])
            epochs = kwargs.get("epochs", model_config["epochs"])
            batch_size = kwargs.get("batch_size", model_config["batch_size"])
            model_name = kwargs.get("model_name", model_config["model_name"])
            
            # Addestra modello
            logger.info(f"Addestramento modello {model_type}...")
            trainer, training_results = train_model_pipeline(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                sequence_length=self.config["data"]["sequence_length"],
                n_features=X_train.shape[2],
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                model_name=model_name
            )
            
            self.model_trainer = trainer
            
            # Aggiorna stato
            self.system_state["model_trained"] = True
            self.system_state["model_info"] = {
                "model_type": model_type,
                "model_name": model_name,
                "sequence_length": self.config["data"]["sequence_length"],
                "n_features": X_train.shape[2],
                "training_date": datetime.now().isoformat(),
                "performance": training_results.get("test_metrics", {})
            }
            
            # Salva in cache
            self.results_cache["model"] = {
                "trainer": trainer,
                "training_results": training_results,
                "model_info": self.system_state["model_info"]
            }
            
            logger.info(f"Modello addestrato: {model_name}")
            logger.info(f"Performance test: MSE={training_results.get('test_metrics', {}).get('final_test_mse', 'N/A'):.6f}")
            
            return {
                "success": True,
                "model_info": self.system_state["model_info"],
                "performance": training_results.get("test_metrics", {}),
                "model_path": f"models/{model_name}.h5"
            }
            
        except Exception as e:
            logger.error(f"Errore nell'addestramento modello: {str(e)}")
            self.system_state["model_trained"] = False
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_model(self, use_cached: bool = True) -> Dict[str, Any]:
        """
        Analizza il modello addestrato.
        
        Args:
            use_cached: Se True, usa dati e modello in cache
            
        Returns:
            Dizionario con risultati analisi
        """
        logger.info("=" * 60)
        logger.info("FASE 3: ANALISI MODELLO")
        logger.info("=" * 60)
        
        try:
            # Verifica requisiti
            if not self.system_state["model_trained"] and not use_cached:
                raise ValueError("Modello non addestrato. Esegui prima train_model()")
            
            if not self.system_state["data_loaded"] and not use_cached:
                raise ValueError("Dati non caricati. Esegui prima load_data()")
            
            # Recupera dati
            if use_cached and "data" in self.results_cache and "model" in self.results_cache:
                data = self.results_cache["data"]
                model_data = self.results_cache["model"]
                
                X_train, X_test = data["X_train"], data["X_test"]
                y_train, y_test = data["y_train"], data["y_test"]
                loader = data["loader"]
                trainer = model_data["trainer"]
            else:
                raise ValueError("Dati o modello non disponibili in cache")
            
            # Analizza modello
            logger.info("Analisi pattern e comportamenti modello...")
            
            # Configura percorsi output
            analysis_dir = self.config["paths"]["analysis_dir"]
            model_name = self.system_state["model_info"]["model_name"]
            output_dir = os.path.join(analysis_dir, model_name)
            
            analyzer = analyze_trading_model(
                model_path=f"models/{model_name}.h5",
                scaler_path=f"models/{model_name}_scaler.pkl",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=loader.feature_names,
                output_dir=output_dir
            )
            
            self.pattern_analyzer = analyzer
            
            # Carica report generato
            report_file = os.path.join(output_dir, f"model_analysis_report_*.json")
            import glob
            report_files = glob.glob(report_file)
            
            analysis_report = {}
            if report_files:
                with open(report_files[0], 'r') as f:
                    analysis_report = json.load(f)
            
            # Aggiorna stato
            self.system_state["model_analyzed"] = True
            self.system_state["analysis_info"] = {
                "output_dir": output_dir,
                "report_generated": len(report_files) > 0,
                "analysis_date": datetime.now().isoformat()
            }
            
            # Salva in cache
            self.results_cache["analysis"] = {
                "analyzer": analyzer,
                "report": analysis_report
            }
            
            logger.info(f"Analisi completata. Report in: {output_dir}")
            
            return {
                "success": True,
                "analysis_info": self.system_state["analysis_info"],
                "insights": analysis_report.get("insights", {}),
                "report_path": report_files[0] if report_files else None
            }
            
        except Exception as e:
            logger.error(f"Errore nell'analisi modello: {str(e)}")
            self.system_state["model_analyzed"] = False
            return {
                "success": False,
                "error": str(e)
            }
    
    def setup_predictor(self, model_name: str = None) -> Dict[str, Any]:
        """
        Configura il predictor per fare previsioni.
        
        Args:
            model_name: Nome specifico modello (se None, usa l'ultimo)
            
        Returns:
            Dizionario con informazioni predictor
        """
        logger.info("=" * 60)
        logger.info("FASE 4: CONFIGURAZIONE PREDICTOR")
        logger.info("=" * 60)
        
        try:
            # Determina modello da usare
            if model_name:
                model_path = f"models/{model_name}.h5"
                scaler_path = f"models/{model_name}_scaler.pkl"
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Modello non trovato: {model_path}")
            else:
                # Cerca ultimo modello
                predictor = create_predictor_from_training(
                    model_dir=self.config["paths"]["models_dir"]
                )
                self.predictor = predictor
                
                # Aggiorna stato
                self.system_state["predictor_ready"] = True
                self.system_state["predictor_info"] = predictor.get_model_info()
                
                logger.info(f"Predictor configurato con modello: {predictor.model_metadata.get('model_name', 'unknown')}")
                
                return {
                    "success": True,
                    "predictor_info": self.system_state["predictor_info"],
                    "model_loaded": True
                }
            
            # Carica modello specifico
            logger.info(f"Caricamento modello: {model_name}")
            
            # Determina feature names
            feature_names = None
            if self.system_state.get("data_loaded"):
                feature_names = self.system_state["data_info"]["feature_names"]
            
            # Crea predictor
            self.predictor = TradingPredictor(
                model_path=model_path,
                scaler_path=scaler_path,
                feature_names=feature_names,
                confidence_level=self.config["prediction"]["confidence_level"]
            )
            
            # Aggiorna stato
            self.system_state["predictor_ready"] = True
            self.system_state["predictor_info"] = self.predictor.get_model_info()
            
            logger.info(f"Predictor configurato con successo")
            
            return {
                "success": True,
                "predictor_info": self.system_state["predictor_info"]
            }
            
        except Exception as e:
            logger.error(f"Errore nella configurazione predictor: {str(e)}")
            self.system_state["predictor_ready"] = False
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, input_data: Any = None, use_latest: bool = True) -> Dict[str, Any]:
        """
        Effettua una predizione.
        
        Args:
            input_data: Dati di input (se None, usa dati di test)
            use_latest: Se True e input_data=None, usa ultimi dati test
            
        Returns:
            Dizionario con risultati predizione
        """
        logger.info("=" * 60)
        logger.info("FASE 5: PREDIZIONE")
        logger.info("=" * 60)
        
        try:
            # Verifica predictor pronto
            if not self.system_state["predictor_ready"]:
                raise ValueError("Predictor non configurato. Esegui prima setup_predictor()")
            
            # Prepara dati input
            if input_data is not None:
                # Usa dati forniti
                prediction_data = input_data
                data_source = "user_provided"
            elif use_latest and "data" in self.results_cache:
                # Usa ultimi dati di test
                test_data = self.results_cache["data"]["X_test"]
                if len(test_data) > 0:
                    # Prendi ultima sequenza
                    last_sequence = test_data[-1:]
                    
                    # Converti in formato adatto (semplificato)
                    # In pratica, dovresti avere i dati originali
                    prediction_data = last_sequence
                    data_source = "test_set"
                else:
                    raise ValueError("Nessun dato di test disponibile")
            else:
                raise ValueError("Nessun dato di input fornito")
            
            # Effettua predizione
            logger.info("Effettuando predizione...")
            prediction_result = self.predictor.predict_single(
                prediction_data,
                return_confidence=True,
                use_cache=self.config["prediction"]["cache_predictions"]
            )
            
            # Salva in cache
            if "predictions" not in self.results_cache:
                self.results_cache["predictions"] = []
            
            self.results_cache["predictions"].append({
                "result": prediction_result,
                "timestamp": datetime.now().isoformat(),
                "data_source": data_source
            })
            
            # Limita cache predizioni
            max_predictions_cache = 100
            if len(self.results_cache["predictions"]) > max_predictions_cache:
                self.results_cache["predictions"] = self.results_cache["predictions"][-max_predictions_cache:]
            
            logger.info(f"Predizione completata: {prediction_result['prediction']:.4f}")
            logger.info(f"Segnale: {prediction_result['trading_signal']}")
            
            return {
                "success": True,
                "prediction": prediction_result,
                "data_source": data_source
            }
            
        except Exception as e:
            logger.error(f"Errore nella predizione: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def setup_chatbot(self, api_key: str = None) -> Dict[str, Any]:
        """
        Configura il chatbot AI per analisi conversazionale.
        
        Args:
            api_key: API key DeepSeek (se None, cerca in env var)
            
        Returns:
            Dizionario con informazioni chatbot
        """
        logger.info("=" * 60)
        logger.info("FASE 6: CONFIGURAZIONE CHATBOT AI")
        logger.info("=" * 60)
        
        try:
            # Determina API key
            if not api_key:
                api_key = os.getenv(self.config["chat"]["api_key_env_var"])
            
            if not api_key:
                logger.warning("API key DeepSeek non trovata. Chatbot in modalità simulazione.")
                # Useremo modalità simulazione
                api_key = "simulation_mode"
            
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
            self.system_state["chatbot_info"] = {
                "model": chat_config["model"],
                "api_key_available": api_key != "simulation_mode",
                "configured_at": datetime.now().isoformat()
            }
            
            logger.info(f"Chatbot configurato con modello: {chat_config['model']}")
            
            return {
                "success": True,
                "chatbot_info": self.system_state["chatbot_info"]
            }
            
        except Exception as e:
            logger.error(f"Errore nella configurazione chatbot: {str(e)}")
            self.system_state["chatbot_ready"] = False
            return {
                "success": False,
                "error": str(e)
            }
    
    def chat_about_prediction(self, prediction_result: Dict = None) -> Dict[str, Any]:
        """
        Chatta con AI per analizzare una predizione.
        
        Args:
            prediction_result: Risultato predizione (se None, usa ultima)
            
        Returns:
            Dizionario con analisi AI
        """
        logger.info("=" * 60)
        logger.info("ANALISI PREDIZIONE CON AI")
        logger.info("=" * 60)
        
        try:
            # Verifica chatbot pronto
            if not self.system_state["chatbot_ready"]:
                raise ValueError("Chatbot non configurato. Esegui prima setup_chatbot()")
            
            # Determina predizione da analizzare
            if prediction_result is None:
                if "predictions" in self.results_cache and self.results_cache["predictions"]:
                    prediction_result = self.results_cache["predictions"][-1]["result"]
                else:
                    raise ValueError("Nessuna predizione disponibile")
            
            # Analizza predizione con AI
            logger.info("Richiesta analisi AI della predizione...")
            ai_analysis = self.chatbot.explain_prediction(prediction_result)
            
            # Salva in cache
            if "ai_analyses" not in self.results_cache:
                self.results_cache["ai_analyses"] = []
            
            self.results_cache["ai_analyses"].append({
                "analysis": ai_analysis,
                "prediction_data": prediction_result,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Analisi AI completata: {len(ai_analysis['explanation'])} caratteri")
            
            return {
                "success": True,
                "ai_analysis": ai_analysis,
                "prediction_summary": {
                    "signal": prediction_result.get("trading_signal"),
                    "prediction": prediction_result.get("prediction"),
                    "confidence": prediction_result.get("signal_confidence")
                }
            }
            
        except Exception as e:
            logger.error(f"Errore nell'analisi AI: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_trading_report(self) -> Dict[str, Any]:
        """
        Genera un report di trading completo con AI.
        
        Returns:
            Dizionario con report
        """
        logger.info("=" * 60)
        logger.info("GENERAZIONE REPORT TRADING CON AI")
        logger.info("=" * 60)
        
        try:
            # Verifica chatbot pronto
            if not self.system_state["chatbot_ready"]:
                raise ValueError("Chatbot non configurato. Esegui prima setup_chatbot()")
            
            # Raccogli informazioni per report
            model_info = self.system_state.get("model_info", {})
            recent_predictions = []
            
            if "predictions" in self.results_cache:
                recent_predictions = [
                    p["result"] for p in self.results_cache["predictions"][-5:]
                ]
            
            # Genera report
            logger.info("Generazione report trading completo...")
            report = self.chatbot.generate_trading_report(
                model_info=model_info,
                recent_predictions=recent_predictions,
                market_conditions={
                    "analysis_timestamp": datetime.now().isoformat(),
                    "system_state": self.system_state
                }
            )
            
            # Salva report su file
            reports_dir = os.path.join(self.config["paths"]["analysis_dir"], "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            report_file = os.path.join(
                reports_dir,
                f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("REPORT DI TRADING AI\n")
                f.write("=" * 60 + "\n\n")
                f.write(report["report"])
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("METADATA\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generato il: {report['generation_date']}\n")
                f.write(f"Modello: {model_info.get('model_name', 'N/A')}\n")
                f.write(f"Predizioni considerate: {len(recent_predictions)}\n")
            
            logger.info(f"Report salvato in: {report_file}")
            
            return {
                "success": True,
                "report": report["report"],
                "report_file": report_file,
                "metadata": {
                    "model": model_info.get("model_name"),
                    "generation_date": report["generation_date"],
                    "predictions_analyzed": len(recent_predictions)
                }
            }
            
        except Exception as e:
            logger.error(f"Errore nella generazione report: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def interactive_chat(self):
        """
        Modalità chat interattiva con il sistema AI.
        """
        logger.info("=" * 60)
        logger.info("CHAT INTERATTIVA CON TRADING AI")
        logger.info("=" * 60)
        print("\n" + "=" * 60)
        print("CHAT INTERATTIVA TRADING AI")
        print("=" * 60)
        print("\nComandi speciali:")
        print("  /help     - Mostra comandi disponibili")
        print("  /status   - Mostra stato sistema")
        print("  /predict  - Effettua predizione")
        print("  /analyze  - Analizza ultima predizione con AI")
        print("  /report   - Genera report")
        print("  /clear    - Pulisce conversazione")
        print("  /exit     - Esci")
        print("\n" + "=" * 60)
        
        # Verifica chatbot pronto
        if not self.system_state["chatbot_ready"]:
            print("Chatbot non configurato. Configuro...")
            self.setup_chatbot()
        
        while True:
            try:
                # Input utente
                user_input = input("\nTu: ").strip()
                
                if not user_input:
                    continue
                
                # Comandi speciali
                if user_input.lower() == "/exit":
                    print("Arrivederci!")
                    break
                
                elif user_input.lower() == "/help":
                    print("\nComandi disponibili:")
                    print("  /help     - Mostra questo messaggio")
                    print("  /status   - Stato sistema e statistiche")
                    print("  /predict  - Effettua predizione con ultimi dati")
                    print("  /analyze  - Analizza ultima predizione con AI")
                    print("  /report   - Genera report trading completo")
                    print("  /clear    - Pulisce storia conversazione")
                    print("  /save     - Salva conversazione corrente")
                    print("  /exit     - Esci dal programma")
                    continue
                
                elif user_input.lower() == "/status":
                    self._print_system_status()
                    continue
                
                elif user_input.lower() == "/predict":
                    if not self.system_state["predictor_ready"]:
                        print("Predictor non configurato. Configuro...")
                        self.setup_predictor()
                    
                    print("Effettuo predizione con ultimi dati disponibili...")
                    prediction_result = self.predict()
                    
                    if prediction_result["success"]:
                        pred = prediction_result["prediction"]
                        print(f"\nPredizione: {pred['prediction']:.4f}")
                        print(f"Segnale: {pred['trading_signal']}")
                        print(f"Confidenza: {pred['signal_confidence']:.2f}")
                        print(f"Intervallo: [{pred['confidence_interval']['lower']:.4f}, "
                              f"{pred['confidence_interval']['upper']:.4f}]")
                        
                        # Chiedi se analizzare con AI
                        analyze = input("\nAnalizzare con AI? (s/n): ").lower()
                        if analyze == 's':
                            ai_result = self.chat_about_prediction(pred)
                            if ai_result["success"]:
                                print("\n" + "=" * 40)
                                print("ANALISI AI:")
                                print("=" * 40)
                                print(ai_result["ai_analysis"]["explanation"])
                    else:
                        print(f"Errore: {prediction_result.get('error')}")
                    
                    continue
                
                elif user_input.lower() == "/analyze":
                    if "predictions" not in self.results_cache or not self.results_cache["predictions"]:
                        print("Nessuna predizione disponibile. Usa prima /predict")
                        continue
                    
                    print("Analizzo ultima predizione con AI...")
                    ai_result = self.chat_about_prediction()
                    
                    if ai_result["success"]:
                        print("\n" + "=" * 40)
                        print("ANALISI AI:")
                        print("=" * 40)
                        print(ai_result["ai_analysis"]["explanation"])
                    else:
                        print(f"Errore: {ai_result.get('error')}")
                    
                    continue
                
                elif user_input.lower() == "/report":
                    print("Generazione report trading...")
                    report_result = self.generate_trading_report()
                    
                    if report_result["success"]:
                        print(f"\nReport generato: {report_result['report_file']}")
                        print("\nAnteprima report:")
                        print("-" * 40)
                        print(report_result["report"][:500] + "...")
                        print("-" * 40)
                    else:
                        print(f"Errore: {report_result.get('error')}")
                    
                    continue
                
                elif user_input.lower() == "/clear":
                    self.chatbot.clear_conversation()
                    print("Conversazione pulita")
                    continue
                
                elif user_input.lower() == "/save":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"chat_conversation_{timestamp}.json"
                    self.chatbot.save_conversation(filepath)
                    print(f"Conversazione salvata in: {filepath}")
                    continue
                
                # Chat normale
                print("AI: ", end="", flush=True)
                
                # Invio messaggio
                response = self.chatbot.chat(user_input)
                
                # Stampa risposta
                print(response["content"])
                
            except KeyboardInterrupt:
                print("\n\nInterrotto dall'utente")
                break
            except Exception as e:
                print(f"\nErrore: {str(e)}")
    
    def _print_system_status(self):
        """Stampa lo stato corrente del sistema."""
        print("\n" + "=" * 60)
        print("STATO SISTEMA TRADING AI")
        print("=" * 60)
        
        for key, value in self.system_state.items():
            if key not in ["last_update"] and not isinstance(value, dict):
                status = "✅" if value else "❌" if isinstance(value, bool) else ""
                print(f"{key.replace('_', ' ').title():20} {status} {value}")
        
        print("\nINFORMAZIONI DETTAGLIATE:")
        
        if self.system_state.get("data_loaded"):
            info = self.system_state["data_info"]
            print(f"\nDati:")
            print(f"  File: {info.get('file', 'N/A')}")
            print(f"  Campioni: {info.get('original_samples', 0)}")
            print(f"  Sequenze train: {info.get('train_samples', 0)}")
            print(f"  Sequenze test: {info.get('test_samples', 0)}")
            print(f"  Features: {', '.join(info.get('feature_names', []))}")
        
        if self.system_state.get("model_trained"):
            info = self.system_state["model_info"]
            print(f"\nModello:")
            print(f"  Nome: {info.get('model_name', 'N/A')}")
            print(f"  Tipo: {info.get('model_type', 'N/A')}")
            print(f"  Sequenze: {info.get('sequence_length', 0)}")
            print(f"  Features: {info.get('n_features', 0)}")
            
            perf = info.get("performance", {})
            if perf:
                print(f"  Performance:")
                for metric, value in perf.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.6f}")
        
        if self.system_state.get("predictor_ready"):
            print(f"\nPredictor: ✅ Pronto")
        
        if self.system_state.get("chatbot_ready"):
            info = self.system_state["chatbot_info"]
            print(f"\nChatbot:")
            print(f"  Modello: {info.get('model', 'N/A')}")
            print(f"  API Key: {'✅ Disponibile' if info.get('api_key_available') else '❌ Simulazione'}")
        
        # Statistiche cache
        print(f"\nCache:")
        for key, value in self.results_cache.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} elementi")
            elif isinstance(value, dict):
                print(f"  {key}: {len(value.keys())} elementi")
        
        print("\n" + "=" * 60)
    
    def run_full_pipeline(self, data_file: str = None):
        """
        Esegue il pipeline completo del sistema.
        CORREZIONE: Ordine corretto delle operazioni
        """
        logger.info("=" * 60)
        logger.info("AVVIO PIPELINE COMPLETO SISTEMA TRADING AI")
        logger.info("=" * 60)
        
        try:
            # 1. Inizializza directory
            self.initialize_directories()
            
            # 2. Carica dati
            data_result = self.load_data(data_file=data_file)
            if not data_result["success"]:
                logger.error("Pipeline interrotto: errore nel caricamento dati")
                return
            
            # 3. Addestra modello
            training_result = self.train_model()
            if not training_result["success"]:
                logger.error("Pipeline interrotto: errore nell'addestramento")
                return
            
            # 4. Configura predictor PRIMA dell'analisi
            logger.info("Configurazione predictor...")
            predictor_result = self.setup_predictor()
            if not predictor_result["success"]:
                logger.error("Pipeline interrotto: errore nella configurazione predictor")
                return
            
            # 5. Analizza modello (OPZIONALE - può fallire senza bloccare)
            logger.info("Analisi modello...")
            try:
                analysis_result = self.analyze_model()
                if not analysis_result["success"]:
                    logger.warning("Analisi modello non riuscita, continuo...")
            except Exception as e:
                logger.warning(f"Analisi modello fallita: {e}")
            
            # 6. Configura chatbot
            logger.info("Configurazione chatbot...")
            chatbot_result = self.setup_chatbot()
            if not chatbot_result["success"]:
                logger.warning("Chatbot non configurato correttamente, continuo...")
            
            # 7. Effettua predizione di test
            logger.info("Predizione di test...")
            if self.system_state["predictor_ready"]:
                try:
                    prediction_result = self.predict()
                    if not prediction_result["success"]:
                        logger.warning("Predizione test non riuscita")
                    else:
                        logger.info(f"Predizione test: {prediction_result['prediction']['prediction']:.4f}")
                except Exception as e:
                    logger.warning(f"Errore nella predizione: {e}")
            
            # 8. Analizza con AI (se chatbot pronto)
            if self.system_state["chatbot_ready"]:
                try:
                    ai_result = self.chat_about_prediction()
                    if ai_result["success"]:
                        logger.info("Analisi AI completata")
                except Exception as e:
                    logger.warning(f"Analisi AI fallita: {e}")
            
            # 9. Genera report (se chatbot pronto)
            if self.system_state["chatbot_ready"]:
                try:
                    report_result = self.generate_trading_report()
                    if report_result["success"]:
                        logger.info(f"Report generato: {report_result['report_file']}")
                except Exception as e:
                    logger.warning(f"Generazione report fallita: {e}")
            
            # Aggiorna stato finale
            self.system_state["initialized"] = True
            self.system_state["last_update"] = datetime.now().isoformat()
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETATO CON SUCCESSO!")
            logger.info("=" * 60)
            
            # Mostra stato finale
            self._print_system_status()
            
            return {
                "success": True,
                "pipeline_completed": True,
                "system_state": self.system_state
            }
            
        except Exception as e:
            logger.error(f"Errore nel pipeline completo: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "pipeline_completed": False
            }
    
    def save_system_state(self, filepath: str = None):
        """
        Salva lo stato del sistema su file.
        
        Args:
            filepath: Percorso file (se None, genera automaticamente)
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"system_state_{timestamp}.json"
        
        try:
            state_to_save = {
                "system_state": self.system_state,
                "config": self.config,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_to_save, f, indent=2, default=str)
            
            logger.info(f"Stato sistema salvato in: {filepath}")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio stato sistema: {str(e)}")


def main():
    """Funzione principale per esecuzione da riga di comando."""
    parser = argparse.ArgumentParser(description="Sistema di Trading AI con DeepSeek Integration")
    
    # Modalità di esecuzione
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["full", "train", "predict", "analyze", "chat", "interactive"],
                       help="Modalità di esecuzione")
    
    # Parametri dati
    parser.add_argument("--data", type=str, help="Percorso file dati CSV")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory dati")
    
    # Parametri modello
    parser.add_argument("--model-type", type=str, default="bidirectional",
                       choices=["standard", "stacked", "bidirectional"],
                       help="Tipo modello LSTM")
    parser.add_argument("--epochs", type=int, default=100, help="Epoche training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    # Parametri predizione
    parser.add_argument("--predict", action="store_true", help="Effettua predizione")
    parser.add_argument("--input-data", type=str, help="File dati per predizione")
    
    # Parametri AI
    parser.add_argument("--api-key", type=str, help="DeepSeek API Key")
    parser.add_argument("--generate-report", action="store_true", help="Genera report AI")
    
    # Configurazione
    parser.add_argument("--config", type=str, help="File configurazione JSON")
    parser.add_argument("--save-state", type=str, help="Salva stato sistema su file")
    
    args = parser.parse_args()
    
    # Crea orchestratore
    orchestrator = TradingAIOrchestrator(config_file=args.config)
    
    # Esegui in base alla modalità
    if args.mode == "full":
        # Pipeline completo
        result = orchestrator.run_full_pipeline(data_file=args.data)
        
        if args.save_state:
            orchestrator.save_system_state(args.save_state)
    
    elif args.mode == "train":
        # Solo training
        orchestrator.initialize_directories()
        data_result = orchestrator.load_data(data_file=args.data, data_dir=args.data_dir)
        
        if data_result["success"]:
            train_result = orchestrator.train_model(
                model_type=args.model_type,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            if train_result["success"]:
                print(f"\nModello addestrato: {train_result['model_info']['model_name']}")
                print(f"Performance: MSE={train_result['performance'].get('final_test_mse', 'N/A'):.6f}")
    
    elif args.mode == "predict":
        # Solo predizione
        orchestrator.initialize_directories()
        
        # Configura predictor
        predictor_result = orchestrator.setup_predictor()
        
        if predictor_result["success"]:
            if args.input_data:
                # Carica dati da file
                import pandas as pd
                input_df = pd.read_csv(args.input_data)
                prediction = orchestrator.predict(input_data=input_df)
            else:
                # Usa dati di test
                data_result = orchestrator.load_data(data_dir=args.data_dir)
                if data_result["success"]:
                    prediction = orchestrator.predict()
            
            if prediction["success"]:
                pred = prediction["prediction"]
                print(f"\nPredizione: {pred['prediction']:.4f}")
                print(f"Segnale: {pred['trading_signal']}")
                print(f"Confidenza: {pred['signal_confidence']:.2f}")
    
    elif args.mode == "analyze":
        # Analisi modello
        orchestrator.initialize_directories()
        
        # Carica dati e modello
        data_result = orchestrator.load_data(data_dir=args.data_dir)
        if data_result["success"]:
            predictor_result = orchestrator.setup_predictor()
            if predictor_result["success"]:
                analysis_result = orchestrator.analyze_model()
                
                if analysis_result["success"]:
                    print(f"\nAnalisi completata.")
                    print(f"Report: {analysis_result.get('report_path')}")
    
    elif args.mode == "chat":
        # Chat interattiva con AI
        orchestrator.initialize_directories()
        
        # Configura chatbot
        chatbot_result = orchestrator.setup_chatbot(api_key=args.api_key)
        
        if chatbot_result["success"]:
            orchestrator.interactive_chat()
    
    elif args.mode == "interactive":
        # Modalità interattiva completa
        orchestrator.initialize_directories()
        
        print("\n" + "=" * 60)
        print("SISTEMA DI TRADING AI - MODALITÀ INTERATTIVA")
        print("=" * 60)
        
        while True:
            print("\nOpzioni disponibili:")
            print("1. Carica dati")
            print("2. Addestra modello")
            print("3. Analizza modello")
            print("4. Configura predictor")
            print("5. Effettua predizione")
            print("6. Configura chatbot AI")
            print("7. Chat con AI")
            print("8. Genera report")
            print("9. Mostra stato sistema")
            print("10. Esegui pipeline completo")
            print("0. Esci")
            
            choice = input("\nScelta: ").strip()
            
            try:
                if choice == "0":
                    print("Arrivederci!")
                    break
                
                elif choice == "1":
                    data_file = input("Percorso file CSV (premi Invio per default): ").strip()
                    if data_file:
                        result = orchestrator.load_data(data_file=data_file)
                    else:
                        result = orchestrator.load_data()
                    
                    if result["success"]:
                        print(f"Dati caricati: {result['data_info']['original_samples']} campioni")
                
                elif choice == "2":
                    model_type = input(f"Tipo modello [{orchestrator.config['model']['type']}]: ").strip()
                    epochs = input(f"Epoche [{orchestrator.config['model']['epochs']}]: ").strip()
                    batch_size = input(f"Batch size [{orchestrator.config['model']['batch_size']}]: ").strip()
                    
                    kwargs = {}
                    if model_type:
                        kwargs["model_type"] = model_type
                    if epochs:
                        kwargs["epochs"] = int(epochs)
                    if batch_size:
                        kwargs["batch_size"] = int(batch_size)
                    
                    result = orchestrator.train_model(**kwargs)
                    
                    if result["success"]:
                        print(f"Modello addestrato: {result['model_info']['model_name']}")
                
                elif choice == "3":
                    result = orchestrator.analyze_model()
                    
                    if result["success"]:
                        print("Analisi completata")
                        if result.get("report_path"):
                            print(f"Report: {result['report_path']}")
                
                elif choice == "4":
                    model_name = input("Nome modello (premi Invio per ultimo): ").strip()
                    if model_name:
                        result = orchestrator.setup_predictor(model_name=model_name)
                    else:
                        result = orchestrator.setup_predictor()
                    
                    if result["success"]:
                        print("Predictor configurato")
                
                elif choice == "5":
                    result = orchestrator.predict()
                    
                    if result["success"]:
                        pred = result["prediction"]
                        print(f"Predizione: {pred['prediction']:.4f}")
                        print(f"Segnale: {pred['trading_signal']}")
                
                elif choice == "6":
                    api_key = input("API Key DeepSeek (premi Invio per env var): ").strip()
                    if not api_key:
                        api_key = None
                    
                    result = orchestrator.setup_chatbot(api_key=api_key)
                    
                    if result["success"]:
                        print("Chatbot configurato")
                
                elif choice == "7":
                    orchestrator.interactive_chat()
                
                elif choice == "8":
                    result = orchestrator.generate_trading_report()
                    
                    if result["success"]:
                        print(f"Report generato: {result['report_file']}")
                
                elif choice == "9":
                    orchestrator._print_system_status()
                
                elif choice == "10":
                    data_file = input("Percorso file dati (premi Invio per default): ").strip()
                    if data_file:
                        result = orchestrator.run_full_pipeline(data_file=data_file)
                    else:
                        result = orchestrator.run_full_pipeline()
                    
                    if result["success"]:
                        print("Pipeline completato con successo!")
                
                else:
                    print("Scelta non valida")
            
            except KeyboardInterrupt:
                print("\nOperazione interrotta")
                continue
            except Exception as e:
                print(f"\nErrore: {str(e)}")
    
    else:
        print("Modalità non riconosciuta. Usa --help per vedere le opzioni.")


if __name__ == "__main__":
    main()