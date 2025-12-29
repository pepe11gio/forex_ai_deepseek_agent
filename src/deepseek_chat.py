"""
deepseek_chat.py
Modulo per l'integrazione conversazionale con DeepSeek API.
Permette di interrogare il modello di trading e ottenere spiegazioni in linguaggio naturale.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import time

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRole(Enum):
    """Ruoli nella conversazione."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Messaggio nella conversazione."""
    role: ChatRole
    content: str
    
    def to_dict(self) -> Dict:
        """Converte in formato API."""
        return {"role": self.role.value, "content": self.content}


class DeepSeekChat:
    """
    Classe per gestire la conversazione con DeepSeek API.
    Integra analisi del modello di trading con spiegazioni in linguaggio naturale.
    """
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat",
                 max_tokens: int = 2000,
                 temperature: float = 0.7):
        """
        Inizializza il client DeepSeek.
        
        Args:
            api_key: API key per DeepSeek (se None, cerca in env var DEEPSEEK_API_KEY)
            base_url: URL base dell'API
            model: Modello da utilizzare
            max_tokens: Massimo token per risposta
            temperature: Temperature per generazione (0.0-1.0)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key non fornita. Imposta DEEPSEEK_API_KEY come env var o passa api_key"
            )
        
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Storia conversazione
        self.conversation_history: List[Message] = []
        
        # Prompt di sistema per trading AI
        self.system_prompt = self._create_system_prompt()
        self._add_system_message(self.system_prompt)
        
        # Cache per performance
        self.response_cache = {}
        self.cache_ttl = 300  # 5 minuti
        
        # Statistiche
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "errors": 0
        }
        
        logger.info(f"DeepSeekChat inizializzato con modello: {model}")
    
    def _create_system_prompt(self) -> str:
        """Crea il prompt di sistema per l'assistente trading AI."""
        return """Sei un esperto assistente AI per trading algoritmico. Il tuo compito è:

1. ANALIZZARE modelli di machine learning per trading
2. SPIEGARE in linguaggio semplice concetti complessi
3. INTERPRETARE segnali e previsioni
4. FORNIRE raccomandazioni basate su dati
5. IDENTIFICARE rischi e opportunità

Competenze richieste:
- Machine Learning per trading (LSTM, Random Forest, XGBoost, etc.)
- Analisi tecnica e indicatori (RSI, MACD, Bollinger Bands, etc.)
- Gestione del rischio e money management
- Python per analisi dati (pandas, numpy, scikit-learn, TensorFlow)
- Statistica e matematica finanziaria

Linee guida:
- Sii preciso ma accessibile
- Fornisci spiegazioni sia qualitative che quantitative
- Indica sempre il livello di confidenza/rischio
- Suggerisci verifiche aggiuntive quando necessario
- Mantieni un approccio equilibrato (né troppo ottimista né pessimistico)
- Cita metriche e dati specifici quando disponibili

Formato risposte:
1. Sintesi iniziale (1-2 frasi)
2. Analisi dettagliata con dati
3. Interpretazione e significato
4. Raccomandazioni pratiche
5. Avvertenze e limiti

Se ti vengono forniti dati specifici (predizioni, metriche, segnali), analizzali in dettaglio.
Se non hai informazioni sufficienti, chiedi chiarimenti.
"""
    
    def _add_system_message(self, content: str):
        """Aggiunge un messaggio di sistema alla conversazione."""
        message = Message(role=ChatRole.SYSTEM, content=content)
        self.conversation_history.append(message)
    
    def _add_user_message(self, content: str):
        """Aggiunge un messaggio dell'utente alla conversazione."""
        message = Message(role=ChatRole.USER, content=content)
        self.conversation_history.append(message)
    
    def _add_assistant_message(self, content: str):
        """Aggiunge un messaggio dell'assistente alla conversazione."""
        message = Message(role=ChatRole.ASSISTANT, content=content)
        self.conversation_history.append(message)
    
    def _prepare_api_payload(self, messages: List[Dict]) -> Dict:
        """Prepara il payload per l'API."""
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
    
    def _call_api(self, payload: Dict) -> Dict:
        """
        Chiama l'API DeepSeek.
        
        Args:
            payload: Payload della richiesta
            
        Returns:
            Risposta dell'API
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            self.stats["total_requests"] += 1
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                self.stats["errors"] += 1
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "Timeout chiamata API"
            logger.error(error_msg)
            self.stats["errors"] += 1
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Request Error: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"] += 1
            raise Exception(error_msg)
    
    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Genera una chiave di cache per i messaggi."""
        import hashlib
        messages_str = json.dumps(messages, sort_keys=True)
        return hashlib.md5(messages_str.encode()).hexdigest()
    
    def _clean_cache(self):
        """Pulisce la cache da voci scadute."""
        current_time = time.time()
        expired_keys = [
            key for key, (timestamp, _) in self.response_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.response_cache[key]
    
    def chat(self, 
             message: str,
             use_cache: bool = True,
             max_history: int = 10) -> Dict[str, Any]:
        """
        Invia un messaggio all'API e ottiene una risposta.
        
        Args:
            message: Messaggio dell'utente
            use_cache: Se True, usa cache per domande simili
            max_history: Massimo messaggi di storia da includere
            
        Returns:
            Dizionario con risposta e metadati
        """
        logger.info(f"Nuovo messaggio utente: {message[:100]}...")
        
        # Aggiungi messaggio utente alla storia
        self._add_user_message(message)
        
        # Prepara messaggi per API (ultimi N)
        recent_messages = [
            msg.to_dict() for msg in self.conversation_history[-max_history:]
        ]
        
        # Controlla cache
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(recent_messages)
            self._clean_cache()
            
            if cache_key in self.response_cache:
                _, cached_response = self.response_cache[cache_key]
                self.stats["cache_hits"] += 1
                logger.info("Risposta recuperata da cache")
                
                # Aggiungi risposta cache alla storia
                self._add_assistant_message(cached_response["content"])
                
                return cached_response
        
        # Chiama API
        payload = self._prepare_api_payload(recent_messages)
        api_response = self._call_api(payload)
        
        # Estrai contenuto risposta
        if "choices" in api_response and len(api_response["choices"]) > 0:
            response_content = api_response["choices"][0]["message"]["content"]
            
            # Calcola token (se disponibili)
            usage = api_response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens
            
            self.stats["total_tokens"] += total_tokens
            
            # Costruisci risposta
            response_data = {
                "content": response_content,
                "model": api_response.get("model", self.model),
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens
                },
                "finish_reason": api_response["choices"][0].get("finish_reason", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }
            
            # Salva in cache
            if use_cache and cache_key:
                self.response_cache[cache_key] = (time.time(), response_data)
            
            # Aggiungi risposta alla storia
            self._add_assistant_message(response_content)
            
            logger.info(f"Risposta ricevuta ({total_tokens} tokens)")
            
            return response_data
        else:
            error_msg = "Risposta API malformata"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def analyze_model_performance(self, 
                                 performance_metrics: Dict[str, Any],
                                 feature_importance: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analizza le performance del modello con DeepSeek.
        
        Args:
            performance_metrics: Metriche di performance del modello
            feature_importance: DataFrame con importanza features
            
        Returns:
            Analisi in linguaggio naturale
        """
        logger.info("Analisi performance modello con AI...")
        
        # Prepara prompt specifico
        prompt = f"""Analizza le performance di questo modello di trading AI:

METRICHE PERFORMANCE:
{json.dumps(performance_metrics, indent=2)}

"""
        
        if feature_importance is not None:
            prompt += f"""
IMPORTANZA FEATURES (top 10):
{feature_importance.head(10).to_string()}

"""
        
        prompt += """
Per favore fornisci:
1. Valutazione complessiva delle performance
2. Punti di forza e debolezze
3. Rischio di overfitting/underfitting
4. Suggerimenti per miglioramento
5. Praticabilità per trading reale
"""

        response = self.chat(prompt, use_cache=False)
        
        return {
            "analysis": response["content"],
            "metrics_provided": performance_metrics,
            "analysis_type": "model_performance"
        }
    
    def explain_prediction(self, 
                          prediction_result: Dict[str, Any],
                          market_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Spiega una predizione del modello in linguaggio naturale.
        
        Args:
            prediction_result: Risultato della predizione dal predictor
            market_context: Contesto di mercato opzionale
            
        Returns:
            Spiegazione in linguaggio naturale
        """
        logger.info("Spiegazione predizione con AI...")
        
        # Prepara prompt
        prompt = f"""Spiega questa predizione di trading AI:

RISULTATO PREDIZIONE:
- Valore predetto: {prediction_result.get('prediction', 'N/A')}
- Segnale: {prediction_result.get('trading_signal', 'N/A')}
- Forza segnale: {prediction_result.get('signal_strength', 'N/A')}
- Confidence: {prediction_result.get('signal_confidence', 'N/A')}
- Cambio % predetto: {prediction_result.get('predicted_change_percent', 'N/A')}%

"""
        
        if 'confidence_interval' in prediction_result:
            ci = prediction_result['confidence_interval']
            if ci:
                prompt += f"- Intervallo confidenza ({ci.get('confidence_level', 0.95)*100}%): "
                prompt += f"[{ci.get('lower', 'N/A'):.4f}, {ci.get('upper', 'N/A'):.4f}]\n"
                prompt += f"- Posizione nell'intervallo: {ci.get('position_in_ci', 'N/A'):.1f}%\n"
        
        if market_context:
            prompt += f"\nCONTESTO MERCATO:\n{json.dumps(market_context, indent=2)}\n"
        
        prompt += f"""
RACCOMANDAZIONI AUTOMATICHE:
{chr(10).join(prediction_result.get('recommendations', ['Nessuna raccomandazione']))}

Per favore fornisci:
1. Interpretazione del segnale di trading
2. Analisi rischio/rendimento
3. Fattori che potrebbero influenzare la predizione
4. Azioni consigliate (entry, exit, stop-loss)
5. Monitoraggio richiesto
6. Alternative da considerare
"""

        response = self.chat(prompt, use_cache=False)
        
        return {
            "explanation": response["content"],
            "prediction_data": prediction_result,
            "explanation_type": "prediction_analysis"
        }
    
    def analyze_trading_pattern(self, 
                               pattern_data: Dict[str, Any],
                               historical_context: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analizza pattern di trading identificati dal modello.
        
        Args:
            pattern_data: Dati del pattern da analizzare
            historical_context: Contesto storico opzionale
            
        Returns:
            Analisi pattern in linguaggio naturale
        """
        logger.info("Analisi pattern trading con AI...")
        
        # Prepara prompt
        prompt = f"""Analizza questo pattern di trading identificato dal modello AI:

DATI PATTERN:
{json.dumps(pattern_data, indent=2)}

"""
        
        if historical_context is not None:
            prompt += f"\nCONTESTO STORICO (ultime statistiche):\n"
            prompt += f"- Periodo: {len(historical_context)} campioni\n"
            
            if 'price' in historical_context.columns:
                prices = historical_context['price']
                prompt += f"- Prezzo medio: {prices.mean():.4f}\n"
                prompt += f"- Volatilità (std): {prices.std():.4f}\n"
                prompt += f"- Range: [{prices.min():.4f}, {prices.max():.4f}]\n"
            
            prompt += "\n"
        
        prompt += """
Per favore fornisci:
1. Identificazione tipo di pattern
2. Significato nel contesto attuale
3. Probabilità di successo
4. Gestione rischio appropriata
5. Condizioni di ingresso/uscita
6. Confronto con pattern storici simili
"""

        response = self.chat(prompt, use_cache=False)
        
        return {
            "pattern_analysis": response["content"],
            "pattern_data": pattern_data,
            "analysis_type": "pattern_analysis"
        }
    
    def generate_trading_report(self,
                               model_info: Dict[str, Any],
                               recent_predictions: List[Dict[str, Any]],
                               market_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Genera un report di trading completo con AI.
        
        Args:
            model_info: Informazioni sul modello
            recent_predictions: Predizioni recenti
            market_conditions: Condizioni di mercato
            
        Returns:
            Report di trading in linguaggio naturale
        """
        logger.info("Generazione report trading con AI...")
        
        # Calcola statistiche predizioni
        if recent_predictions:
            predictions = [p.get('prediction') for p in recent_predictions 
                          if p.get('prediction') is not None]
            signals = [p.get('trading_signal') for p in recent_predictions 
                      if p.get('trading_signal')]
            
            prediction_stats = {
                "total_predictions": len(recent_predictions),
                "avg_prediction": np.mean(predictions) if predictions else 0,
                "std_prediction": np.std(predictions) if len(predictions) > 1 else 0,
                "signal_distribution": {
                    signal: signals.count(signal) for signal in set(signals)
                }
            }
        else:
            prediction_stats = {"total_predictions": 0}
        
        # Prepara prompt
        prompt = f"""Genera un report di trading professionale basato su:

INFORMAZIONI MODELLO:
- Tipo: {model_info.get('model_type', 'N/A')}
- Features: {model_info.get('n_features', 'N/A')}
- Sequenze: {model_info.get('sequence_length', 'N/A')} tick
- Parametri: {model_info.get('total_params', 'N/A')}

STATISTICHE PREDIZIONI RECENTI:
{json.dumps(prediction_stats, indent=2)}

"""
        
        if market_conditions:
            prompt += f"\nCONDIZIONI MERCATO:\n{json.dumps(market_conditions, indent=2)}\n"
        
        if recent_predictions:
            prompt += "\nPREDIZIONI DETTAGLIATE (ultime 3):\n"
            for i, pred in enumerate(recent_predictions[-3:], 1):
                prompt += f"{i}. {pred.get('trading_signal', 'N/A')} "
                prompt += f"(pred: {pred.get('prediction', 'N/A'):.4f}, "
                prompt += f"conf: {pred.get('signal_confidence', 'N/A'):.2f})\n"
            prompt += "\n"
        
        prompt += """
Per favore genera un report strutturato che includa:
1. ESECUTIVE SUMMARY (1 paragrafo)
2. PERFORMANCE MODELLO (metriche, stabilità, affidabilità)
3. ANALISI SEGNALI (tendenza, consistenza, forza)
4. CONTESTO MERCATO (volatilità, trend, rischi)
5. RACCOMANDAZIONI OPERATIVE (strategie, posizionamento)
6. GESTIONE RISCHIO (stop-loss, sizing, diversificazione)
7. MONITORAGGIO (cosa osservare, trigger di allerta)
8. PROSPETTIVE (scenari, opportunità, minacce)

Formato professionale per trader istituzionali.
"""

        response = self.chat(prompt, use_cache=False)
        
        return {
            "report": response["content"],
            "model_info": model_info,
            "prediction_stats": prediction_stats,
            "report_type": "trading_report",
            "generation_date": datetime.now().isoformat()
        }
    
    def ask_trading_question(self, 
                            question: str,
                            context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Risponde a domande generiche sul trading con contesto opzionale.
        
        Args:
            question: Domanda dell'utente
            context_data: Dati di contesto opzionali
            
        Returns:
            Risposta alla domanda
        """
        logger.info(f"Domanda trading: {question[:100]}...")
        
        prompt = question
        
        if context_data:
            prompt += f"\n\nCONTESTO AGGIUNTIVO:\n{json.dumps(context_data, indent=2)}"
        
        prompt += "\n\nPer favore fornisci una risposta completa basata su best practices di trading."
        
        response = self.chat(prompt)
        
        return {
            "answer": response["content"],
            "question": question,
            "context_provided": context_data is not None,
            "answer_type": "trading_qa"
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Ottiene un riepilogo della conversazione corrente.
        
        Returns:
            Riepilogo conversazione
        """
        # Conta messaggi per ruolo
        role_counts = {}
        for msg in self.conversation_history:
            role = msg.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Estrai temi principali (semplificato)
        user_messages = [
            msg.content for msg in self.conversation_history 
            if msg.role == ChatRole.USER
        ]
        
        # Analizza temi con AI
        if len(user_messages) > 0:
            themes_prompt = f"Identifica i 3-5 temi principali di questa conversazione:\n\n"
            themes_prompt += "\n".join([f"- {msg[:200]}..." for msg in user_messages[-5:]])
            
            try:
                themes_response = self.chat(themes_prompt, use_cache=False)
                themes_analysis = themes_response["content"]
            except:
                themes_analysis = "Analisi temi non disponibile"
        else:
            themes_analysis = "Nessun messaggio utente"
        
        return {
            "total_messages": len(self.conversation_history),
            "role_counts": role_counts,
            "themes_analysis": themes_analysis,
            "stats": self.stats.copy(),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_conversation(self, keep_system: bool = True):
        """
        Pulisce la storia della conversazione.
        
        Args:
            keep_system: Se True, mantiene il messaggio di sistema
        """
        if keep_system and self.conversation_history:
            system_messages = [
                msg for msg in self.conversation_history 
                if msg.role == ChatRole.SYSTEM
            ]
            self.conversation_history = system_messages
        else:
            self.conversation_history = []
            if keep_system:
                self._add_system_message(self.system_prompt)
        
        logger.info("Conversazione pulita")
    
    def save_conversation(self, filepath: str):
        """
        Salva la conversazione corrente in un file.
        
        Args:
            filepath: Percorso del file di output
        """
        try:
            conversation_data = {
                "messages": [
                    {"role": msg.role.value, "content": msg.content, "timestamp": datetime.now().isoformat()}
                    for msg in self.conversation_history
                ],
                "stats": self.stats,
                "model": self.model,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversazione salvata in: {filepath}")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio conversazione: {str(e)}")
            raise
    
    def load_conversation(self, filepath: str):
        """
        Carica una conversazione da file.
        
        Args:
            filepath: Percorso del file da caricare
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Pulisci conversazione corrente
            self.conversation_history = []
            
            # Ricostruisci messaggi
            for msg_data in conversation_data.get("messages", []):
                role = ChatRole(msg_data["role"])
                message = Message(role=role, content=msg_data["content"])
                self.conversation_history.append(message)
            
            # Aggiorna stats
            self.stats.update(conversation_data.get("stats", {}))
            
            logger.info(f"Conversazione caricata da: {filepath}")
            
        except Exception as e:
            logger.error(f"Errore nel caricamento conversazione: {str(e)}")
            raise


# Funzioni di utilità per l'integrazione con il sistema di trading
def create_trading_chatbot(api_key: str = None) -> DeepSeekChat:
    """
    Crea un chatbot specializzato per trading.
    
    Args:
        api_key: API key DeepSeek
        
    Returns:
        Istanza DeepSeekChat configurata per trading
    """
    chatbot = DeepSeekChat(
        api_key=api_key,
        model="deepseek-chat",
        max_tokens=3000,
        temperature=0.5  # Più conservativo per trading
    )
    
    return chatbot


def analyze_trading_system_integrated(model_trainer: Any,
                                     predictor: Any,
                                     recent_data: pd.DataFrame,
                                     api_key: str = None) -> Dict[str, Any]:
    """
    Analisi integrata completa del sistema di trading.
    
    Args:
        model_trainer: Istanze di model_trainer
        predictor: Istanza di predictor
        recent_data: Dati recenti per analisi
        api_key: API key DeepSeek
        
    Returns:
        Analisi completa in linguaggio naturale
    """
    logger.info("Avvio analisi integrata sistema trading...")
    
    # Crea chatbot
    chatbot = create_trading_chatbot(api_key)
    
    # Raccogli informazioni
    model_info = predictor.get_model_info()
    
    # Fai predizioni sui dati recenti
    recent_predictions = []
    try:
        if len(recent_data) >= predictor.sequence_length:
            # Prendi ultime N sequenze
            n_sequences = min(5, len(recent_data) - predictor.sequence_length + 1)
            
            for i in range(n_sequences):
                start_idx = len(recent_data) - predictor.sequence_length - i
                end_idx = len(recent_data) - i
                if start_idx >= 0:
                    sequence_data = recent_data.iloc[start_idx:end_idx]
                    prediction = predictor.predict_single(
                        sequence_data,
                        return_confidence=True
                    )
                    recent_predictions.append(prediction)
    except Exception as e:
        logger.warning(f"Impossibile generare predizioni per analisi: {str(e)}")
    
    # Genera report
    report = chatbot.generate_trading_report(
        model_info=model_info,
        recent_predictions=recent_predictions,
        market_conditions={
            "data_points": len(recent_data),
            "analysis_date": datetime.now().isoformat(),
            "data_freshness": "recent" if len(recent_data) > 0 else "unknown"
        }
    )
    
    # Ottieni riepilogo
    summary = chatbot.get_conversation_summary()
    
    return {
        "trading_report": report["report"],
        "model_analysis": f"Modello: {model_info.get('model_type', 'unknown')} con {model_info.get('n_features', 'unknown')} features",
        "recent_signals": [
            f"{p.get('trading_signal', 'N/A')} (conf: {p.get('signal_confidence', 0):.2f})"
            for p in recent_predictions[-3:]
        ],
        "conversation_summary": summary,
        "analysis_timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test del modulo DeepSeek Chat
    print("Test del deepseek_chat.py")
    print("=" * 50)
    
    try:
        # Controlla se API key è disponibile
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            print("\n⚠️  ATTENZIONE: DEEPSEEK_API_KEY non trovata nelle env var")
            print("   Il test userà una simulazione dell'API")
            print("   Per test reali, esporta la tua API key:")
            print("   export DEEPSEEK_API_KEY='la_tua_api_key'")
            
            # Modalità simulazione per test
            class MockDeepSeekChat(DeepSeekChat):
                def _call_api(self, payload):
                    # Simula risposta API
                    return {
                        "choices": [{
                            "message": {
                                "content": f"[SIMULAZIONE] Risposta per: {payload['messages'][-1]['content'][:50]}..."
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 200,
                            "total_tokens": 300
                        },
                        "model": "deepseek-chat"
                    }
            
            # Usa classe mock
            chat = MockDeepSeekChat(api_key="mock_key")
            print("\nModalità simulazione attivata")
            
        else:
            # Usa API reale
            chat = DeepSeekChat(api_key=api_key)
            print(f"\nConnesso a DeepSeek API con modello: {chat.model}")
        
        # Test chat base
        print("\nTest chat base...")
        response = chat.chat("Ciao, puoi spiegarmi cos'è un LSTM nel contesto del trading?")
        print(f"Risposta: {response['content'][:200]}...")
        
        # Test analisi performance modello
        print("\nTest analisi performance modello...")
        mock_metrics = {
            "train_mse": 0.0012,
            "val_mse": 0.0015,
            "train_mae": 0.025,
            "val_mae": 0.028,
            "r2_score": 0.78,
            "overfitting_ratio": 1.25
        }
        
        mock_features = pd.DataFrame({
            'feature': ['price', 'rsi', 'macd', 'volume', 'volatility'],
            'importance': [0.35, 0.25, 0.20, 0.15, 0.05]
        })
        
        analysis = chat.analyze_model_performance(mock_metrics, mock_features)
        print(f"Analisi: {analysis['analysis'][:200]}...")
        
        # Test spiegazione predizione
        print("\nTest spiegazione predizione...")
        mock_prediction = {
            "prediction": 102.45,
            "trading_signal": "STRONG_BUY",
            "signal_strength": 0.85,
            "signal_confidence": 0.72,
            "predicted_change_percent": 2.3,
            "confidence_interval": {
                "lower": 101.20,
                "upper": 103.70,
                "confidence_level": 0.95,
                "position_in_ci": 65.2
            },
            "recommendations": [
                "Considera posizione long",
                "Imposta stop-loss a 100.50",
                "Monitora resistenza a 103.00"
            ]
        }
        
        explanation = chat.explain_prediction(mock_prediction)
        print(f"Spiegazione: {explanation['explanation'][:200]}...")
        
        # Test report trading
        print("\nTest generazione report trading...")
        mock_model_info = {
            "model_type": "LSTM Bidirezionale",
            "n_features": 8,
            "sequence_length": 20,
            "total_params": 14520
        }
        
        mock_predictions = [
            {"prediction": 102.45, "trading_signal": "BUY", "signal_confidence": 0.72},
            {"prediction": 101.80, "trading_signal": "NEUTRAL", "signal_confidence": 0.45},
            {"prediction": 103.10, "trading_signal": "STRONG_BUY", "signal_confidence": 0.85}
        ]
        
        report = chat.generate_trading_report(mock_model_info, mock_predictions)
        print(f"Report generato: {len(report['report'])} caratteri")
        
        # Test riepilogo conversazione
        print("\nTest riepilogo conversazione...")
        summary = chat.get_conversation_summary()
        print(f"Messaggi totali: {summary['total_messages']}")
        print(f"Statistiche: {summary['stats']}")
        
        # Salva conversazione
        chat.save_conversation("test_conversation.json")
        print("\nConversazione salvata in test_conversation.json")
        
        print("\n" + "=" * 50)
        print("TEST COMPLETATO CON SUCCESSO!")
        print("=" * 50)
        
        # Pulisci file test
        import os
        if os.path.exists("test_conversation.json"):
            os.remove("test_conversation.json")
            print("\nFile di test puliti")
        
    except Exception as e:
        print(f"\nErrore durante il test: {str(e)}")
        import traceback
        traceback.print_exc()