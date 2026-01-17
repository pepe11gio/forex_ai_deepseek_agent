"""
deepseek_chat.py
Chat interattiva con DeepSeek - Versione semplificata
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekChat:
    """Chatbot per trading AI."""
    
    def __init__(self, api_key: str = None,
                 model: str = "deepseek-chat",
                 max_tokens: int = 2000,
                 temperature: float = 0.7):
        
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Storia conversazione
        self.conversation_history = []
        
        # Prompt di sistema
        self.system_prompt = """Sei un esperto assistente AI per trading algoritmico.
        Analizzi modelli ML, spieghi predizioni, fornisci raccomandazioni.
        Sii preciso ma accessibile."""
        
        self._add_message("system", self.system_prompt)
        
        logger.info(f"DeepSeekChat inizializzato")
    
    def _add_message(self, role: str, content: str):
        """Aggiunge messaggio alla storia."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def chat(self, message: str) -> Dict[str, Any]:
        """
        Invia messaggio e ottiene risposta.
        Se API key non disponibile, usa simulazione.
        """
        logger.info(f"Chat: {message[:100]}...")
        
        # Aggiungi messaggio utente
        self._add_message("user", message)
        
        # Se non c'è API key, usa simulazione
        if not self.api_key or self.api_key == "simulation":
            return self._simulate_chat(message)
        
        try:
            # Chiamata API reale
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": self.conversation_history[-10:],  # Ultimi 10 messaggi
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Aggiungi risposta
                self._add_message("assistant", content)
                
                return {
                    "content": content,
                    "tokens": data.get("usage", {}),
                    "model": self.model,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"API Error: {response.status_code}")
                return self._simulate_chat(message)
                
        except Exception as e:
            logger.error(f"Errore API: {e}")
            return self._simulate_chat(message)
    
    def _simulate_chat(self, message: str) -> Dict[str, Any]:
        """Simula risposta chatbot."""
        responses = {
            "hello": "Ciao! Sono il tuo assistente AI per trading. Come posso aiutarti?",
            "prediction": "Analizzando le predizioni del modello, vedo un segnale di BUY con confidenza moderata.",
            "analysis": "Il modello mostra buona generalizzazione con R² di 0.75. Nessun evidente overfitting.",
            "help": "Posso aiutarti con: analisi predizioni, spiegazione modelli, raccomandazioni trading.",
            "default": f"Ho ricevuto il tuo messaggio: '{message[:50]}...'. Come assistente trading AI, ti consiglio di verificare sempre TP/SL e gestire il rischio."
        }
        
        # Trova risposta appropriata
        message_lower = message.lower()
        response = responses["default"]
        
        for key, resp in responses.items():
            if key in message_lower:
                response = resp
                break
        
        # Aggiungi alla storia
        self._add_message("assistant", response)
        
        return {
            "content": f"[SIMULAZIONE] {response}",
            "tokens": {"total": 100},
            "model": "simulation",
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_prediction(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza predizione con AI."""
        prompt = f"""Analizza questa predizione di trading:

Predizione: {prediction_result.get('prediction', 'N/A')}
Segnale: {prediction_result.get('trading_signal', 'N/A')}
Confidenza: {prediction_result.get('signal_confidence', 'N/A')}
Variazione % predetta: {prediction_result.get('predicted_change_percent', 'N/A')}%

Dettagli operazione:
- Entry: {prediction_result.get('operation_details', {}).get('entry_price', 'N/A')}
- Take Profit: {prediction_result.get('operation_details', {}).get('take_profit', 'N/A')}
- Stop Loss: {prediction_result.get('operation_details', {}).get('stop_loss', 'N/A')}
- R/R Ratio: {prediction_result.get('operation_details', {}).get('risk_reward_ratio', 'N/A')}:1

Fornisci:
1. Interpretazione del segnale
2. Analisi rischio/rendimento
3. Raccomandazioni operative
4. Monitoraggio richiesto
"""
        
        response = self.chat(prompt)
        
        return {
            "explanation": response["content"],
            "prediction_data": prediction_result,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def clear_conversation(self, keep_system: bool = True):
        """Pulisce conversazione."""
        if keep_system:
            # Mantieni solo messaggio di sistema
            self.conversation_history = [self.conversation_history[0]] if self.conversation_history else []
        else:
            self.conversation_history = []
        
        logger.info("Conversazione pulita")
    
    def save_conversation(self, filepath: str):
        """Salva conversazione."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "messages": self.conversation_history,
                    "saved_at": datetime.now().isoformat(),
                    "model": self.model
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversazione salvata: {filepath}")
            
        except Exception as e:
            logger.error(f"Errore salvataggio: {e}")

def create_trading_chatbot(api_key: str = None) -> DeepSeekChat:
    """Crea chatbot per trading."""
    chatbot = DeepSeekChat(
        api_key=api_key,
        model="deepseek-chat",
        max_tokens=2000,
        temperature=0.5  # Più conservativo per trading
    )
    
    return chatbot

if __name__ == "__main__":
    # Test
    print("Test deepseek_chat.py")
    
    # Crea chatbot
    chatbot = create_trading_chatbot()
    
    # Test chat
    response = chatbot.chat("Ciao, puoi spiegarmi cos'è il risk management nel trading?")
    print(f"Risposta: {response['content'][:100]}...")
    
    # Test analisi predizione
    test_prediction = {
        "prediction": 1.1025,
        "trading_signal": "BUY",
        "signal_confidence": 0.72,
        "predicted_change_percent": 2.3,
        "operation_details": {
            "entry_price": 1.1000,
            "take_profit": 1.1030,
            "stop_loss": 1.0980,
            "risk_reward_ratio": 1.5
        }
    }
    
    analysis = chatbot.analyze_prediction(test_prediction)
    print(f"\nAnalisi predizione: {analysis['explanation'][:150]}...")
    
    print("✅ Test completato")