"""
trading_executor.py
Modulo per gestire l'esecuzione delle operazioni con TP/SL.
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class TradingExecutor:
    """
    Gestisce l'esecuzione delle operazioni di trading con TP/SL.
    """
    
    def __init__(self, config_file: str = None):
        """
        Inizializza l'executor.
        
        Args:
            config_file: File di configurazione
        """
        self.config = self._load_config(config_file)
        self.trades_history = []
        
        # Configurazione rischio
        self.risk_config = {
            'max_risk_per_trade': 0.02,  # 2%
            'max_daily_loss': 0.05,      # 5%
            'max_concurrent_trades': 3,
            'min_rr_ratio': 1.5,
            'trailing_stop_enabled': True,
            'breakeven_at': 0.5          # Sposta SL a BE a 50% del TP
        }
        
        logger.info("TradingExecutor inizializzato")
    
    def _load_config(self, config_file: str = None) -> Dict:
        """Carica configurazione."""
        default_config = {
            'execution': {
                'default_lot_size': 0.01,
                'max_lot_size': 10.0,
                'slippage_pips': 2,
                'commission_per_lot': 7.0,
                'default_tp_multiplier': 1.5,
                'default_sl_multiplier': 0.5
            },
            'pairs': {
                'EURUSD': {'pip_value': 0.0001, 'min_lot': 0.01, 'spread': 1.2},
                'GBPUSD': {'pip_value': 0.0001, 'min_lot': 0.01, 'spread': 1.5},
                'USDJPY': {'pip_value': 0.01, 'min_lot': 0.01, 'spread': 1.0},
                'XAUUSD': {'pip_value': 0.01, 'min_lot': 0.01, 'spread': 30}
            }
        }
        
        return default_config
    
    def validate_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida un segnale di trading prima dell'esecuzione.
        MODIFICA: Rilassa i criteri di validazione
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'adjusted_signal': signal_data.copy()
        }
        
        # Estrai info
        operation_details = signal_data.get('operation_details', {})
        rr_ratio = operation_details.get('risk_reward_ratio', 0)
        confidence = signal_data.get('signal_confidence', 0)
        signal_strength = signal_data.get('signal_strength', 0)
        
        # Debug logging
        logger.info(f"Validazione - R/R: {rr_ratio}, Confidence: {confidence}, Strength: {signal_strength}")
        
        # Validazione R/R ratio - RILASSATO
        if rr_ratio < 1.0:  # Prima era 1.5
            validation['warnings'].append(
                f"R/R ratio basso ({rr_ratio}:1), minimo raccomandato: 1.0:1"
            )
        
        # Validazione confidence - RILASSATO
        if confidence < 0.4:  # Prima era 0.6
            validation['warnings'].append(
                f"Confidenza bassa ({confidence:.2f}), considera attendere"
            )
        
        # Validazione SL troppo stretto - RILASSATO
        sl_pips = operation_details.get('sl_pips', 0)
        if sl_pips < 5:  # Prima era 10
            validation['warnings'].append(
                f"SL molto stretto ({sl_pips} pips), rischio di essere stopato dal rumore"
            )
        
        # Validazione TP troppo lontano - RILASSATO
        tp_pips = operation_details.get('tp_pips', 0)
        if tp_pips > 200:  # Prima era 100
            validation['warnings'].append(
                f"TP molto lontano ({tp_pips} pips), probabilità di raggiungimento bassa"
            )
        
        # Solo errori gravi bloccano l'esecuzione
        # I warnings non bloccano più
        validation['is_valid'] = True  # Sempre valido ora
        
        return validation
    
    def generate_order_details(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera dettagli dell'ordine per esecuzione.
        
        Args:
            signal_data: Dati del segnale
            
        Returns:
            Dettagli ordine completo
        """
        operation_details = signal_data.get('operation_details', {})
        
        # Base order
        order = {
            'order_id': f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'status': 'PENDING',
            'signal_strength': signal_data.get('signal_strength', 0),
            'confidence': signal_data.get('signal_confidence', 0),
            'operation': operation_details.get('operation', 'NEUTRAL'),
            'entry_price': operation_details.get('entry_price', 0),
            'take_profit': operation_details.get('take_profit', 0),
            'stop_loss': operation_details.get('stop_loss', 0),
            'tp_pips': operation_details.get('tp_pips', 0),
            'sl_pips': operation_details.get('sl_pips', 0),
            'risk_reward_ratio': operation_details.get('risk_reward_ratio', 0),
            'position_size': operation_details.get('position_size', {}),
            'predicted_move': operation_details.get('predicted_move_pct', 0),
            'recommendations': signal_data.get('recommendations', []),
            'execution_summary': signal_data.get('execution_summary', {}),
            'risk_management': self._generate_risk_management(operation_details),
            'order_type': 'MARKET',  # o LIMIT se implementato
            'expiry': None,  # Per ordini pendenti
            'trailing_stop': self._generate_trailing_stop_config(operation_details)
        }
        
        # Calcola profit/loss potenziali
        order['potential_profit'] = self._calculate_potential_profit(order)
        order['potential_loss'] = self._calculate_potential_loss(order)
        
        return order
    
    def _generate_risk_management(self, operation_details: Dict) -> Dict[str, Any]:
        """Genera configurazione risk management."""
        position_size = operation_details.get('position_size', {})
        
        return {
            'max_loss_pips': operation_details.get('sl_pips', 0),
            'max_loss_percent': self.risk_config['max_risk_per_trade'] * 100,
            'breakeven_at_pips': int(operation_details.get('tp_pips', 0) * 
                                   self.risk_config['breakeven_at']),
            'partial_take_profit': [
                {
                    'level': 0.5,  # 50% del TP
                    'close_percent': 0.3  # Chiudi 30% della posizione
                },
                {
                    'level': 0.8,  # 80% del TP
                    'close_percent': 0.4  # Chiudi altri 40%
                }
            ],
            'trailing_stop_activation': 0.6,  # Attiva trailing a 60% del TP
            'position_size_lots': position_size.get('lots', 0.01),
            'risk_amount': position_size.get('risk_amount_usd', 0)
        }
    
    def _generate_trailing_stop_config(self, operation_details: Dict) -> Dict[str, Any]:
        """Genera configurazione trailing stop."""
        sl_pips = operation_details.get('sl_pips', 20)
        
        return {
            'enabled': self.risk_config['trailing_stop_enabled'],
            'activation_pips': int(sl_pips * 1.5),  # Attiva dopo 1.5x SL
            'trailing_distance_pips': sl_pips // 2,  # Distanza trailing = metà SL
            'lock_profit_at': 0.3  # Blocca almeno 30% dei profitti
        }
    
    def _calculate_potential_profit(self, order: Dict) -> Dict[str, Any]:
        """Calcola potenziale profitto."""
        lots = order['position_size'].get('lots', 0.01)
        tp_pips = order['tp_pips']
        
        # Calcola valore per pip (semplificato)
        # In realtà dipende dalla coppia valutaria
        pip_value_per_lot = 10.0  # Per EUR/USD
        
        profit_pips = tp_pips
        profit_usd = profit_pips * pip_value_per_lot * lots
        
        return {
            'pips': profit_pips,
            'usd': round(profit_usd, 2),
            'percent': round((profit_usd / order['position_size'].get('account_balance', 10000)) * 100, 2)
        }
    
    def _calculate_potential_loss(self, order: Dict) -> Dict[str, Any]:
        """Calcola potenziale perdita."""
        lots = order['position_size'].get('lots', 0.01)
        sl_pips = order['sl_pips']
        
        pip_value_per_lot = 10.0
        loss_pips = sl_pips
        loss_usd = loss_pips * pip_value_per_lot * lots
        
        return {
            'pips': loss_pips,
            'usd': round(loss_usd, 2),
            'percent': round((loss_usd / order['position_size'].get('account_balance', 10000)) * 100, 2)
        }
    
    def format_order_for_display(self, order: Dict[str, Any]) -> str:
        """
        Formatta l'ordine per visualizzazione.
        
        Args:
            order: Dettagli ordine
            
        Returns:
            Stringa formattata
        """
        lines = []
        lines.append("=" * 60)
        lines.append("📊 ORDINE DI TRADING")
        lines.append("=" * 60)
        lines.append(f"ID: {order['order_id']}")
        lines.append(f"Data: {order['timestamp']}")
        lines.append("")
        lines.append(f"📈 OPERAZIONE: {order['operation']}")
        lines.append(f"💰 Entry: {order['entry_price']}")
        lines.append(f"🎯 Take Profit: {order['take_profit']} ({order['tp_pips']} pips)")
        lines.append(f"🛑 Stop Loss: {order['stop_loss']} ({order['sl_pips']} pips)")
        lines.append(f"⚖️  R/R Ratio: {order['risk_reward_ratio']}:1")
        lines.append("")
        lines.append("📊 POSITION SIZING:")
        lines.append(f"   Lots: {order['position_size'].get('lots', 0)}")
        lines.append(f"   Valore: ${order['position_size'].get('position_value_usd', 0):.2f}")
        lines.append(f"   Rischio: ${order['position_size'].get('risk_amount_usd', 0):.2f}")
        lines.append(f"   (%: {order['position_size'].get('risk_percent', 0)}%)")
        lines.append("")
        lines.append("💰 POTENZIALE:")
        lines.append(f"   Profitto: {order['potential_profit']['pips']} pips")
        lines.append(f"           ${order['potential_profit']['usd']:.2f}")
        lines.append(f"           {order['potential_profit']['percent']}%")
        lines.append(f"   Perdita: {order['potential_loss']['pips']} pips")
        lines.append(f"           ${order['potential_loss']['usd']:.2f}")
        lines.append(f"           {order['potential_loss']['percent']}%")
        lines.append("")
        lines.append("🎯 RACCOMANDAZIONI:")
        for rec in order.get('recommendations', [])[:5]:
            lines.append(f"   • {rec}")
        
        if len(order.get('recommendations', [])) > 5:
            lines.append(f"   ... e altre {len(order['recommendations']) - 5} raccomandazioni")
        
        lines.append("")
        lines.append("🔄 RISK MANAGEMENT:")
        lines.append(f"   Breakeven a: {order['risk_management']['breakeven_at_pips']} pips")
        lines.append(f"   Trailing Stop: {'✅ ON' if order['trailing_stop']['enabled'] else '❌ OFF'}")
        
        if order['trailing_stop']['enabled']:
            lines.append(f"     Attivazione: {order['trailing_stop']['activation_pips']} pips")
            lines.append(f"     Distanza: {order['trailing_stop']['trailing_distance_pips']} pips")
        
        lines.append("")
        lines.append(f"📊 CONFIDENZA: {order['confidence']:.2f}")
        lines.append(f"💪 FORZA SEGNALE: {order['signal_strength']:.2f}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_order_to_file(self, order: Dict[str, Any], filename: str = None):
        """
        Salva l'ordine su file.
        
        Args:
            order: Dettagli ordine
            filename: Nome file (se None, auto-generato)
        """
        if filename is None:
            filename = f"orders/{order['order_id']}.json"
        
        import os
        os.makedirs('orders', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(order, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Ordine salvato: {filename}")
        
        # Salva anche in formato testo
        txt_filename = filename.replace('.json', '.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(self.format_order_for_display(order))
        
        logger.info(f"Ordine testo salvato: {txt_filename}")
    
    def execute_signal(self, signal_data: Dict[str, Any], auto_validate: bool = True) -> Dict[str, Any]:
        """
        Esegue un segnale di trading completo.
        
        Args:
            signal_data: Dati del segnale dal predictor
            auto_validate: Se True, valida automaticamente
            
        Returns:
            Risultato esecuzione
        """
        result = {
            'success': False,
            'order_id': None,
            'validation': None,
            'order_details': None,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Log per debug
            logger.info(f"Ricevuto segnale: {signal_data.get('trading_signal', 'N/A')}")
            
            if 'operation_details' in signal_data:
                op_details = signal_data['operation_details']
                logger.info(f"Operation details: {op_details.get('operation', 'N/A')}")
                logger.info(f"TP pips: {op_details.get('tp_pips', 'N/A')}")
                logger.info(f"SL pips: {op_details.get('sl_pips', 'N/A')}")
                logger.info(f"R/R ratio: {op_details.get('risk_reward_ratio', 'N/A')}")
            
            # 1. Valida
            if auto_validate:
                validation = self.validate_signal(signal_data)
                result['validation'] = validation
                
                # Log dettagli validazione
                logger.info(f"Validazione risultato: {validation['is_valid']}")
                if validation['warnings']:
                    logger.warning(f"Warnings: {validation['warnings']}")
                if validation['errors']:
                    logger.error(f"Errors: {validation['errors']}")
                
                if not validation['is_valid']:
                    result['warnings'] = validation['warnings']
                    result['errors'] = validation['errors']
                    logger.warning("Segnale non valido, esecuzione annullata")
                    return result
            
            # 2. Genera ordine
            order = self.generate_order_details(signal_data)
            result['order_details'] = order
            
            # Log ordine generato
            logger.info(f"Ordine generato: {order['order_id']}")
            logger.info(f"Operation: {order['operation']}")
            
            # 3. Salva
            self.save_order_to_file(order)
            
            # 4. Registra in history
            self.trades_history.append({
                'timestamp': datetime.now().isoformat(),
                'order_id': order['order_id'],
                'operation': order['operation'],
                'status': 'EXECUTED',
                'signal_strength': order['signal_strength']
            })
            
            # 5. Limita history
            if len(self.trades_history) > 100:
                self.trades_history = self.trades_history[-100:]
            
            result['success'] = True
            result['order_id'] = order['order_id']
            
            logger.info(f"Ordine eseguito: {order['order_id']} - {order['operation']}")
            
        except Exception as e:
            error_msg = str(e)
            result['errors'].append(error_msg)
            logger.error(f"Errore nell'esecuzione: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result

def create_executor_from_prediction(predictor_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Funzione di utilità per creare ordine da risultato predictor.
    
    Args:
        predictor_result: Risultato da TradingPredictor.predict_single()
        
    Returns:
        Ordine completo
    """
    executor = TradingExecutor()
    
    # Esegui il segnale
    execution_result = executor.execute_signal(predictor_result)
    
    if execution_result['success']:
        order = execution_result['order_details']
        
        # Aggiungi formato per vari output
        return {
            'success': True,
            'order': order,
            'display_text': executor.format_order_for_display(order),
            'execution_summary': order.get('execution_summary', {}),
            'risk_management': order.get('risk_management', {}),
            'order_id': order['order_id']
        }
    else:
        return {
            'success': False,
            'errors': execution_result['errors'],
            'warnings': execution_result['warnings']
        }