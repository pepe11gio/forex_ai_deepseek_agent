"""
trading_executor.py
Gestione ordini TP/SL - Versione semplificata
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TradingExecutor:
    """Gestisce esecuzione operazioni con TP/SL."""
    
    def __init__(self):
        self.trades_history = []
        
        # Config rischio
        self.risk_config = {
            'max_risk_per_trade': 0.02,
            'min_rr_ratio': 1.0,
            'trailing_stop_enabled': True
        }
        
        logger.info("TradingExecutor inizializzato")
    
    def validate_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Valida segnale (versione semplificata)."""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Estrai info
        op_details = signal_data.get('operation_details', {})
        rr_ratio = op_details.get('risk_reward_ratio', 0)
        confidence = signal_data.get('signal_confidence', 0)
        
        # Validazioni semplici
        if rr_ratio < 0.8:
            validation['warnings'].append(f"R/R ratio basso: {rr_ratio}:1")
        
        if confidence < 0.3:
            validation['warnings'].append(f"Confidenza bassa: {confidence:.2f}")
        
        # Non blocchiamo mai, solo warnings
        validation['is_valid'] = True
        
        return validation
    
    def execute_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue segnale di trading.
        Versione semplificata: sempre successo, genera ordine.
        """
        result = {
            'success': False,
            'order_id': None,
            'order_details': None,
            'errors': []
        }
        
        try:
            # Valida
            validation = self.validate_signal(signal_data)
            
            # Genera ID ordine
            order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepara dettagli ordine
            op_details = signal_data.get('operation_details', {})
            
            order = {
                'order_id': order_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'GENERATED',
                'signal': signal_data.get('trading_signal', 'NEUTRAL'),
                'confidence': signal_data.get('signal_confidence', 0),
                'operation': op_details.get('operation', 'NEUTRAL'),
                'entry_price': op_details.get('entry_price', 0),
                'take_profit': op_details.get('take_profit', 0),
                'stop_loss': op_details.get('stop_loss', 0),
                'tp_pips': op_details.get('tp_pips', 0),
                'sl_pips': op_details.get('sl_pips', 0),
                'risk_reward_ratio': op_details.get('risk_reward_ratio', 0),
                'position_size': op_details.get('position_size', {}),
                'recommendations': signal_data.get('recommendations', [])
            }
            
            # Salva ordine
            self._save_order(order)
            
            # Registra in history
            self.trades_history.append({
                'order_id': order_id,
                'timestamp': order['timestamp'],
                'operation': order['operation'],
                'status': 'EXECUTED'
            })
            
            result['success'] = True
            result['order_id'] = order_id
            result['order_details'] = order
            result['display_text'] = self._format_order_display(order)
            
            logger.info(f"✅ Ordine generato: {order_id}")
            
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"Errore esecuzione: {e}")
        
        return result
    
    def _save_order(self, order: Dict[str, Any]):
        """Salva ordine su file."""
        import os
        
        # Crea directory orders se non esiste
        orders_dir = "orders"
        os.makedirs(orders_dir, exist_ok=True)
        
        # Salva JSON
        json_file = os.path.join(orders_dir, f"{order['order_id']}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(order, f, indent=2, ensure_ascii=False)
        
        # Salva testo
        txt_file = os.path.join(orders_dir, f"{order['order_id']}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(self._format_order_display(order))
        
        logger.debug(f"Ordine salvato: {json_file}")
    
    def _format_order_display(self, order: Dict[str, Any]) -> str:
        """Formatta ordine per visualizzazione."""
        lines = []
        lines.append("=" * 50)
        lines.append("📊 ORDINE DI TRADING")
        lines.append("=" * 50)
        lines.append(f"ID: {order['order_id']}")
        lines.append(f"Data: {order['timestamp']}")
        lines.append(f"Operazione: {order['operation']}")
        lines.append(f"Entry: {order['entry_price']}")
        lines.append(f"TP: {order['take_profit']} ({order['tp_pips']} pips)")
        lines.append(f"SL: {order['stop_loss']} ({order['sl_pips']} pips)")
        lines.append(f"R/R: {order['risk_reward_ratio']}:1")
        
        if 'position_size' in order:
            ps = order['position_size']
            lines.append(f"Lots: {ps.get('lots', 0)}")
            lines.append(f"Rischio: ${ps.get('risk_amount_usd', 0):.2f}")
        
        lines.append(f"Confidenza: {order['confidence']:.2f}")
        lines.append("=" * 50)
        
        return "\n".join(lines)

def create_executor_from_prediction(predictor_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea ordine da risultato predictor.
    """
    executor = TradingExecutor()
    execution_result = executor.execute_signal(predictor_result)
    
    if execution_result['success']:
        order = execution_result['order_details']
        
        # Aggiungi execution summary
        exec_summary = {
            'action': order['operation'],
            'entry': order['entry_price'],
            'tp': order['take_profit'],
            'sl': order['stop_loss'],
            'rr_ratio': order['risk_reward_ratio'],
            'position_size_lots': order.get('position_size', {}).get('lots', 0)
        }
        
        return {
            'success': True,
            'order': order,
            'display_text': execution_result['display_text'],
            'execution_summary': exec_summary,
            'order_id': order['order_id']
        }
    else:
        return {
            'success': False,
            'errors': execution_result['errors']
        }

if __name__ == "__main__":
    # Test
    print("Test trading_executor.py")
    
    # Dati di test
    test_signal = {
        'trading_signal': 'BUY',
        'signal_confidence': 0.75,
        'operation_details': {
            'operation': 'BUY',
            'entry_price': 1.10000,
            'take_profit': 1.10300,
            'stop_loss': 1.09800,
            'tp_pips': 30,
            'sl_pips': 20,
            'risk_reward_ratio': 1.5,
            'position_size': {
                'lots': 0.1,
                'risk_amount_usd': 20.0
            }
        },
        'recommendations': ['Test recommendation 1', 'Test 2']
    }
    
    result = create_executor_from_prediction(test_signal)
    
    if result['success']:
        print("✅ Ordine generato con successo")
        print(result['display_text'])
    else:
        print(f"❌ Errore: {result.get('errors')}")