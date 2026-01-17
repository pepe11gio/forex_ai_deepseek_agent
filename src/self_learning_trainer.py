"""
self_learning_trainer.py
Training con validazione retrospettiva e pattern recognition
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class SelfLearningTrainer:
    """
    Sistema di training che:
    1. Fa predizioni su tutte le sequenze note
    2. Verifica se le predizioni erano corrette
    3. Identifica pattern ricorrenti negli errori
    4. Ri-addestra con focus sugli errori
    """
    
    def __init__(self, predictor, data_loader, confidence_threshold: float = 0.7):
        self.predictor = predictor
        self.data_loader = data_loader
        self.confidence_threshold = confidence_threshold
        self.error_patterns = defaultdict(list)
        self.correct_patterns = defaultdict(list)
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'wrong_predictions': 0,
            'tp_reached': 0,
            'sl_reached': 0,
            'win_rate': 0.0
        }
    
    def analyze_historical_predictions(self, X_data: np.ndarray, 
                                       price_data: np.ndarray,
                                       tp_pips: int = 30, 
                                       sl_pips: int = 20) -> Dict[str, Any]:
        """
        Analizza tutte le sequenze storiche e valuta le predizioni.
        """
        logger.info("=" * 60)
        logger.info("ANALISI PREDIZIONI STORICHE")
        logger.info("=" * 60)
        
        results = []
        pip_value = 0.0001
        
        for i in range(len(X_data) - 100):  # Lascia buffer per future prices
            sequence = X_data[i:i+1]  # Batch di 1
            
            # Fai predizione
            prediction_result = self.predictor.predict_single(
                sequence, 
                return_confidence=True
            )
            
            # Skip predizioni neutrali o low confidence
            signal = prediction_result.get('trading_signal', 'NEUTRAL')
            confidence = prediction_result.get('signal_confidence', 0)
            
            if signal == 'NEUTRAL' or confidence < self.confidence_threshold:
                continue
            
            # Verifica risultato reale
            current_idx = i + self.predictor.sequence_length - 1
            current_price = price_data[current_idx]
            
            future_prices = price_data[current_idx + 1:current_idx + 1 + 100]
            
            tp_reached = False
            sl_reached = False
            tp_sl_hit = None
            
            for j, future_price in enumerate(future_prices):
                if signal == 'BUY':
                    if future_price >= current_price + (tp_pips * pip_value):
                        tp_reached = True
                        tp_sl_hit = 'TP'
                        break
                    elif future_price <= current_price - (sl_pips * pip_value):
                        sl_reached = True
                        tp_sl_hit = 'SL'
                        break
                else:  # SELL
                    if future_price <= current_price - (tp_pips * pip_value):
                        tp_reached = True
                        tp_sl_hit = 'TP'
                        break
                    elif future_price >= current_price + (sl_pips * pip_value):
                        sl_reached = True
                        tp_sl_hit = 'SL'
                        break
            
            # Valuta se la predizione era corretta
            prediction_correct = False
            if tp_sl_hit == 'TP' and signal in ['BUY', 'SELL']:
                prediction_correct = True
                self.performance_stats['tp_reached'] += 1
            elif tp_sl_hit == 'SL':
                prediction_correct = False
                self.performance_stats['sl_reached'] += 1
            
            # Aggiorna statistiche
            self.performance_stats['total_predictions'] += 1
            if prediction_correct:
                self.performance_stats['correct_predictions'] += 1
            else:
                self.performance_stats['wrong_predictions'] += 1
            
            # Salva risultato per pattern analysis
            result = {
                'index': i,
                'signal': signal,
                'confidence': confidence,
                'prediction_correct': prediction_correct,
                'tp_sl_hit': tp_sl_hit,
                'current_price': current_price,
                'sequence_features': self._extract_sequence_features(sequence)
            }
            results.append(result)
            
            # Raccogli pattern per apprendimento
            if prediction_correct:
                self._record_correct_pattern(sequence, signal)
            else:
                self._record_error_pattern(sequence, signal, tp_sl_hit)
        
        # Calcola win rate
        if self.performance_stats['total_predictions'] > 0:
            self.performance_stats['win_rate'] = (
                self.performance_stats['correct_predictions'] / 
                self.performance_stats['total_predictions']
            )
        
        return {
            'results': results,
            'stats': self.performance_stats,
            'error_patterns': dict(self.error_patterns),
            'correct_patterns': dict(self.correct_patterns)
        }
    
    def _extract_sequence_features(self, sequence: np.ndarray) -> Dict[str, float]:
        """Estrae caratteristiche chiave dalla sequenza."""
        seq = sequence[0]  # Rimuovi dimensione batch
        
        features = {
            'mean_price': float(np.mean(seq[:, 0])),
            'std_price': float(np.std(seq[:, 0])),
            'trend_slope': float(self._calculate_trend_slope(seq[:, 0])),
            'bb_width': float(np.mean(seq[:, 1] - seq[:, 3])),  # BB_UPPER - BB_LOWER
            'rsi_mean': float(np.mean(seq[:, 7])),
            'rsi_oversold': float(np.sum(seq[:, 7] < 30)),
            'rsi_overbought': float(np.sum(seq[:, 7] > 70)),
            'macd_cross': float(self._check_macd_cross(seq[:, 4], seq[:, 5]))
        }
        return features
    
    def _calculate_trend_slope(self, prices: np.ndarray) -> float:
        """Calcola pendenza del trend."""
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return slope
    
    def _check_macd_cross(self, macd_main: np.ndarray, macd_signal: np.ndarray) -> int:
        """Verifica se MACD incrocia il segnale."""
        if len(macd_main) < 2:
            return 0
        
        # Bullish cross: MACD sopra SIGNAL e precedentemente sotto
        if macd_main[-1] > macd_signal[-1] and macd_main[-2] <= macd_signal[-2]:
            return 1  # Bullish cross
        # Bearish cross: MACD sotto SIGNAL e precedentemente sopra
        elif macd_main[-1] < macd_signal[-1] and macd_main[-2] >= macd_signal[-2]:
            return -1  # Bearish cross
        return 0  # No cross
    
    def _record_error_pattern(self, sequence: np.ndarray, signal: str, tp_sl_hit: str):
        """Registra pattern quando la predizione è sbagliata."""
        features = self._extract_sequence_features(sequence)
        
        pattern_key = f"{signal}_TO_{tp_sl_hit}"
        
        # Aggiungi caratteristiche ricorrenti
        self.error_patterns[pattern_key].append({
            'features': features,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        # Mantieni solo ultimi N pattern per ogni tipo
        max_patterns = 100
        if len(self.error_patterns[pattern_key]) > max_patterns:
            self.error_patterns[pattern_key] = self.error_patterns[pattern_key][-max_patterns:]
    
    def _record_correct_pattern(self, sequence: np.ndarray, signal: str):
        """Registra pattern quando la predizione è corretta."""
        features = self._extract_sequence_features(sequence)
        
        pattern_key = f"{signal}_CORRECT"
        
        self.correct_patterns[pattern_key].append({
            'features': features,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    
    def identify_common_error_patterns(self) -> List[Dict[str, Any]]:
        """Identifica pattern ricorrenti negli errori."""
        common_patterns = []
        
        for error_type, patterns in self.error_patterns.items():
            if len(patterns) < 5:  # Troppi pochi esempi
                continue
            
            # Analizza features comuni
            all_features = [p['features'] for p in patterns]
            df_features = pd.DataFrame(all_features)
            
            # Trova caratteristiche con valori anomali
            pattern_analysis = {
                'error_type': error_type,
                'count': len(patterns),
                'common_features': {}
            }
            
            for col in df_features.columns:
                median_val = df_features[col].median()
                std_val = df_features[col].std()
                
                # Se valori sono concentrati (bassa deviazione)
                if std_val < median_val * 0.1:  # Poco variabilità
                    pattern_analysis['common_features'][col] = {
                        'median': median_val,
                        'std': std_val,
                        'concentrated': True
                    }
            
            if pattern_analysis['common_features']:
                common_patterns.append(pattern_analysis)
        
        return common_patterns
    
    def generate_learning_examples(self, X_data: np.ndarray, 
                                   y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera esempi di training focalizzati sugli errori.
        """
        # Trova pattern di errore comuni
        error_patterns = self.identify_common_error_patterns()
        
        # Filtra sequenze simili a pattern di errore
        error_indices = []
        
        for i in range(len(X_data)):
            sequence = X_data[i:i+1]
            features = self._extract_sequence_features(sequence)
            
            # Verifica se questa sequenza assomiglia a un pattern di errore
            for pattern in error_patterns:
                if self._matches_error_pattern(features, pattern):
                    error_indices.append(i)
                    break
        
        # Crea dataset bilanciato
        n_error_samples = len(error_indices)
        n_correct_samples = min(n_error_samples * 3, len(X_data) - n_error_samples)
        
        # Seleziona random correct samples
        all_indices = list(range(len(X_data)))
        correct_indices = [i for i in all_indices if i not in error_indices]
        selected_correct = np.random.choice(correct_indices, n_correct_samples, replace=False)
        
        # Combina indices
        selected_indices = np.concatenate([error_indices, selected_correct])
        
        # Crea dataset focalizzato
        X_focused = X_data[selected_indices]
        y_focused = y_data[selected_indices]
        
        logger.info(f"✅ Dataset focalizzato creato: {len(X_focused)} esempi")
        logger.info(f"   Errori: {len(error_indices)}, Corretti: {len(selected_correct)}")
        
        return X_focused, y_focused
    
    def _matches_error_pattern(self, features: Dict, pattern: Dict) -> bool:
        """Verifica se le features corrispondono a un pattern di errore."""
        error_type = pattern['error_type']
        
        for feat_name, feat_info in pattern['common_features'].items():
            if feat_name not in features:
                continue
            
            feat_value = features[feat_name]
            median_val = feat_info['median']
            
            # Verifica se il valore è vicino alla mediana del pattern
            tolerance = median_val * 0.15  # 15% tolleranza
            
            if abs(feat_value - median_val) > tolerance:
                return False
        
        return True