"""
pattern_analyzer.py
Analisi modello - Versione semplificata
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from typing import Dict, List, Any
import joblib
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Analizza pattern e comportamenti modello."""
    
    def __init__(self, model: tf.keras.Model, scaler: Any = None):
        self.model = model
        self.scaler = scaler
        self.feature_names = None
        
        if len(model.input_shape) == 3:
            self.sequence_length = model.input_shape[1]
            self.n_features = model.input_shape[2]
    
    def analyze_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     feature_names: List[str] = None,
                     output_dir: str = 'analysis') -> Dict[str, Any]:
        """
        Analisi completa modello.
        """
        logger.info("Analisi modello in corso...")
        
        # Crea directory output
        os.makedirs(output_dir, exist_ok=True)
        
        # Analisi base
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'model_summary': self._get_model_summary(),
            'data_stats': self._get_data_stats(X_train, y_train, X_test, y_test),
            'performance': self._calculate_performance(X_train, y_train, X_test, y_test),
            'insights': []
        }
        
        # Feature importance semplificata
        if feature_names:
            self.feature_names = feature_names
            feat_analysis = self._analyze_feature_sensitivity(X_test[:100])
            analysis['feature_analysis'] = feat_analysis
        
        # Predizioni
        y_pred_train = self.model.predict(X_train, verbose=0).flatten()
        y_pred_test = self.model.predict(X_test, verbose=0).flatten()
        
        # Error analysis
        analysis['error_analysis'] = self._analyze_errors(y_test, y_pred_test)
        
        # Genera insights
        analysis['insights'] = self._generate_insights(analysis)
        
        # Salva report
        report_file = os.path.join(output_dir, f"model_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        import json
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Genera grafici
        self._generate_plots(X_test, y_test, y_pred_test, output_dir)
        
        logger.info(f"✅ Analisi completata: {report_file}")
        
        return {
            'success': True,
            'report_path': report_file,
            'analysis': analysis
        }
    
    def _get_model_summary(self) -> Dict:
        """Ottiene summary modello."""
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        
        return {
            'input_shape': str(self.model.input_shape),
            'output_shape': str(self.model.output_shape),
            'total_params': self.model.count_params(),
            'layers': len(self.model.layers)
        }
    
    def _get_data_stats(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Statistiche dati."""
        return {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_target_mean': float(np.mean(y_train)),
            'train_target_std': float(np.std(y_train)),
            'test_target_mean': float(np.mean(y_test)),
            'test_target_std': float(np.std(y_test))
        }
    
    def _calculate_performance(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Calcola performance."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred_train = self.model.predict(X_train, verbose=0).flatten()
        y_pred_test = self.model.predict(X_test, verbose=0).flatten()
        
        return {
            'train_mse': float(mean_squared_error(y_train, y_pred_train)),
            'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
            'train_r2': float(r2_score(y_train, y_pred_train)),
            'test_mse': float(mean_squared_error(y_test, y_pred_test)),
            'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
            'test_r2': float(r2_score(y_test, y_pred_test))
        }
    
    def _analyze_feature_sensitivity(self, X_sample: np.ndarray) -> Dict:
        """Analisi sensibilità features."""
        if self.feature_names is None:
            return {}
        
        base_pred = self.model.predict(X_sample, verbose=0).flatten()
        
        sensitivity = {}
        for feat_idx in range(min(5, self.n_features)):  # Analizza prime 5
            feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"feature_{feat_idx}"
            
            # Perturba feature
            X_perturbed = X_sample.copy()
            perturbation = np.random.normal(0, 0.1, size=X_sample.shape)
            X_perturbed[:, :, feat_idx] += perturbation[:, :, feat_idx]
            
            perturbed_pred = self.model.predict(X_perturbed, verbose=0).flatten()
            
            diff = np.mean(np.abs(perturbed_pred - base_pred))
            sensitivity[feat_name] = float(diff)
        
        return sensitivity
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Analizza errori."""
        errors = y_true - y_pred
        
        return {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(np.abs(errors))),
            'error_skewness': float(pd.Series(errors).skew()),
            'error_distribution': {
                'negative': int(np.sum(errors < -0.1)),
                'small': int(np.sum((errors >= -0.1) & (errors <= 0.1))),
                'positive': int(np.sum(errors > 0.1))
            }
        }
    
    def _generate_insights(self, analysis: Dict) -> List[str]:
        """Genera insights."""
        insights = []
        perf = analysis['performance']
        
        # Analizza R²
        test_r2 = perf['test_r2']
        if test_r2 > 0.7:
            insights.append(f"Ottimo potere predittivo (R²={test_r2:.3f})")
        elif test_r2 > 0.4:
            insights.append(f"Potere predittivo moderato (R²={test_r2:.3f})")
        else:
            insights.append(f"Potere predittivo limitato (R²={test_r2:.3f})")
        
        # Analizza overfitting
        train_mse = perf['train_mse']
        test_mse = perf['test_mse']
        if test_mse > train_mse * 1.5:
            insights.append("Possibile overfitting (MSE test > 1.5x train)")
        elif test_mse < train_mse:
            insights.append("Buona generalizzazione (MSE test < train)")
        
        # Error analysis
        error_stats = analysis.get('error_analysis', {})
        if error_stats.get('mean_error', 0) > 0.05:
            insights.append("Tendenza a sovrastimare (mean error positivo)")
        elif error_stats.get('mean_error', 0) < -0.05:
            insights.append("Tendenza a sottostimare (mean error negativo)")
        
        return insights
    
    def _generate_plots(self, X_test: np.ndarray, y_test: np.ndarray, 
                       y_pred_test: np.ndarray, output_dir: str):
        """Genera grafici."""
        try:
            # 1. Predizioni vs Real
            plt.figure(figsize=(10, 6))
            plt.plot(y_test[:100], label='Reale', alpha=0.7)
            plt.plot(y_pred_test[:100], label='Predetto', alpha=0.7)
            plt.title('Predizioni vs Valori Reali (primi 100 campioni)')
            plt.xlabel('Campione')
            plt.ylabel('Valore')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'predictions_vs_real.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. Error distribution
            plt.figure(figsize=(10, 6))
            errors = y_test - y_pred_test
            plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(0, color='red', linestyle='--', label='Zero error')
            plt.title('Distribuzione Errori di Predizione')
            plt.xlabel('Errore')
            plt.ylabel('Frequenza')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # 3. Scatter plot
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred_test, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', label='Linea perfetta')
            plt.title('Valori Reali vs Predetti')
            plt.xlabel('Reale')
            plt.ylabel('Predetto')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Grafici generati in {output_dir}")
            
        except Exception as e:
            logger.warning(f"Impossibile generare grafici: {e}")

def analyze_trading_model(model_path: str, 
                         scaler_path: str,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         feature_names: List[str] = None,
                         output_dir: str = 'analysis_results'):
    """
    Funzione wrapper per analisi modello.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Analisi modello con compatibilità garantita...")
    
    # Carica modello con sistema garantito
    from model_compatibility import load_model_guaranteed
    model = load_model_guaranteed(model_path)
    scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
    
    # Inizializza analyzer
    analyzer = PatternAnalyzer(model, scaler)
    
    # Esegui analisi
    result = analyzer.analyze_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dir
    )
    
    return analyzer

if __name__ == "__main__":
    # Test
    print("Test pattern_analyzer.py")
    
    # Dati dummy
    np.random.seed(42)
    X = np.random.randn(100, 20, 9)
    y = np.random.randn(100, 1)
    
    # Modello dummy
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(16, input_shape=(20, 9)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Test analyzer
    analyzer = PatternAnalyzer(model)
    result = analyzer.analyze_model(
        X_train=X[:80], y_train=y[:80],
        X_test=X[80:], y_test=y[80:],
        output_dir='test_analysis'
    )
    
    print(f"✅ Analisi completata: {result['success']}")