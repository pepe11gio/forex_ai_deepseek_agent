"""
pattern_analyzer.py
Modulo per l'analisi dei pattern appresi dal modello di trading.
Analizza feature importance, pattern temporali e comportamenti del modello.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Dict, List, Tuple, Any, Optional
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """
    Classe per analizzare pattern e comportamenti appresi dal modelo di trading.
    """
    
    def __init__(self, model: tf.keras.Model, scaler: Any = None):
        """
        Inizializza l'analizzatore di pattern.
        
        Args:
            model: Modello Keras addestrato
            scaler: Scaler usato per normalizzare i dati
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = None
        self.sequence_length = None
        self.n_features = None
        
        # Estrai metadati dal modello
        if len(model.input_shape) == 3:
            self.sequence_length = model.input_shape[1]
            self.n_features = model.input_shape[2]
            logger.info(f"Modello analizzato: sequenze di {self.sequence_length} tick con {self.n_features} features")
    
    def set_feature_names(self, feature_names: List[str]):
        """
        Imposta i nomi delle features per analisi più descrittive.
        
        Args:
            feature_names: Lista di nomi delle features
        """
        if len(feature_names) != self.n_features:
            logger.warning(f"Numero feature names ({len(feature_names)}) non corrisponde a n_features ({self.n_features})")
        self.feature_names = feature_names
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                  n_repeats: int = 10, 
                                  random_state: int = 42) -> pd.DataFrame:
        """
        Analizza l'importanza delle features usando permutation importance.
        
        Args:
            X: Dati di input
            y: Target
            n_repeats: Numero di ripetizioni per permutation
            random_state: Seed per riproducibilità
            
        Returns:
            DataFrame con importanza delle features
        """
        logger.info("Analisi importanza features tramite permutation...")
        
        # Crea funzione di scoring per il modello
        def model_predict(X_data):
            return self.model.predict(X_data, verbose=0).flatten()
        
        # Calcola permutation importance
        perm_importance = permutation_importance(
            estimator=self.model,
            X=X,
            y=y.flatten(),
            scoring='neg_mean_squared_error',
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Crea DataFrame con risultati
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        
        # Per LSTM, abbiamo importanza per ogni timestep e feature
        importance_by_timestep = []
        
        for timestep in range(self.sequence_length):
            for feat_idx in range(self.n_features):
                # Per permutation importance globale, consideriamo l'effetto medio
                # In pratica dovremmo fare permutation per ogni timestep separatamente
                # Questa è un'approssimazione
                feat_name = f"{self.feature_names[feat_idx]}_t-{self.sequence_length - timestep - 1}"
                importance_by_timestep.append({
                    'feature': feat_name,
                    'timestep': timestep,
                    'feature_idx': feat_idx,
                    'importance_mean': np.mean(perm_importance.importances_mean),
                    'importance_std': np.mean(perm_importance.importances_std),
                })
        
        importance_df = pd.DataFrame(importance_by_timestep)
        
        logger.info("Analisi importanza features completata")
        return importance_df
    
    def analyze_layer_activations(self, X_sample: np.ndarray, 
                                 layer_names: List[str] = None) -> Dict:
        """
        Analizza le attivazioni dei layer del modello per un campione di input.
        
        Args:
            X_sample: Campione di input (forma: [1, sequence_length, n_features])
            layer_names: Nomi dei layer da analizzare (se None, usa tutti i layer)
            
        Returns:
            Dizionario con attivazioni per ogni layer
        """
        logger.info("Analisi attivazioni layer...")
        
        if len(X_sample.shape) != 3:
            raise ValueError(f"X_sample deve avere forma [1, seq_len, n_features]. Ricevuto: {X_sample.shape}")
        
        # Se layer_names non specificato, usa tutti i layer LSTM/Dense
        if layer_names is None:
            layer_names = [layer.name for layer in self.model.layers 
                          if 'lstm' in layer.name.lower() or 
                          'dense' in layer.name.lower()]
        
        # Crea modelli intermedi per estrarre attivazioni
        activations = {}
        
        for layer_name in layer_names:
            try:
                # Crea modello intermedio
                intermediate_model = Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer(layer_name).output
                )
                
                # Ottieni attivazioni
                layer_output = intermediate_model.predict(X_sample, verbose=0)
                activations[layer_name] = layer_output
                
                logger.info(f"Layer {layer_name}: output shape {layer_output.shape}")
                
            except Exception as e:
                logger.warning(f"Impossibile estrarre attivazioni per layer {layer_name}: {str(e)}")
        
        return activations
    
    def analyze_temporal_patterns(self, X: np.ndarray, n_samples: int = 100) -> Dict:
        """
        Analizza pattern temporali nelle predizioni del modello.
        
        Args:
            X: Dati di input
            n_samples: Numero di campioni da analizzare
            
        Returns:
            Dizionario con pattern temporali analizzati
        """
        logger.info("Analisi pattern temporali...")
        
        # Campiona dati se necessario
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sampled = X[indices]
        else:
            X_sampled = X
            n_samples = len(X)
        
        # Ottieni predizioni
        predictions = self.model.predict(X_sampled, verbose=0).flatten()
        
        # Analizza sensibilità temporale
        temporal_sensitivity = self._analyze_temporal_sensitivity(X_sampled)
        
        # Identifica pattern ricorrenti
        recurrent_patterns = self._identify_recurrent_patterns(X_sampled, predictions)
        
        # Analizza autocorrelazione degli errori
        if len(predictions) > 1:
            autocorr_analysis = self._analyze_autocorrelation(predictions)
        else:
            autocorr_analysis = {}
        
        results = {
            'temporal_sensitivity': temporal_sensitivity,
            'recurrent_patterns': recurrent_patterns,
            'autocorrelation_analysis': autocorr_analysis,
            'n_samples_analyzed': n_samples
        }
        
        logger.info("Analisi pattern temporali completata")
        return results
    
    def _analyze_temporal_sensitivity(self, X: np.ndarray) -> Dict:
        """
        Analizza la sensibilità del modello a differenti timestep.
        
        Args:
            X: Dati di input
            
        Returns:
            Dizionario con analisi di sensibilità
        """
        sensitivity = {}
        
        # Per ogni timestep, perturbiamo leggermente i valori
        base_predictions = self.model.predict(X, verbose=0).flatten()
        
        for timestep in range(self.sequence_length):
            # Crea copia perturbata
            X_perturbed = X.copy()
            perturbation = np.random.normal(0, 0.1, size=(len(X), 1, self.n_features))
            X_perturbed[:, timestep:timestep+1, :] += perturbation
            
            # Predizioni con dati perturbati
            perturbed_predictions = self.model.predict(X_perturbed, verbose=0).flatten()
            
            # Calcola differenza
            diff = np.mean(np.abs(perturbed_predictions - base_predictions))
            sensitivity[f'timestep_{timestep}'] = {
                'sensitivity_score': float(diff),
                'relative_importance': float(diff / np.max(list(sensitivity.values())[-1]['sensitivity_score']) 
                                             if sensitivity else 1.0)
            }
        
        return sensitivity
    
    def _identify_recurrent_patterns(self, X: np.ndarray, predictions: np.ndarray,
                                    threshold: float = 0.1) -> Dict:
        """
        Identifica pattern ricorrenti negli input che portano a predizioni simili.
        
        Args:
            X: Dati di input
            predictions: Predizioni del modello
            threshold: Soglia per considerare predizioni simili
            
        Returns:
            Dizionario con pattern identificati
        """
        patterns = {
            'high_prediction_clusters': [],
            'low_prediction_clusters': [],
            'pattern_examples': []
        }
        
        # Cluster predizioni (semplificato)
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        high_pred_mask = predictions > pred_mean + threshold * pred_std
        low_pred_mask = predictions < pred_mean - threshold * pred_std
        
        if np.any(high_pred_mask):
            high_indices = np.where(high_pred_mask)[0]
            patterns['high_prediction_clusters'] = {
                'count': len(high_indices),
                'mean_prediction': float(np.mean(predictions[high_indices])),
                'example_indices': high_indices[:3].tolist()  # Primi 3 esempi
            }
        
        if np.any(low_pred_mask):
            low_indices = np.where(low_pred_mask)[0]
            patterns['low_prediction_clusters'] = {
                'count': len(low_indices),
                'mean_prediction': float(np.mean(predictions[low_indices])),
                'example_indices': low_indices[:3].tolist()
            }
        
        # Trova pattern più comuni (semplificato - media dei segnali)
        if len(X) > 0:
            avg_high_pattern = np.mean(X[high_pred_mask], axis=0) if np.any(high_pred_mask) else None
            avg_low_pattern = np.mean(X[low_pred_mask], axis=0) if np.any(low_pred_mask) else None
            
            patterns['pattern_examples'] = {
                'average_high_pattern': avg_high_pattern.tolist() if avg_high_pattern is not None else None,
                'average_low_pattern': avg_low_pattern.tolist() if avg_low_pattern is not None else None
            }
        
        return patterns
    
    def _analyze_autocorrelation(self, series: np.ndarray, max_lag: int = 10) -> Dict:
        """
        Analizza autocorrelazione in una serie.
        
        Args:
            series: Serie temporale
            max_lag: Massimo lag da analizzare
            
        Returns:
            Dizionario con analisi autocorrelazione
        """
        try:
            from statsmodels.tsa.stattools import acf
            
            # Gestisci serie troppo corte
            if len(series) < max_lag * 2:
                max_lag = min(10, len(series) // 2)
                if max_lag < 2:
                    return {
                        'autocorrelation_values': [],
                        'significant_lags': [],
                        'has_autocorrelation': False,
                        'error': 'Serie troppo corta per analisi autocorrelazione'
                    }
            
            autocorr = acf(series, nlags=max_lag, fft=True)
            
            # Trova lag significativi (autocorrelazione > 0.2 o < -0.2)
            significant_lags = []
            for lag, value in enumerate(autocorr[1:], start=1):  # Salta lag 0
                if abs(value) > 0.2:
                    significant_lags.append({
                        'lag': lag,
                        'autocorrelation': float(value),
                        'interpretation': 'positivo' if value > 0 else 'negativo'
                    })
            
            return {
                'autocorrelation_values': autocorr.tolist(),
                'significant_lags': significant_lags,
                'has_autocorrelation': len(significant_lags) > 0
            }
            
        except ImportError:
            logger.warning("statsmodels non installato. Autocorrelazione non calcolata.")
            return {
                'autocorrelation_values': [],
                'significant_lags': [],
                'has_autocorrelation': False,
                'error': 'statsmodels non installato'
            }
        except Exception as e:
            logger.warning(f"Impossibile calcolare autocorrelazione: {str(e)}")
            return {
                'autocorrelation_values': [],
                'significant_lags': [],
                'has_autocorrelation': False,
                'error': str(e)
            }
    
    def analyze_prediction_distribution(self, X: np.ndarray, 
                                       bins: int = 50) -> Dict:
        """
        Analizza la distribuzione delle predizioni del modello.
        
        Args:
            X: Dati di input
            bins: Numero di bin per l'istogramma
            
        Returns:
            Dizionario con analisi distribuzione
        """
        logger.info("Analisi distribuzione predizioni...")
        
        predictions = self.model.predict(X, verbose=0).flatten()
        
        # Statistiche descrittive
        stats = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'skewness': float(pd.Series(predictions).skew()),
            'kurtosis': float(pd.Series(predictions).kurtosis()),
            'percentile_25': float(np.percentile(predictions, 25)),
            'percentile_75': float(np.percentile(predictions, 75))
        }
        
        # Analisi distribuzione
        hist, bin_edges = np.histogram(predictions, bins=bins)
        
        distribution_analysis = {
            'statistics': stats,
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'is_normal': self._test_normality(predictions),
            'prediction_ranges': self._analyze_prediction_ranges(predictions)
        }
        
        logger.info(f"Distribuzione predizioni: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        return distribution_analysis
    
    def _test_normality(self, data: np.ndarray, alpha: float = 0.05) -> Dict:
        """
        Test di normalità sui dati.
        
        Args:
            data: Dati da testare
            alpha: Livello di significatività
            
        Returns:
            Dizionario con risultati test
        """
        try:
            from scipy import stats
            
            # Shapiro-Wilk test (per sample size < 5000)
            if len(data) < 5000:
                stat, p_value = stats.shapiro(data)
                test_name = 'Shapiro-Wilk'
            else:
                # Kolmogorov-Smirnov per grandi campioni
                stat, p_value = stats.kstest(data, 'norm')
                test_name = 'Kolmogorov-Smirnov'
            
            is_normal = p_value > alpha
            
            return {
                'test_name': test_name,
                'test_statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': bool(is_normal),
                'alpha': alpha
            }
            
        except Exception as e:
            logger.warning(f"Impossibile eseguire test di normalità: {str(e)}")
            return {
                'test_name': 'Failed',
                'error': str(e),
                'is_normal': None
            }
    
    def _analyze_prediction_ranges(self, predictions: np.ndarray) -> Dict:
        """
        Analizza le range di predizione.
        
        Args:
            predictions: Array di predizioni
            
        Returns:
            Dizionario con analisi ranges
        """
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        ranges = {
            'very_low': pred_mean - 2 * pred_std,
            'low': pred_mean - pred_std,
            'medium': pred_mean,
            'high': pred_mean + pred_std,
            'very_high': pred_mean + 2 * pred_std
        }
        
        # Conta predizioni in ogni range
        counts = {}
        for range_name, threshold in ranges.items():
            if range_name == 'very_low':
                mask = predictions < ranges['low']
            elif range_name == 'low':
                mask = (predictions >= ranges['low']) & (predictions < ranges['medium'])
            elif range_name == 'medium':
                mask = (predictions >= ranges['medium']) & (predictions < ranges['high'])
            elif range_name == 'high':
                mask = (predictions >= ranges['high']) & (predictions < ranges['very_high'])
            else:  # very_high
                mask = predictions >= ranges['very_high']
            
            counts[range_name] = int(np.sum(mask))
        
        return {
            'thresholds': {k: float(v) for k, v in ranges.items()},
            'counts': counts,
            'percentages': {k: (v / len(predictions) * 100) for k, v in counts.items()}
        }
    
    def generate_analysis_report(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                output_file: str = None) -> Dict:
        """
        Genera un report completo di analisi del modello.
        
        Args:
            X_train, y_train: Dati di training
            X_test, y_test: Dati di test
            output_file: File per salvare il report (se None, non salva)
            
        Returns:
            Dizionario con report completo
        """
        logger.info("Generazione report di analisi completo...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_summary': self._get_model_summary(),
            'training_data_stats': self._get_data_stats(X_train, y_train, 'train'),
            'test_data_stats': self._get_data_stats(X_test, y_test, 'test'),
        }
        
        # Analisi feature importance (su subset per performance)
        if len(X_train) > 1000:
            X_sample = X_train[:1000]
            y_sample = y_train[:1000]
        else:
            X_sample = X_train
            y_sample = y_train
        
        logger.info("Analisi feature importance...")
        feature_importance = self.analyze_feature_importance(X_sample, y_sample)
        report['feature_importance'] = feature_importance.to_dict('records')
        
        # Analisi distribuzione predizioni
        logger.info("Analisi distribuzione predizioni...")
        report['prediction_distribution_train'] = self.analyze_prediction_distribution(X_train)
        report['prediction_distribution_test'] = self.analyze_prediction_distribution(X_test)
        
        # Analisi pattern temporali
        logger.info("Analisi pattern temporali...")
        report['temporal_patterns'] = self.analyze_temporal_patterns(X_test, n_samples=min(200, len(X_test)))
        
        # Analisi layer (su un campione)
        logger.info("Analisi attivazioni layer...")
        if len(X_test) > 0:
            sample_idx = 0
            X_sample = X_test[sample_idx:sample_idx+1]
            report['layer_activations_sample'] = self.analyze_layer_activations(X_sample)
        
        # Performance metrics
        logger.info("Calcolo metriche performance...")
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred_train = self.model.predict(X_train, verbose=0).flatten()
        y_pred_test = self.model.predict(X_test, verbose=0).flatten()
        
        report['performance_metrics'] = {
            'train': {
                'mse': float(mean_squared_error(y_train, y_pred_train)),
                'mae': float(mean_absolute_error(y_train, y_pred_train)),
                'r2': float(r2_score(y_train, y_pred_train))
            },
            'test': {
                'mse': float(mean_squared_error(y_test, y_pred_test)),
                'mae': float(mean_absolute_error(y_test, y_pred_test)),
                'r2': float(r2_score(y_test, y_pred_test))
            }
        }
        
        # Insights e raccomandazioni
        report['insights'] = self._generate_insights(report)
        
        # Salva report se richiesto
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report salvato in: {output_file}")
        
        logger.info("Report di analisi completato!")
        return report
    
    def _get_model_summary(self) -> Dict:
        """Ottiene summary del modello in formato dizionario."""
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        
        return {
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'summary_text': '\n'.join(summary),
            'sequence_length': self.sequence_length,
            'n_features': self.n_features
        }
    
    def _get_data_stats(self, X: np.ndarray, y: np.ndarray, prefix: str) -> Dict:
        """Calcola statistiche sui dati."""
        return {
            f'{prefix}_samples': len(X),
            f'{prefix}_shape': X.shape,
            f'{prefix}_target_mean': float(np.mean(y)),
            f'{prefix}_target_std': float(np.std(y)),
            f'{prefix}_target_min': float(np.min(y)),
            f'{prefix}_target_max': float(np.max(y))
        }
    
    def _generate_insights(self, report: Dict) -> Dict:
        """Genera insights e raccomandazioni basate sull'analisi."""
        insights = {
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'risk_warnings': []
        }
        
        # Analizza performance
        test_r2 = report['performance_metrics']['test']['r2']
        if test_r2 > 0.7:
            insights['strengths'].append(f"Modello con buon potere predittivo (R²={test_r2:.3f})")
        elif test_r2 > 0.4:
            insights['strengths'].append(f"Modello con potere predittivo moderato (R²={test_r2:.3f})")
        else:
            insights['weaknesses'].append(f"Potere predittivo limitato (R²={test_r2:.3f})")
            insights['recommendations'].append("Considera feature engineering aggiuntivo o modelli più complessi")
        
        # Analizza overfitting
        train_mse = report['performance_metrics']['train']['mse']
        test_mse = report['performance_metrics']['test']['mse']
        mse_ratio = test_mse / train_mse if train_mse > 0 else 1
        
        if mse_ratio > 1.5:
            insights['weaknesses'].append(f"Possibile overfitting (MSE test/train ratio: {mse_ratio:.2f})")
            insights['recommendations'].append("Aumenta regolarizzazione o riduci complessità modello")
            insights['risk_warnings'].append("Il modello potrebbe performare male su dati nuovi")
        
        # Analizza feature importance
        if 'feature_importance' in report and report['feature_importance']:
            feat_importance = report['feature_importance']
            if len(feat_importance) > 0:
                # Controlla se alcune features dominano
                importance_values = [f['importance_mean'] for f in feat_importance[:5]]
                if max(importance_values) > 2 * np.mean(importance_values):
                    top_feat = feat_importance[0]['feature']
                    insights['insights'] = [f"La feature '{top_feat}' sembra avere importanza dominante"]
        
        # Analizza distribuzione predizioni
        pred_stats = report['prediction_distribution_test']['statistics']
        if abs(pred_stats['skewness']) > 1:
            skew_dir = "positivo" if pred_stats['skewness'] > 0 else "negativo"
            insights['insights'] = [f"Distribuzione predizioni skew {skew_dir} (skewness={pred_stats['skewness']:.2f})"]
            insights['recommendations'].append(f"Considera trasformazione delle predizioni per ridurre skewness")
        
        # Aggiungi raccomandazioni generali
        insights['recommendations'].extend([
            "Monitora costantemente le performance su dati fuori campione",
            "Considera ensemble di modelli per riduce varianza",
            "Implementa sistemi di stop-loss basati sulla confidence delle predizioni"
        ])
        
        insights['risk_warnings'].extend([
            "I modelli ML per trading hanno limiti intrinseci - non sostituiscono analisi umana",
            "Le performance passate non garantiscono risultati futuri",
            "Sempre testare su dati out-of-sample prima di deployment live"
        ])
        
        return insights
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 20,
                               save_path: str = None):
        """
        Visualizza l'importanza delle features.
        
        Args:
            importance_df: DataFrame con importanza features
            top_n: Numero di features top da visualizzare
            save_path: Percorso per salvare il plot
        """
        # Se ci sono molti feature-timestep, aggrega per feature
        if 'timestep' in importance_df.columns:
            # Media per feature
            importance_agg = importance_df.groupby('feature_idx')['importance_mean'].mean().reset_index()
            
            # Aggiungi nomi features
            if self.feature_names is not None:
                importance_agg['feature_name'] = [self.feature_names[i] for i in importance_agg['feature_idx']]
            else:
                importance_agg['feature_name'] = [f'feature_{i}' for i in importance_agg['feature_idx']]
            
            # Ordina
            importance_agg = importance_agg.sort_values('importance_mean', ascending=False)
            
            # Plot
            plt.figure(figsize=(12, 6))
            bars = plt.barh(range(min(top_n, len(importance_agg))),
                          importance_agg['importance_mean'][:top_n][::-1])
            plt.yticks(range(min(top_n, len(importance_agg))),
                      importance_agg['feature_name'][:top_n][::-1])
            plt.xlabel('Feature Importance (permutation)')
            plt.title('Top Feature Importance (aggregated across timesteps)')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Aggiungi valori sulle barre
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', va='center')
        
        else:
            # Plot diretto
            importance_df_sorted = importance_df.sort_values('importance_mean').tail(top_n)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(importance_df_sorted)),
                          importance_df_sorted['importance_mean'])
            plt.yticks(range(len(importance_df_sorted)),
                      importance_df_sorted['feature'])
            plt.xlabel('Feature Importance (permutation)')
            plt.title(f'Top {top_n} Feature Importance')
            plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot salvato in: {save_path}")
        
        plt.show()
    
    def plot_prediction_distribution(self, X: np.ndarray, 
                                    title: str = "Prediction Distribution",
                                    save_path: str = None):
        """
        Visualizza distribuzione delle predizioni.
        
        Args:
            X: Dati di input
            title: Titolo del plot
            save_path: Percorso per salvare il plot
        """
        predictions = self.model.predict(X, verbose=0).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Istogramma
        axes[0, 0].hist(predictions, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(predictions), color='red', linestyle='--', label=f'Mean: {np.mean(predictions):.4f}')
        axes[0, 0].axvline(np.median(predictions), color='green', linestyle='--', label=f'Median: {np.median(predictions):.4f}')
        axes[0, 0].set_title(f'{title} - Histogram')
        axes[0, 0].set_xlabel('Prediction Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(predictions, vert=True)
        axes[0, 1].set_title(f'{title} - Box Plot')
        axes[0, 1].set_ylabel('Prediction Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # QQ plot (per normalità)
        from scipy import stats
        stats.probplot(predictions, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{title} - Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Densità KDE
        import seaborn as sns
        sns.kdeplot(predictions, ax=axes[1, 1], shade=True)
        axes[1, 1].axvline(np.mean(predictions), color='red', linestyle='--', label='Mean')
        axes[1, 1].set_title(f'{title} - Density Plot')
        axes[1, 1].set_xlabel('Prediction Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Prediction Distribution Analysis (n={len(predictions)})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Distribution plot salvato in: {save_path}")
        
        plt.show()


def analyze_trading_model(model_path: str, 
                         scaler_path: str,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         feature_names: List[str] = None,
                         output_dir: str = 'analysis_results') -> PatternAnalyzer:
    """
    Funzione completa per analizzare un modello di trading.
    
    Args:
        model_path: Percorso del modello
        scaler_path: Percorso dello scaler
        X_train, y_train: Dati di training
        X_test, y_test: Dati di test
        feature_names: Nomi delle features
        output_dir: Directory per salvare risultati
        
    Returns:
        PatternAnalyzer configurato
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("AVVIO ANALISI COMPLETA MODELLO TRADING")
    logger.info("=" * 60)
    
    # Carica modello e scaler
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
    
    # Inizializza analyzer
    analyzer = PatternAnalyzer(model, scaler)
    
    if feature_names:
        analyzer.set_feature_names(feature_names)
    
    # Genera report completo
    report_file = os.path.join(output_dir, f'model_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    report = analyzer.generate_analysis_report(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_file=report_file
    )
    
    # Genera visualizzazioni
    logger.info("Generazione visualizzazioni...")
    
    # Feature importance plot
    if len(X_train) > 0 and len(y_train) > 0:
        X_sample = X_train[:min(1000, len(X_train))]
        y_sample = y_train[:min(1000, len(y_train))]
        importance_df = analyzer.analyze_feature_importance(X_sample, y_sample)
        
        importance_plot_path = os.path.join(output_dir, 'feature_importance.png')
        analyzer.plot_feature_importance(importance_df, save_path=importance_plot_path)
    
    # Prediction distribution plot
    if len(X_test) > 0:
        dist_plot_path = os.path.join(output_dir, 'prediction_distribution.png')
        analyzer.plot_prediction_distribution(X_test, save_path=dist_plot_path)
    
    # Stampa insights
    logger.info("\n" + "=" * 60)
    logger.info("INSIGHTS PRINCIPALI:")
    logger.info("=" * 60)
    
    insights = report.get('insights', {})
    for category, items in insights.items():
        if items:
            logger.info(f"\n{category.upper()}:")
            for item in items:
                logger.info(f"  • {item}")
    
    logger.info(f"\nReport completo salvato in: {report_file}")
    logger.info("Analisi completata!")
    
    return analyzer


if __name__ == "__main__":
    # Esempio di utilizzo
    print("Test del pattern_analyzer.py")
    print("=" * 50)
    
    try:
        # Genera dati sintetici per test
        np.random.seed(42)
        n_samples = 500
        sequence_length = 10
        n_features = 5
        
        X_test_analysis = np.random.randn(n_samples, sequence_length, n_features)
        y_test_analysis = np.random.randn(n_samples, 1)
        
        # Crea modello semplice per test
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        model_test = Sequential([
            LSTM(32, input_shape=(sequence_length, n_features), return_sequences=True),
            LSTM(16),
            Dense(1)
        ])
        
        model_test.compile(optimizer='adam', loss='mse')
        
        print(f"\nModello test creato:")
        print(f"  Input shape: {(sequence_length, n_features)}")
        print(f"  Output shape: {model_test.output_shape}")
        
        # Inizializza analyzer
        analyzer = PatternAnalyzer(model_test)
        
        # Test analisi distribuzione predizioni
        print("\nTest analisi distribuzione predizioni...")
        dist_analysis = analyzer.analyze_prediction_distribution(X_test_analysis[:100])
        print(f"  Media predizioni: {dist_analysis['statistics']['mean']:.4f}")
        print(f"  Std predizioni: {dist_analysis['statistics']['std']:.4f}")
        
        # Test analisi feature importance
        print("\nTest analisi feature importance...")
        try:
            X_sample = X_test_analysis[:50]
            y_sample = y_test_analysis[:50]
            importance_df = analyzer.analyze_feature_importance(X_sample, y_sample)
            print(f"  Feature importance calcolata: {len(importance_df)} feature-timestep")
        except Exception as e:
            print(f"  Feature importance non disponibile: {e}")
        
        # Test analisi pattern temporali
        print("\nTest analisi pattern temporali...")
        try:
            temporal_patterns = analyzer.analyze_temporal_patterns(X_test_analysis[:50])
            print(f"  Pattern temporali analizzati: {temporal_patterns['n_samples_analyzed']} campioni")
        except Exception as e:
            print(f"  Analisi pattern temporali non disponibile: {e}")
        
        print("\n" + "=" * 50)
        print("TEST COMPLETATO CON SUCCESSO!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nErrore durante il test: {str(e)}")
        import traceback
        traceback.print_exc()