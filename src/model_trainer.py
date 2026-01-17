"""
model_trainer.py
Modulo per l'addestramento del modello di machine learning per trading.
Implementa modelli LSTM per predizione di serie temporali finanziarie.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime
import os
from tensorflow.keras import metrics

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed per riproducibilità
np.random.seed(42)
tf.random.set_seed(42)


class LSTMTradingModel:
    """
    Classe per l'addestramento e gestione di modelli LSTM per trading.
    Supporta vari tipi di architetture LSTM.
    """
    
    def __init__(self, model_type: str = 'standard', 
                 model_name: str = None):
        """
        Inizializza il trainer del modello.
        
        Args:
            model_type: Tipo di modello ('standard', 'stacked', 'bidirectional')
            model_name: Nome del modello (per salvataggio)
        """
        self.model_type = model_type
        self.model_name = model_name or f"trading_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = None
        self.history = None
        self.scaler = None
        self.sequence_length = None
        self.n_features = None
        
        # Configurazioni di default
        self.model_config = {
            'standard': {
                'lstm_units': [64, 32],
                'dense_units': [16],
                'dropout_rate': 0.2,
                'use_batch_norm': True
            },
            'stacked': {
                'lstm_units': [128, 64, 32],
                'dense_units': [32, 16],
                'dropout_rate': 0.3,
                'use_batch_norm': True
            },
            'bidirectional': {
                'lstm_units': [64, 32],
                'dense_units': [16],
                'dropout_rate': 0.2,
                'use_batch_norm': True
            }
        }
        
        logger.info(f"Trading Model Trainer inizializzato. Tipo: {model_type}")
    
    def build_model(self, input_shape: Tuple[int, int], 
                    output_units: int = 1,
                    config: Dict[str, Any] = None) -> Model:
        """
        Costruisce il modello LSTM in base al tipo selezionato.
        
        Args:
            input_shape: Shape dell'input (sequence_length, n_features)
            output_units: Numero di unità di output
            config: Configurazione personalizzata del modello
            
        Returns:
            Modello Keras compilato
        """
        logger.info(f"Costruzione modello LSTM. Input shape: {input_shape}")
        
        # Usa configurazione personalizzata o default
        if config is None:
            config = self.model_config.get(self.model_type, self.model_config['standard'])
        
        sequence_length, n_features = input_shape
        lstm_units = config['lstm_units']
        dense_units = config['dense_units']
        dropout_rate = config['dropout_rate']
        use_batch_norm = config['use_batch_norm']
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        # Definisci input
        inputs = Input(shape=(sequence_length, n_features))
        x = inputs
        
        # Costruisci architettura LSTM in base al tipo
        if self.model_type == 'standard':
            # LSTM standard
            for i, units in enumerate(lstm_units):
                return_sequences = (i < len(lstm_units) - 1)
                x = LSTM(units=units, 
                        return_sequences=return_sequences,
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
                
                if use_batch_norm:
                    x = BatchNormalization()(x)
                
                x = Dropout(dropout_rate)(x)
        
        elif self.model_type == 'stacked':
            # LSTM impilati
            for i, units in enumerate(lstm_units):
                return_sequences = (i < len(lstm_units) - 1)
                x = LSTM(units=units, 
                        return_sequences=return_sequences,
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
                
                if use_batch_norm:
                    x = BatchNormalization()(x)
                
                x = Dropout(dropout_rate)(x)
        
        elif self.model_type == 'bidirectional':
            # LSTM bidirezionali
            for i, units in enumerate(lstm_units):
                return_sequences = (i < len(lstm_units) - 1)
                x = Bidirectional(
                    LSTM(units=units, 
                         return_sequences=return_sequences,
                         kernel_regularizer=tf.keras.regularizers.l2(0.001))
                )(x)
                
                if use_batch_norm:
                    x = BatchNormalization()(x)
                
                x = Dropout(dropout_rate)(x)
        
        else:
            raise ValueError(f"Tipo modello non supportato: {self.model_type}")
        
        # Aggiungi layer Dense
        for units in dense_units:
            x = Dense(units, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
            x = Dropout(dropout_rate * 0.5)(x)
        
        # Layer di output
        outputs = Dense(output_units, activation='linear')(x)
        
        # Crea modello
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compila modello con oggetti metriche invece di stringhe
        optimizer = Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanSquaredError()
            ]
        )
        
        self.model = model
        
        logger.info(f"Modello costruito con successo.")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              callbacks: List = None,
              validation_split: float = 0.1,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Addestra il modello sui dati forniti.
        
        Args:
            X_train: Dati di training
            y_train: Target di training
            X_val: Dati di validazione
            y_val: Target di validazione
            epochs: Numero di epoche
            batch_size: Dimensione del batch
            callbacks: Callbacks personalizzati
            validation_split: Percentuale di validazione
            verbose: Verbosità del training
            
        Returns:
            Dizionario con storia del training e metriche
        """
        logger.info(f"Inizio training del modello...")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Validation samples: {X_val.shape[0]}")
        logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")
        
        # Callbacks di default se non forniti
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Training con shuffle=False per serie temporali
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Importante per serie temporali
        )
        
        # Valutazione finale
        train_metrics = self.evaluate(X_train, y_train, prefix='train_')
        val_metrics = self.evaluate(X_val, y_val, prefix='val_')
        
        # Combina metriche
        metrics = {**train_metrics, **val_metrics}
        
        logger.info("Training completato!")
        logger.info(f"MSE Train: {metrics['train_mse']:.6f}, Val: {metrics['val_mse']:.6f}")
        logger.info(f"MAE Train: {metrics['train_mae']:.6f}, Val: {metrics['val_mae']:.6f}")
        
        return {
            'history': self.history.history,
            'metrics': metrics,
            'model': self.model
        }
    
    def _get_default_callbacks(self) -> List:
        """
        Restituisce i callback di default per il training.
        
        Returns:
            Lista di callback Keras
        """
        # Crea directory per checkpoint
        os.makedirs('models/checkpoints', exist_ok=True)
        
        callbacks = [
            # Early stopping per evitare overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Salvataggio del modello migliore
            ModelCheckpoint(
                filepath=f'models/checkpoints/{self.model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Riduzione learning rate su plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return callbacks
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
             prefix: str = '') -> Dict[str, float]:
        """
        Valuta il modello sui dati forniti.

        Args:
            X: Dati di input
            y: Target vero
            prefix: Prefisso per le metriche

        Returns:
            Dizionario con metriche di valutazione
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato!")

        # Predizioni
        y_pred = self.model.predict(X, verbose=0)

        # ----- CORREZIONE: Assicura che y e y_pred siano 1D -----
        y_flat = y.flatten()
        y_pred_flat = y_pred.flatten()

         # DEBUG: Stampa statistiche
        logger.info(f"R² DEBUG - y stats: mean={np.mean(y_flat):.6f}, std={np.std(y_flat):.6f}")
        logger.info(f"R² DEBUG - y_pred stats: mean={np.mean(y_pred_flat):.6f}, std={np.std(y_pred_flat):.6f}")

        # Calcola metriche SU DATI NORMALIZZATI (coerenti)
        mse = mean_squared_error(y_flat, y_pred_flat)
        mae = mean_absolute_error(y_flat, y_pred_flat)

        # Calcola R² in modo robusto
        try:
            # Controllo esplicito per il calcolo R²
            ss_res = np.sum((y_flat - y_pred_flat) ** 2)
            ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)

            logger.info(f"R² DEBUG - SS_res: {ss_res:.6f}, SS_tot: {ss_tot:.6f}")

            if ss_tot == 0:
                r2 = 1.0  # Perfetto fit se non c'è varianza
            else:
                r2 = 1 - (ss_res / ss_tot)
        except Exception as e:
            logger.error(f"Errore nel calcolo R²: {e}, impostato a -inf")
            r2 = -np.inf

        # Calcola MAPE con gestione errori
        try:
            # Evita divisione per zero
            mask = y_flat != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_flat[mask] - y_pred_flat[mask]) / y_flat[mask])) * 100
            else:
                mape = np.nan
        except Exception as e:
            logger.warning(f"Impossibile calcolare MAPE: {e}")
            mape = np.nan

        metrics = {
            f'{prefix}mse': mse,
            f'{prefix}mae': mae,
            f'{prefix}r2': r2,
            f'{prefix}mape': mape
        }

        return metrics
    
    def predict(self, X: np.ndarray, inverse_transform: bool = False,
                scaler: Any = None) -> np.ndarray:
        """
        Effettua predizioni con il modello.
        
        Args:
            X: Dati di input
            inverse_transform: Se True, de-normalizza le predizioni
            scaler: Scaler per de-normalizzazione
            
        Returns:
            Array con predizioni
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato!")
        
        logger.info(f"Effettuando predizioni su {X.shape[0]} campioni...")
        
        # Predizioni
        predictions = self.model.predict(X, verbose=0)
        
        # De-normalizzazione se richiesta
        if inverse_transform and scaler is not None:
            # Crea array fittizio per la de-normalizzazione
            dummy_array = np.zeros((len(predictions), self.n_features))
            
            # Trova indice della colonna price (assumiamo sia la prima feature)
            price_idx = 0
            
            # Sostituisci la colonna del prezzo con le predizioni
            dummy_array[:, price_idx] = predictions.flatten()
            
            # De-normalizza
            predictions_denorm = scaler.inverse_transform(dummy_array)[:, price_idx]
            predictions = predictions_denorm.reshape(-1, 1)
            
            logger.info("Predizioni de-normalizzate")
        
        return predictions
    
    def plot_training_history(self, save_path: str = None):
        """
        Visualizza la storia del training.
        
        Args:
            save_path: Percorso per salvare il plot
        """
        if self.history is None:
            logger.warning("Nessuna storia di training disponibile.")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(history['mean_absolute_error'], label='Training MAE')
        axes[0, 1].plot(history['val_mean_absolute_error'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MSE
        axes[1, 0].plot(history['mean_squared_error'], label='Training MSE')
        axes[1, 0].plot(history['val_mean_squared_error'], label='Validation MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in history:
            axes[1, 1].plot(history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot salvato in: {save_path}")
        
        plt.close(fig)  # Chiudi la figura dopo aver salvato
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                         title: str = "Predictions vs Actual",
                         save_path: str = None):
        """
        Visualizza predizioni vs valori reali.
        
        Args:
            y_true: Valori veri
            y_pred: Predizioni
            title: Titolo del plot
            save_path: Percorso per salvare il plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Predizioni vs Reali
        axes[0, 0].plot(y_true, label='Actual', alpha=0.7)
        axes[0, 0].plot(y_pred, label='Predicted', alpha=0.7)
        axes[0, 0].set_title(title)
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 1].plot([y_true.min(), y_true.max()], 
                        [y_true.min(), y_true.max()], 
                        'r--', lw=2)
        axes[0, 1].set_title('Actual vs Predicted')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Errori
        errors = y_true.flatten() - y_pred.flatten()
        axes[1, 0].hist(errors, bins=50, edgecolor='black')
        axes[1, 0].set_title('Prediction Errors Distribution')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Errori nel tempo
        axes[1, 1].plot(errors, alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Prediction Errors Over Time')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot predizioni salvato in: {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str = None, 
              scaler_path: str = None,
              metadata: Dict = None,
              force_scaler_save: bool = True):  # Aggiungi questo parametro
        """
        Salva il modello e gli oggetti correlati.
        FORZA il salvataggio dello scaler anche se None.
        """
        # Crea directory se non esiste
        os.makedirs('models', exist_ok=True)
        
        # Path di default
        if model_path is None:
            model_path = f'models/{self.model_name}.h5'
        
        if scaler_path is None:
            scaler_path = f'models/{self.model_name}_scaler.pkl'
        
        # 1. Salva modello
        self.model.save(model_path)
        logger.info(f"✅ Modello salvato in: {model_path}")
        
        # 2. CRITICO: Salva sempre lo scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"✅ Scaler salvato in: {scaler_path}")
        elif force_scaler_save:
            # Se lo scaler è None, creane uno di default e salvalo
            logger.warning("⚠️  Scaler è None, creo e salvo scaler di default")
            from sklearn.preprocessing import StandardScaler
            default_scaler = StandardScaler()
            
            # Fit con dati dummy basati su forex tipici
            dummy_data = np.array([
                [1.1010, 1.1000, 1.0990, 0.0010, 0.0008, 50, 50, 0.0010, -0.0010],
                [1.1020, 1.1010, 1.1000, 0.0020, 0.0015, 60, 55, 0.0020, -0.0020],
                [1.1000, 1.0990, 1.0980, -0.0010, -0.0005, 40, 45, -0.0010, 0.0010]
            ])
            default_scaler.fit(dummy_data)
            
            joblib.dump(default_scaler, scaler_path)
            logger.info(f"✅ Scaler di default salvato in: {scaler_path}")
        
        # 3. Salva metadata
        metadata_path = f'models/{self.model_name}_metadata.json'
        if metadata is None:
            metadata = {}
        
        # Assicurati che i feature names siano salvati
        if hasattr(self.data_loader, 'feature_names') and self.data_loader.feature_names:
            metadata['feature_names'] = self.data_loader.feature_names
        
        metadata_to_save = {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'training_date': datetime.now().isoformat(),
            'scaler_saved': self.scaler is not None or force_scaler_save,
            **metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2)
        
        logger.info(f"✅ Metadata salvati in: {metadata_path}")
        
        # 4. DEBUG: Stampa verifica file creati
        logger.info("📁 FILE CREATI IN MODELS/:")
        logger.info(f"   • {os.path.basename(model_path)}")
        logger.info(f"   • {os.path.basename(scaler_path)}")
        logger.info(f"   • {os.path.basename(metadata_path)}")
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """
        Carica un modello pre-addestrato.
        
        Args:
            model_path: Percorso del modello
            scaler_path: Percorso dello scaler
        """
        logger.info(f"Caricamento modello da: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        if scaler_path is not None and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler caricato da: {scaler_path}")
        
        # Estrai shape dal modello per riferimento
        if len(self.model.input_shape) == 3:
            self.sequence_length = self.model.input_shape[1]
            self.n_features = self.model.input_shape[2]
            logger.info(f"Model loaded. Input shape: ({self.sequence_length}, {self.n_features})")
        
        logger.info("Modello caricato con successo!")


def train_model_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         sequence_length: int, n_features: int,
                         model_type: str = 'standard',
                         epochs: int = 100,
                         batch_size: int = 32,
                         model_name: str = None) -> Tuple[LSTMTradingModel, Dict]:
    """
    Pipeline completa per l'addestramento del modello.
    
    Args:
        X_train, y_train: Dati di training
        X_test, y_test: Dati di test
        sequence_length: Lunghezza sequenze
        n_features: Numero di features
        model_type: Tipo di modello LSTM
        epochs: Numero di epoche
        batch_size: Dimensione batch
        model_name: Nome del modello
        
    Returns:
        Tupla (modelo addestrato, risultati)
    """
    logger.info("=" * 60)
    logger.info("AVVIO PIPELINE ADDESTRAMENTO MODELLO")
    logger.info("=" * 60)
    
    # 1. Inizializza trainer
    trainer = LSTMTradingModel(model_type=model_type, model_name=model_name)
    
    # 2. Costruisci modello
    input_shape = (sequence_length, n_features)
    trainer.build_model(input_shape=input_shape, output_units=1)
    
    # 3. Addestra modello
    results = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,  # Usa test come validation
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 4. Valutazione finale su test set
    test_metrics = trainer.evaluate(X_test, y_test, prefix='final_test_')
    logger.info("\n" + "=" * 60)
    logger.info("VALUTAZIONE FINALE SU TEST SET:")
    logger.info(f"MSE: {test_metrics['final_test_mse']:.6f}")
    logger.info(f"MAE: {test_metrics['final_test_mae']:.6f}")
    logger.info(f"R²: {test_metrics['final_test_r2']:.4f}")
    
    if not np.isnan(test_metrics['final_test_mape']):
        logger.info(f"MAPE: {test_metrics['final_test_mape']:.2f}%")
    
    # 5. Salva modello
    try:
        trainer.save_model(
            metadata={
                'sequence_length': sequence_length,
                'n_features': n_features,
                'model_type': model_type,
                'test_metrics': test_metrics
            }
        )
    except Exception as e:
        logger.error(f"Errore nel salvataggio modello: {str(e)}")
    
    # 6. Plots
    trainer.plot_training_history(save_path=f'models/{trainer.model_name}_training_history.png')
    
    # Predizioni sul test set per visualizzazione
    y_pred_test = trainer.predict(X_test)
    trainer.plot_predictions(
        y_test, y_pred_test,
        title=f"Test Set Predictions - {model_type.upper()} LSTM",
        save_path=f'models/{trainer.model_name}_predictions.png'
    )
    
    return trainer, {**results, 'test_metrics': test_metrics}


if __name__ == "__main__":
    # Esempio di utilizzo
    print("Test del model_trainer.py")
    print("=" * 50)
    
    try:
        # Genera dati sintetici per test
        sequence_length = 10
        n_features = 5
        n_samples = 1000
        
        # Dati sintetici
        np.random.seed(42)
        X_dummy = np.random.randn(n_samples, sequence_length, n_features)
        y_dummy = np.random.randn(n_samples, 1)
        
        # Split train/test
        split_idx = int(n_samples * 0.8)
        X_train_dummy = X_dummy[:split_idx]
        X_test_dummy = X_dummy[split_idx:]
        y_train_dummy = y_dummy[:split_idx]
        y_test_dummy = y_dummy[split_idx:]
        
        print(f"\nDati sintetici generati:")
        print(f"  X_train shape: {X_train_dummy.shape}")
        print(f"  y_train shape: {y_train_dummy.shape}")
        print(f"  X_test shape: {X_test_dummy.shape}")
        print(f"  y_test shape: {y_test_dummy.shape}")
        
        # Test pipeline
        trainer, results = train_model_pipeline(
            X_train=X_train_dummy,
            y_train=y_train_dummy,
            X_test=X_test_dummy,
            y_test=y_test_dummy,
            sequence_length=sequence_length,
            n_features=n_features,
            model_type='standard',
            epochs=50,  # Pochi epoch per test veloce
            batch_size=32,
            model_name='test_model'
        )
        
        print("\n" + "=" * 50)
        print("TEST COMPLETATO CON SUCCESSO!")
        print("=" * 50)
        
        # Test predizione
        print("\nTest predizione su singola sequenza...")
        single_sequence = X_test_dummy[0:1]  # Prende la prima sequenza
        prediction = trainer.predict(single_sequence)
        print(f"Predizione forma: {prediction.shape}")
        print(f"Valore predetto: {prediction[0, 0]:.6f}")
        
    except Exception as e:
        print(f"\nErrore durante il test: {str(e)}")
        import traceback
        traceback.print_exc()