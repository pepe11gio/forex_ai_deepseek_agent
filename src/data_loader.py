"""
data_loader.py
Carica TUTTI i file CSV dalla directory data/.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import logging
from timestamp_features import TimestampFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataLoader:
    """Carica e preprocessa TUTTI i dati finanziari."""
    
    def __init__(self, sequence_length: int = 20, test_size: float = 0.2, 
                 random_state: int = 42, lookahead: int = 10):
        self.timestamp_extractor = TimestampFeatureExtractor()
        self.include_time_features = True  # Nuovo flag
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.random_state = random_state
        self.lookahead = lookahead
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_columns = None
        self.target_column = 'price'
        self.price_data_original = None
    
    def load_all_csvs(self, data_dir: str) -> pd.DataFrame:
        """
        Carica TUTTI i file CSV dalla directory.
        
        Args:
            data_dir: Directory con file CSV
            
        Returns:
            DataFrame combinato
        """
        import glob
        
        csv_files = glob.glob(f"{data_dir}/*.csv")
        
        if not csv_files:
            raise FileNotFoundError(f"Nessun file CSV trovato in {data_dir}")
        
        logger.info(f"Caricamento {len(csv_files)} file CSV...")
        
        all_dataframes = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"  ✓ {csv_file}: {len(df)} righe")
                all_dataframes.append(df)
            except Exception as e:
                logger.error(f"  ✗ {csv_file}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("Nessun file CSV caricato con successo")
        
        # Combina tutti i DataFrame
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Ordina per timestamp
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            combined_df = combined_df.sort_values('timestamp')
        
        logger.info(f"✅ Dati combinati: {len(combined_df)} righe totali")
        
        return combined_df
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_column: str = 'price',
                        exclude_columns: List[str] = None) -> pd.DataFrame:
        """Prepara le features INCLUSO timestamp."""
        self.target_column = target_column
        
        if exclude_columns is None:
            exclude_columns = []
        
        # 1. Salva timestamp se presente
        if 'timestamp' in df.columns:
            self.timestamps = df['timestamp']
            
            # Estrai features temporali
            if self.include_time_features:
                time_features = self.timestamp_extractor.extract_features(df['timestamp'])
                df = pd.concat([df, time_features], axis=1)
                logger.info(f"Aggiunte {len(time_features.columns)} features temporali")
        
        # 2. Identifica colonne features (escludi timestamp e target)
        self.feature_columns = [col for col in df.columns 
                               if col not in exclude_columns + ['timestamp', self.target_column]]
        
        # 3. Se non ci sono indicatori, usa solo prezzo
        if not self.feature_columns:
            self.feature_columns = [self.target_column]
        
        self.feature_names = self.feature_columns.copy()
        
        logger.info(f"Features totali: {len(self.feature_columns)}")
        logger.info(f"Prime 5 features: {self.feature_columns[:5]}")
        
        return df
    
    def create_sequences(self, data: np.ndarray, target_data: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Crea sequenze temporali."""
        X, y = [], []
        
        n_samples = len(data) - self.sequence_length - self.lookahead
        
        if n_samples <= 0:
            raise ValueError(f"Dati insufficienti: {len(data)} righe")
        
        for i in range(n_samples):
            # Features: ultimi N tick
            sequence = data[i:i + self.sequence_length]
            X.append(sequence)
            
            # Target: variazione % nei prossimi M tick
            current_idx = i + self.sequence_length - 1
            price_idx = 0  # Prima colonna = prezzo
            
            if target_data is not None:
                current_price = target_data[current_idx]
                future_prices = target_data[current_idx + 1:current_idx + 1 + self.lookahead]
            else:
                current_price = data[current_idx, price_idx]
                future_prices = data[current_idx + 1:current_idx + 1 + self.lookahead, price_idx]
            
            if len(future_prices) > 0 and current_price != 0:
                # Calcola massima variazione %
                changes = [(future_price - current_price) / current_price * 100 
                          for future_price in future_prices]
                target_value = np.mean(changes)
            else:
                target_value = 0.0
            
            y.append(target_value)
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalizza i dati."""
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != 'timestamp']
            self.feature_names = self.feature_columns.copy()
        
        df_normalized = df.copy()
        
        if fit:
            df_normalized[self.feature_columns] = self.scaler.fit_transform(
                df_normalized[self.feature_columns]
            )
            logger.info("Scaler fittato sui dati")
        else:
            df_normalized[self.feature_columns] = self.scaler.transform(
                df_normalized[self.feature_columns]
            )
            logger.info("Scaler applicato (transform)")
        
        return df_normalized
    
    def process_all_data(self, data_dir: str,
                        target_column: str = 'price',
                        exclude_columns: List[str] = None) -> Tuple:
        """
        Processa TUTTI i dati.
        """
        # 1. Carica TUTTI i file
        df = self.load_all_csvs(data_dir)
        
        # 2. Prepara features
        df = self.prepare_features(df, target_column, exclude_columns)
        
        # 🔥 SALVA COPIA DEI DATI ORIGINALI
        self.df_original = df.copy()
        self.price_original = df[target_column].values

         # 3. Normalizza
        df_normalized = self.normalize_data(df, fit=True)
        
        # 4. Prezzi originali per target
        price_original = df[target_column].values
        
        # 5. Crea sequenze
        features_data = df_normalized[self.feature_columns].values
        X, y = self.create_sequences(features_data, price_original)
        
        # 6. Normalizza target
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        self.y_scaler = y_scaler
        
        # 7. Split temporale
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"✅ Processamento completato")
        logger.info(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        # 🔥 CORREZIONE: restituisci self invece di loader
        return X_train, X_test, y_train, y_test, df, df_normalized, self  # <-- CORRETTO

def load_and_prepare_data(filepath: str, sequence_length: int = 20, 
                         test_size: float = 0.2, lookahead: int = 10) -> Tuple:
    """Funzione di utilità."""
    loader = FinancialDataLoader(
        sequence_length=sequence_length,
        test_size=test_size,
        lookahead=lookahead
    )
    
    # Usa directory se è una directory
    if os.path.isdir(filepath):
        # 🔥 process_all_data restituisce 7 valori
        results = loader.process_all_data(
            data_dir=filepath,
            target_column='price',
            exclude_columns=['timestamp']
        )
        
        # 🔥 VERIFICA quanti valori restituisce
        if len(results) == 7:
            X_train, X_test, y_train, y_test, df, df_normalized, loader = results
            return (X_train, X_test, y_train, y_test, df, df_normalized, loader)
        elif len(results) == 6:
            # Vecchia versione
            X_train, X_test, y_train, y_test, df, df_normalized = results
            return (X_train, X_test, y_train, y_test, df, df_normalized, loader)
        else:
            raise ValueError(f"process_all_data ha restituito {len(results)} valori, attesi 6 o 7")
    else:
        # File singolo
        df = pd.read_csv(filepath)
        df = loader.prepare_features(df, 'price', ['timestamp'])
        df_normalized = loader.normalize_data(df, fit=True)
        
        price_original = df['price'].values
        features_data = df_normalized[loader.feature_columns].values
        
        X, y = loader.create_sequences(features_data, price_original)
        
        # Normalizza target
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        loader.y_scaler = y_scaler
        
        # Split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 🔥 Restituisci 7 valori
        return (X_train, X_test, y_train, y_test, df, df_normalized, loader)

def create_sequences_with_tp_sl_target(self, data: np.ndarray, price_data: np.ndarray, 
                                      tp_pips: int = 30, sl_pips: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea sequenze con target binario: 1=TP raggiunto, 0=SL raggiunto
    """
    X, y = [], []
    
    n_samples = len(data) - self.sequence_length - max(tp_pips, sl_pips) * 10  # Buffer sicuro
    
    pip_value = 0.0001  # Per EUR/USD
    
    for i in range(n_samples):
        # Features: ultimi N tick
        sequence = data[i:i + self.sequence_length]
        X.append(sequence)
        
        # Prezzo corrente
        current_idx = i + self.sequence_length - 1
        current_price = price_data[current_idx]
        
        # Analizza prossimi N pips per vedere se TP o SL viene raggiunto prima
        future_prices = price_data[current_idx + 1:current_idx + 1 + (max(tp_pips, sl_pips) * 10)]
        
        tp_reached = False
        sl_reached = False
        
        for future_price in future_prices:
            # Check TP (per BUY - assumiamo sempre long per training)
            if future_price >= current_price + (tp_pips * pip_value):
                tp_reached = True
                break
            
            # Check SL (per BUY)
            if future_price <= current_price - (sl_pips * pip_value):
                sl_reached = True
                break
        
        # Target: 1=TP raggiunto, 0=SL raggiunto
        if tp_reached:
            y.append(1)  # SUCCESSO: TP raggiunto
        elif sl_reached:
            y.append(0)  # FALLIMENTO: SL raggiunto
        else:
            y.append(0.5)  # NEUTRAL: nessuno dei due (ma raro con buffer sufficiente)
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Test
    loader = FinancialDataLoader()
    X_train, X_test, y_train, y_test, df, df_norm, loader = loader.process_all_data("data")  # <-- Aggiornato
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
