"""
timestamp_features.py
Conversione timestamp in features numeriche per ML
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class TimestampFeatureExtractor:
    """Estrae features numeriche dai timestamp."""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, timestamp_series: pd.Series) -> pd.DataFrame:
        """
        Converte serie di timestamp in features numeriche.
        """
        features = {}
        
        # Converti a datetime
        timestamps = pd.to_datetime(timestamp_series)
        
        # 1. Features temporali di base
        features['hour'] = timestamps.dt.hour
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        features['day_of_week'] = timestamps.dt.dayofweek
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        features['day_of_month'] = timestamps.dt.day
        features['month'] = timestamps.dt.month
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # 2. Sessioni di trading
        # London: 8-16 UTC
        features['is_london_session'] = ((features['hour'] >= 8) & (features['hour'] <= 16)).astype(int)
        # New York: 13-21 UTC
        features['is_ny_session'] = ((features['hour'] >= 13) & (features['hour'] <= 21)).astype(int)
        # Overlap London-NY: 13-16 UTC
        features['is_overlap_session'] = ((features['hour'] >= 13) & (features['hour'] <= 16)).astype(int)
        
        # 3. Fine settimana/feriali
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)
        
        # 4. Orari specifici importanti
        features['is_market_open'] = ((features['hour'] >= 0) & (features['hour'] <= 21)).astype(int)
        features['is_asia_session'] = ((features['hour'] >= 0) & (features['hour'] <= 8)).astype(int)
        
        # 5. Minuto del giorno (per granularità)
        features['minute_of_day'] = timestamps.dt.hour * 60 + timestamps.dt.minute
        features['mod_sin'] = np.sin(2 * np.pi * features['minute_of_day'] / 1440)
        features['mod_cos'] = np.cos(2 * np.pi * features['minute_of_day'] / 1440)
        
        # 6. Quarter dell'ora (per pattern di 15min)
        features['quarter_hour'] = timestamps.dt.minute // 15
        
        # Crea DataFrame
        features_df = pd.DataFrame(features)
        self.feature_names = features_df.columns.tolist()
        
        logger.info(f"Estratte {len(self.feature_names)} features temporali")
        
        return features_df
    
    def get_time_based_scaling(self, timestamps: pd.Series) -> Dict[str, float]:
        """
        Restituisce fattori di scaling basati sull'ora.
        """
        hour = timestamps.iloc[-1].hour if len(timestamps) > 0 else 12
        
        # Volatilità tipica per ora
        volatility_factors = {
            'night_asia': 0.3,    # 00-08
            'london_open': 1.2,   # 08-09  
            'london_mid': 0.8,    # 09-13
            'overlap': 1.5,       # 13-16 (London-NY overlap)
            'ny_mid': 0.9,        # 16-20
            'ny_close': 1.1,      # 20-21
            'night': 0.2          # 21-00
        }
        
        # Determina fascia
        if 0 <= hour < 8:
            factor = volatility_factors['night_asia']
        elif 8 <= hour < 9:
            factor = volatility_factors['london_open']
        elif 9 <= hour < 13:
            factor = volatility_factors['london_mid']
        elif 13 <= hour < 16:
            factor = volatility_factors['overlap']
        elif 16 <= hour < 20:
            factor = volatility_factors['ny_mid']
        elif 20 <= hour < 21:
            factor = volatility_factors['ny_close']
        else:
            factor = volatility_factors['night']
        
        return {
            'volatility_factor': factor,
            'position_size_multiplier': factor,
            'hour': hour,
            'session': self._get_session_name(hour)
        }
    
    def _get_session_name(self, hour: int) -> str:
        """Restituisce nome sessione trading."""
        if 0 <= hour < 8:
            return "ASIA"
        elif 8 <= hour < 13:
            return "LONDON"
        elif 13 <= hour < 16:
            return "OVERLAP"
        elif 16 <= hour < 21:
            return "NEW_YORK"
        else:
            return "NIGHT"