"""
data_loader.py
Modulo per il caricamento e preprocessing dei dati finanziari da file CSV.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """
    Classe per il caricamento e preprocessing dei dati finanziari.
    Gestisce la lettura di file CSV, normalizzazione e creazione di sequenze per LSTM.
    """
    
    def __init__(self, sequence_length: int = 10, test_size: float = 0.2, 
                 random_state: int = 42, lookahead: int = 10):
        """
        Inizializza il data loader.
        
        Args:
            sequence_length: Lunghezza delle sequenze temporali (default: 10)
            test_size: Percentuale di dati per il test set (default: 0.2)
            random_state: Seed per la riproducibilità (default: 42)
        """
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.random_state = random_state
        self.lookahead = lookahead 
        self.scaler = StandardScaler()
        self.feature_names = []  # Inizializza lista vuota
        self.feature_columns = None
        self.target_column = 'price'
        
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Carica un file CSV con dati finanziari.
        
        Args:
            filepath: Percorso del file CSV
            
        Returns:
            DataFrame pandas con i dati
            
        Raises:
            FileNotFoundError: Se il file non esiste
            ValueError: Se il file non ha le colonne richieste
        """
        try:
            logger.info(f"Caricamento file CSV: {filepath}")
            df = pd.read_csv(filepath)
            
            # Verifica colonne minime richieste
            required_columns = ['timestamp', 'price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Colonne mancanti nel CSV: {missing_columns}")
            
            logger.info(f"Dati caricati: {df.shape[0]} righe, {df.shape[1]} colonne")
            logger.info(f"Colonne presenti: {list(df.columns)}")
            
            # Converti timestamp se presente
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                logger.info(f"Periodo dati: da {df['timestamp'].min()} a {df['timestamp'].max()}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File non trovato: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Errore nel caricamento del CSV: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_column: str = 'price',
                        exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Prepara le features per il training.
        
        Args:
            df: DataFrame originale
            target_column: Nome della colonna target (default: 'price')
            exclude_columns: Colonne da escludere dal preprocessing
            
        Returns:
            DataFrame con features preparate
        """
        logger.info("Preparazione features...")
        
        # Imposta colonna target
        self.target_column = target_column
        
        # Colonne da escludere
        if exclude_columns is None:
            exclude_columns = ['timestamp']
        
        # Identifica colonne features (tutto tranne target ed escluse)
        self.feature_columns = [col for col in df.columns 
                               if col not in exclude_columns + [self.target_column]]
        
        # Imposta feature_names
        self.feature_names = self.feature_columns.copy()
        
        # Se non ci sono indicatori, usa solo il prezzo come feature
        if not self.feature_columns:
            self.feature_columns = [self.target_column]
            self.feature_names = [self.target_column]
            logger.warning("Nessun indicatore trovato. Userò solo il prezzo come feature.")
        
        logger.info(f"Colonne features: {self.feature_columns}")
        logger.info(f"Colonna target: {self.target_column}")
        logger.info(f"Nomi features: {self.feature_names}")
        
        return df
    
    def create_sequences(self, data: np.ndarray, target_data: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea sequenze temporali per l'input LSTM.
        MODIFICA: Predice la massima variazione % nei prossimi self.lookahead tick
        CON BILANCIAMENTO per correggere bias positivo
        """
        logger.info(f"Creazione sequenze con lunghezza: {self.sequence_length}")
        logger.info(f"Lookahead per target: {self.lookahead} tick futuri")
        
        X, y = [], []
        
        # Calcola quanti campioni possiamo creare
        n_samples = len(data) - self.sequence_length - self.lookahead
        
        if n_samples <= 0:
            raise ValueError(
                f"Non abbastanza dati. "
                f"Servono almeno: sequence_length({self.sequence_length}) + "
                f"lookahead({self.lookahead}) = {self.sequence_length + self.lookahead} campioni, "
                f"ma ne ho solo {len(data)}"
            )
        
        # Contatori per debug
        stats = {
            'positive': 0,
            'negative': 0,
            'zero': 0,
            'total_changes': [],
            'adjusted_changes': []
        }
        
        for i in range(n_samples):
            # 1. Features: ultimi N tick
            sequence = data[i:i + self.sequence_length]
            X.append(sequence)
            
            # 2. Target: massima variazione % nei prossimi M tick CON BILANCIAMENTO
            # 2a. Trova indice del prezzo corrente (ultimo della sequenza)
            current_idx = i + self.sequence_length - 1
            
            # 2b. Prezzo corrente (assume che la prima colonna sia il prezzo)
            price_idx = 0  # Prima colonna = prezzo
            
            if target_data is not None:
                # Usa target_data separato se fornito
                current_price = target_data[current_idx]
                future_prices = target_data[current_idx + 1:current_idx + 1 + self.lookahead]
            else:
                # Usa la colonna prezzo dai dati features
                current_price = data[current_idx, price_idx]
                future_prices = data[current_idx + 1:current_idx + 1 + self.lookahead, price_idx]
            
            # 2c. Calcola massima variazione % con bilanciamento
            target_value = 0.0  # Default
            
            if len(future_prices) > 0 and current_price != 0:
                # Calcola tutte le variazioni % possibili
                changes = []
                for future_price in future_prices:
                    change_pct = ((future_price - current_price) / current_price) * 100
                    changes.append(change_pct)
                
                changes_array = np.array(changes)
                stats['total_changes'].extend(changes_array.tolist())
                
                # SOLUZIONE 1: BILANCIAMENTO DEL TARGET
                # Opzione A: Usa la MEDIA delle variazioni (più stabile, meno bias)
                avg_change = np.mean(changes_array)
                
                # Opzione B: Media ponderata - dà più peso ai movimenti significativi
                # Pesa di più i movimenti più grandi (in valore assoluto)
                weights = np.abs(changes_array)
                if np.sum(weights) > 0:
                    weighted_avg = np.average(changes_array, weights=weights)
                else:
                    weighted_avg = avg_change
                
                # Opzione C: Mediana (robusta agli outliers)
                median_change = np.median(changes_array)
                
                # SCEGLI LA STRATEGIA (raccomando Opzione B per trading)
                # 1. avg_change - più stabile ma meno sensibile a trend forti
                # 2. weighted_avg - buon compromesso, cattura trend ma meno rumoroso
                # 3. median_change - robusta ma può perdere segnali
                target_value = weighted_avg  # <-- MODIFICA QUI PER TESTARE
                
                # DEBUG: Opzionale per vedere la differenza
                if i < 3:  # Solo per i primi 3 campioni
                    logger.debug(f"Esempio {i}:")
                    logger.debug(f"  Cambiamenti: {changes_array[:3]}...")
                    logger.debug(f"  Media: {avg_change:.4f}%, Ponderata: {weighted_avg:.4f}%, Mediana: {median_change:.4f}%")
                    logger.debug(f"  Scelta: {target_value:.4f}%")
                
                stats['adjusted_changes'].append(target_value)
                
                # Conta segno
                if target_value > 0.01:  # > 0.01% considerato positivo
                    stats['positive'] += 1
                elif target_value < -0.01:  # < -0.01% considerato negativo
                    stats['negative'] += 1
                else:
                    stats['zero'] += 1
            else:
                # Nessun dato futuro disponibile
                target_value = 0.0
                stats['zero'] += 1
            
            y.append(target_value)
        
        X = np.array(X)
        y_array = np.array(y)
        
        # ANALISI DETTAGLIATA DEL TARGET
        logger.info("=" * 60)
        logger.info("📊 ANALISI COMPLETA DEL NUOVO TARGET (BILANCIATO)")
        logger.info("=" * 60)
        
        # Statistiche base
        logger.info(f"Statistiche target FINALE:")
        logger.info(f"  Media: {y_array.mean():.6f}%")
        logger.info(f"  Mediana: {np.median(y_array):.6f}%")
        logger.info(f"  Std: {y_array.std():.6f}%")
        logger.info(f"  Min: {y_array.min():.6f}%, Max: {y_array.max():.6f}%")
        logger.info(f"  Intervallo: [{y_array.min():.4f}%, {y_array.max():.4f}%]")
        
        # Distribuzione segni
        total_samples = len(y_array)
        logger.info(f"\nDistribuzione segni:")
        logger.info(f"  Positivi (>0.01%): {stats['positive']} campioni ({stats['positive']/total_samples*100:.1f}%)")
        logger.info(f"  Negativi (<-0.01%): {stats['negative']} campioni ({stats['negative']/total_samples*100:.1f}%)")
        logger.info(f"  Quasi-zero (±0.01%): {stats['zero']} campioni ({stats['zero']/total_samples*100:.1f}%)")
        
        # Distribuzione per intervalli
        bins = [-np.inf, -2.0, -1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0, 2.0, np.inf]
        bin_labels = ["<-2%", "-2:-1%", "-1:-0.5%", "-0.5:-0.2%", "-0.2:-0.05%", 
                    "-0.05:0.05%", "0.05:0.2%", "0.2:0.5%", "0.5:1%", "1:2%", ">2%"]
        
        hist, bin_edges = np.histogram(y_array, bins=bins)
        
        logger.info(f"\nDistribuzione per intervalli:")
        for label, count, low, high in zip(bin_labels, hist, bin_edges[:-1], bin_edges[1:]):
            if count > 0:
                percentage = (count / total_samples) * 100
                logger.info(f"  {label:12} {count:4d} campioni ({percentage:5.1f}%)")
        
        # Skewness e Kurtosis (misure di asimmetria)
        from scipy import stats as sp_stats
        if len(y_array) > 1:
            skewness = sp_stats.skew(y_array)
            kurtosis = sp_stats.kurtosis(y_array)
            
            logger.info(f"\nAnalisi forma distribuzione:")
            logger.info(f"  Skewness: {skewness:.4f}")
            logger.info(f"    {self._interpret_skewness(skewness)}")
            logger.info(f"  Kurtosis: {kurtosis:.4f}")
            logger.info(f"    {self._interpret_kurtosis(kurtosis)}")
        
        # Informazioni utili per il modello
        logger.info(f"\n🔍 INFORMAZIONI PER IL MODELLO:")
        logger.info(f"  Target idealmente dovrebbe avere:")
        logger.info(f"    • Media vicina a 0.0 (ora: {y_array.mean():.6f})")
        logger.info(f"    • Skewness vicina a 0.0 (simmetria)")
        logger.info(f"    • Kurtosis tra -1 e 1 (non troppo picchiata)")
        
        logger.info(f"\n✅ Sequenze create: {X.shape[0]}")
        logger.info(f"   X shape: {X.shape}")
        logger.info(f"   y shape: {y_array.shape}")
        
        return X, y_array

    def _interpret_skewness(self, skewness: float) -> str:
        """Interpreta il valore di skewness."""
        if skewness < -0.5:
            return "Distribuzione con coda a sinistra (più valori negativi estremi)"
        elif skewness < -0.1:
            return "Leggermente asimmetrica a sinistra"
        elif -0.1 <= skewness <= 0.1:
            return "Quasi perfettamente simmetrica (IDEALE)"
        elif skewness <= 0.5:
            return "Leggermente asimmetrica a destra"
        else:
            return "Distribuzione con coda a destra (più valori positivi estremi)"

    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpreta il valore di kurtosis."""
        if kurtosis < -1.0:
            return "Distribuzione molto piatta (platykurtic)"
        elif kurtosis < -0.5:
            return "Distribuzione piatta"
        elif -0.5 <= kurtosis <= 0.5:
            return "Normale (mesokurtic, IDEALE)"
        elif kurtosis <= 1.0:
            return "Distribuzione appuntita"
        else:
            return "Distribuzione molto appuntita con code pesanti (leptokurtic)"
    
    def split_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Suddivide le sequenze in train e test set mantenendo l'ordine temporale.
        
        Args:
            X: Sequenze di input
            y: Valori target
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Calcola indice di split mantenendo l'ordine temporale
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Split temporale:")
        logger.info(f"  Train set: {X_train.shape[0]} sequenze")
        logger.info(f"  Test set: {X_test.shape[0]} sequenze")
        
        return X_train, X_test, y_train, y_test
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalizza i dati usando StandardScaler.
        
        Args:
            df: DataFrame con dati originali
            fit: Se True, fit dello scaler sui dati
            
        Returns:
            DataFrame normalizzato
        """
        logger.info("Normalizzazione dati...")
        
        # Seleziona solo le colonne features per la normalizzazione
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != 'timestamp']
            self.feature_names = self.feature_columns.copy()
        
        # Crea copia per non modificare l'originale
        df_normalized = df.copy()
        
        # Normalizza le features
        if fit:
            df_normalized[self.feature_columns] = self.scaler.fit_transform(
                df_normalized[self.feature_columns]
            )
            logger.info("Scaler fittato sui dati")
        else:
            df_normalized[self.feature_columns] = self.scaler.transform(
                df_normalized[self.feature_columns]
            )
            logger.info("Scaler applicato in transform mode")
        
        return df_normalized
    
    def get_inverse_scaler(self):
        """Restituisce lo scaler per de-normalizzare i dati."""
        return self.scaler
    
    def process_pipeline(self, filepath: str, 
                    target_column: str = 'price',
                    exclude_columns: List[str] = None) -> Tuple:
        """
        Pipeline completa di processing dei dati.
        MODIFICA: Supporta nuovo target di massima variazione %
        """
        # 1. Carica dati
        df = self.load_csv(filepath)
        
        # 2. Prepara features (colonne normali)
        df = self.prepare_features(df, target_column, exclude_columns)
        
        # 3. Normalizza TUTTE le features (ma non il prezzo per ora)
        df_normalized = self.normalize_data(df, fit=True)
        
        # 4. Per il target, abbiamo bisogno del prezzo ORIGINALE non normalizzato
        # per calcolare variazioni % corrette
        price_original = df[target_column].values  # Prezzi originali
        
        # 5. Usa features normalizzate ma prezzi originali per calcolo target
        features_data = df_normalized[self.feature_columns].values
        
        # 6. Crea sequenze con il nuovo target
        X, y = self.create_sequences(features_data, price_original)
        
        # 7. Split train/test
        X_train, X_test, y_train, y_test = self.split_sequences(X, y)
        
        # 8. Ora normalizza ANCHE y (dopo averlo calcolato)
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Salva lo scaler per y per de-normalizzare dopo
        self.y_scaler = y_scaler
        
        logger.info(f"✅ Pipeline completata con nuovo target")
        logger.info(f"   X_train shape: {X_train.shape}")
        logger.info(f"   y_train shape: {y_train.shape} (target normalizzato)")
        logger.info(f"   Target originale - Media: {np.mean(y):.4f}%, Std: {np.std(y):.4f}%")
        logger.info(f"   Target normalizzato - Media: {y_train.mean():.6f}, Std: {y_train.std():.6f}")
        
        return X_train, X_test, y_train, y_test, df, df_normalized
    
    def prepare_for_prediction(self, df: pd.DataFrame, scaler: StandardScaler = None) -> np.ndarray:
        """
        Prepara i dati per la predizione (senza target).
        
        Args:
            df: DataFrame con ultimi N tick
            scaler: Scaler già addestrato (se None, usa self.scaler)
            
        Returns:
            Array numpy pronto per la predizione
        """
        # Usa scaler fornito o quello interno
        current_scaler = scaler if scaler is not None else self.scaler
        
        # Seleziona solo le colonne features
        if self.feature_columns is None:
            # Se non settato, usa tutte le colonne tranne timestamp
            self.feature_columns = [col for col in df.columns if col != 'timestamp']
            self.feature_names = self.feature_columns.copy()
        
        # Normalizza i dati
        df_normalized = df.copy()
        df_normalized[self.feature_columns] = current_scaler.transform(
            df_normalized[self.feature_columns]
        )
        
        # Estrai sequenza
        sequence_data = df_normalized[self.feature_columns].values
        
        # Reshape per il modello (1 sequenza, sequence_length, n_features)
        if len(sequence_data) == self.sequence_length:
            return sequence_data.reshape(1, self.sequence_length, -1)
        else:
            raise ValueError(
                f"I dati devono avere esattamente {self.sequence_length} righe. "
                f"Ricevuti: {len(sequence_data)} righe"
            )


# Funzione di utilità per uso rapido
def load_and_prepare_data(filepath: str, sequence_length: int = 10, 
                         test_size: float = 0.2, lookahead: int = 10) -> Tuple:  # <-- Aggiungi lookahead
    """
    Funzione di utilità per caricare e preparare i dati in un unico passaggio.
    
    Args:
        filepath: Percorso del file CSV
        sequence_length: Lunghezza sequenze LSTM
        test_size: Percentuale test set
        lookahead: Tick futuri da considerare per target
    """
    loader = FinancialDataLoader(
        sequence_length=sequence_length,
        test_size=test_size,
        lookahead=lookahead  # <-- Passa il nuovo parametro
    )
    
    results = loader.process_pipeline(
        filepath=filepath,
        target_column='price',
        exclude_columns=['timestamp']
    )
    
    return results + (loader,)


if __name__ == "__main__":
    # Esempio di utilizzo
    try:
        # Crea dati di esempio se il file non esiste
        import os
        
        if not os.path.exists("sample_data.csv"):
            print("Creazione dati di esempio...")
            dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
            data = {
                'timestamp': dates,
                'price': np.random.randn(1000).cumsum() + 100,
                'indicator1': np.random.randn(1000),
                'indicator2': np.random.randn(1000),
                'volume': np.random.randint(100, 10000, 1000)
            }
            df_sample = pd.DataFrame(data)
            df_sample.to_csv("sample_data.csv", index=False)
            print("File di esempio creato: sample_data.csv")
        
        # Test del loader
        print("\nTest del FinancialDataLoader:")
        print("=" * 50)
        
        X_train, X_test, y_train, y_test, df_original, df_normalized, loader = load_and_prepare_data(
            "sample_data.csv",
            sequence_length=10,
            test_size=0.2
        )
        
        print(f"\nShape X_train: {X_train.shape}")
        print(f"Shape y_train: {y_train.shape}")
        print(f"Shape X_test: {X_test.shape}")
        print(f"Shape y_test: {y_test.shape}")
        print(f"\nNumero di features: {len(loader.feature_names)}")
        print(f"Features: {loader.feature_names}")
        
    except Exception as e:
        print(f"Errore durante il test: {str(e)}")