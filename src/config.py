# Configurazione percorsi sistema
import os

# Ottieni la directory radice del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "root": BASE_DIR,
    "src": os.path.join(BASE_DIR, "src"),
    "models": os.path.join(BASE_DIR, "models"),
    "data": os.path.join(BASE_DIR, "data"),
    "analysis": os.path.join(BASE_DIR, "analysis"),
    "logs": os.path.join(BASE_DIR, "logs"),
    "orders": os.path.join(BASE_DIR, "orders")
}

# Configurazione data loader
DATA_CONFIG = {
    "sequence_length": 20,
    "test_size": 0.2,
    "target_column": "price",
    "exclude_columns": ["timestamp"]
}

# Configurazione modello
MODEL_CONFIG = {
    "type": "bidirectional",
    "epochs": 100,
    "batch_size": 32,
    "model_name_template": "trading_model_{timestamp}"
}

# Configurazione predizione
PREDICTION_CONFIG = {
    "confidence_level": 0.95,
    "cache_predictions": True
}

# Configurazione chat
CHAT_CONFIG = {
    "api_key_env_var": "DEEPSEEK_API_KEY",
    "model": "deepseek-chat",
    "max_tokens": 2000,
    "temperature": 0.7
}

# Funzione per creare directory
def create_directories():
    """Crea tutte le directory necessarie."""
    for dir_path in PATHS.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory verificata: {dir_path}")
    
    # Directory extra
    checkpoints_dir = os.path.join(PATHS["models"], "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    return PATHS

# Configurazione globale
CONFIG = {
    "paths": PATHS,
    "data": DATA_CONFIG,
    "model": MODEL_CONFIG,
    "prediction": PREDICTION_CONFIG,
    "chat": CHAT_CONFIG
}