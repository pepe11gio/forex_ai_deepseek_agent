# Configurazione percorsi sistema
import os

# Ottieni la directory radice del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "root": r"C:/Users/Giovanni/Documents/Python/forex_ai_deepseek_agent",
    "src": os.path.join(BASE_DIR, "src"),
    "models": os.path.join(BASE_DIR, "models"),
    "data": os.path.join(BASE_DIR, "data"),
    "analysis": os.path.join(BASE_DIR, "analysis"),
    "logs": os.path.join(BASE_DIR, "logs")
}

# Configurazione data loader
DATA_CONFIG = {
    "sequence_length": 20,
    "test_size": 0.2,
    "target_column": "price",
    "exclude_columns": ["timestamp"]
}

# Funzione per creare directory
def create_directories():
    """Crea tutte le directory necessarie se non esistono."""
    for dir_path in PATHS.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory verificata: {dir_path}")
    
    # Directory extra
    checkpoints_dir = os.path.join(PATHS["models"], "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Directory verificata: {checkpoints_dir}")
    
    return PATHS
