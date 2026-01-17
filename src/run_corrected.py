#!/usr/bin/env python3
"""
Script di avvio corretto per il sistema di trading AI
Gestione directory corretta: al livello superiore, non dentro src/
"""

import os
import sys
import logging
from datetime import datetime

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TROVA LA DIRECTORY RADICE DEL PROGETTO (non src/)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)  # Salgo di un livello da src/

print(f"📁 Directory progetto: {project_root}")
print(f"📁 Directory script: {current_file_dir}")

def get_project_root():
    """Restituisce la directory radice del progetto."""
    # Se siamo in src/, saliamo di un livello
    if os.path.basename(current_file_dir) == 'src':
        return os.path.dirname(current_file_dir)
    return current_file_dir

def get_paths():
    """Restituisce tutti i percorsi corretti."""
    root = get_project_root()
    return {
        'root': root,
        'src': os.path.join(root, 'src'),
        'models': os.path.join(root, 'models'),
        'data': os.path.join(root, 'data'),
        'analysis': os.path.join(root, 'analysis'),
        'logs': os.path.join(root, 'logs'),
        'checkpoints': os.path.join(root, 'models', 'checkpoints')
    }

def check_dependencies():
    """Verifica che tutte le dipendenze siano installate."""
    required_packages = [
        'tensorflow',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'scipy',
        'joblib',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def create_sample_data():
    """Crea dati di esempio se non esistono."""
    paths = get_paths()
    data_dir = paths['data']
    os.makedirs(data_dir, exist_ok=True)
    
    sample_file = os.path.join(data_dir, 'sample_trading_data.csv')
    
    if not os.path.exists(sample_file):
        logger.info("Creazione dati di esempio...")
        
        try:
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Crea dati temporali
            start_date = datetime(2024, 1, 1)
            dates = [start_date + timedelta(minutes=i) for i in range(1000)]
            
            # Crea dati di trading sintetici
            np.random.seed(42)
            base_price = 100.0
            price_changes = np.random.randn(1000) * 0.5  # Volatilità 0.5%
            prices = base_price + np.cumsum(price_changes)
            
            # Crea indicatori tecnici sintetici
            rsi = 50 + np.random.randn(1000) * 10  # RSI intorno a 50
            macd = np.random.randn(1000) * 2       # MACD
            volume = np.random.randint(1000, 10000, 1000)  # Volume
            
            # Crea DataFrame
            data = {
                'timestamp': dates,
                'price': prices,
                'rsi': np.clip(rsi, 0, 100),  # RSI tra 0 e 100
                'macd': macd,
                'volume': volume,
                'sma_20': pd.Series(prices).rolling(20).mean().values,
                'ema_12': pd.Series(prices).ewm(span=12).mean().values,
                'bollinger_upper': pd.Series(prices).rolling(20).mean().values + 
                                   pd.Series(prices).rolling(20).std().values * 2,
                'bollinger_lower': pd.Series(prices).rolling(20).mean().values - 
                                   pd.Series(prices).rolling(20).std().values * 2
            }
            
            df = pd.DataFrame(data)
            df.to_csv(sample_file, index=False)
            
            logger.info(f"✅ Dati di esempio creati: {sample_file}")
            logger.info(f"   Righe: {len(df)}, Colonne: {len(df.columns)}")
            
        except Exception as e:
            logger.error(f"❌ Errore nella creazione dati di esempio: {e}")
            return False
    
    return True

def create_directory_structure():
    """Crea la struttura di directory necessaria al livello corretto."""
    paths = get_paths()
    
    directories = [
        ('models', paths['models']),
        ('data', paths['data']),
        ('analysis', paths['analysis']),
        ('logs', paths['logs']),
        ('models/checkpoints', paths['checkpoints'])
    ]
    
    print("\n📁 CREAZIONE STRUTTURA DIRECTORY:")
    print("=" * 40)
    
    for dir_name, dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        relative_path = os.path.relpath(dir_path, paths['root'])
        print(f"  📂 {dir_name:20} → {relative_path}")
    
    print("=" * 40)
    logger.info("Struttura directory verificata")
    return paths

def setup_environment():
    """Configura l'ambiente per l'esecuzione."""
    print("=" * 60)
    print("🎯 TRADING AI SYSTEM - SETUP")
    print("=" * 60)
    
    # Mostra struttura
    paths = get_paths()
    print(f"\n📍 Percorsi:")
    print(f"   Root:      {paths['root']}")
    print(f"   Scripts:   {paths['src']}")
    print(f"   Modelli:   {paths['models']}")
    print(f"   Dati:      {paths['data']}")
    print(f"   Analisi:   {paths['analysis']}")
    print(f"   Logs:      {paths['logs']}")
    
    # Verifica dipendenze
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"\n⚠️  DIPENDENZE MANCANTI:")
        for package in missing_packages:
            print(f"   - {package}")
        
        install_command = f"pip install {' '.join(missing_packages)}"
        print(f"\n📦 Installa con:")
        print(f"   {install_command}")
        
        choice = input("\nInstallare ora? (s/n): ").lower()
        if choice == 's':
            os.system(install_command)
        else:
            print("\n❌ Alcune dipendenze mancano. Il sistema potrebbe non funzionare correttamente.")
    
    # Crea struttura directory
    paths = create_directory_structure()
    
    # Crea dati di esempio
    if create_sample_data():
        print("\n✅ Ambiente configurato con successo!")
        return paths
    else:
        print("\n❌ Errore nella configurazione dell'ambiente")
        return None

def update_system_config(paths):
    """Aggiorna la configurazione del sistema per usare i percorsi corretti."""
    # Crea un file di configurazione temporaneo
    config_content = '''# Configurazione percorsi sistema
import os

# Ottieni la directory radice del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "root": r"''' + paths['root'].replace('\\', '/') + '''",
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
'''
    
    config_file = os.path.join(paths['src'], 'config.py')
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    logger.info(f"Configurazione salvata in: {config_file}")
    return config_file

def run_quick_test(paths):
    """Esegue un test rapido del sistema."""
    try:
        print("\n🧪 Test 1/4: Import moduli...")
        sys.path.insert(0, paths['src'])
        from data_loader import FinancialDataLoader
        from model_trainer import LSTMTradingModel
        from predictor import TradingPredictor
        print("✅ Import moduli OK")
        
        print("\n🧪 Test 2/4: Verifica directory...")
        for dir_name, dir_path in [
            ("models", paths['models']),
            ("data", paths['data']),
            ("analysis", paths['analysis']),
            ("logs", paths['logs'])
        ]:
            if os.path.exists(dir_path):
                print(f"  ✅ {dir_name}: {dir_path}")
            else:
                print(f"  ❌ {dir_name}: NON ESISTE")
                return False
        
        print("\n🧪 Test 3/4: Test dati di esempio...")
        data_file = os.path.join(paths['data'], 'sample_trading_data.csv')
        if os.path.exists(data_file):
            import pandas as pd
            df = pd.read_csv(data_file, nrows=5)
            print(f"  ✅ Dati caricati: {len(df)} righe, {len(df.columns)} colonne")
        else:
            print(f"  ❌ File dati non trovato: {data_file}")
        
        print("\n🧪 Test 4/4: Test creazione modello...")
        import tensorflow as tf
        import numpy as np
        
        # Crea modello semplice
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, input_shape=(20, 5)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        print("✅ Modello compilato correttamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Test fallito: {str(e)}")
        return False

def main():
    """Avvia il sistema corretto."""
    print("=" * 60)
    print("🤖 SISTEMA DI TRADING AI - VERSIONE CORRETTA")
    print("=" * 60)
    
    # Configura ambiente
    paths = setup_environment()
    if not paths:
        print("\nImpossibile configurare l'ambiente. Uscita...")
        return
    
    # Aggiorna configurazione
    update_system_config(paths)
    
    # Aggiungi src/ al path di Python
    sys.path.insert(0, paths['src'])
    
    try:
        # Importa l'orchestratore corretto
        from main import TradingAIOrchestrator
        
        print("\n" + "=" * 60)
        print("🚀 AVVIO SISTEMA TRADING AI")
        print("=" * 60)
        
        # Crea orchestratore con configurazione corretta
        config_data = {
            "paths": {
                "data_dir": paths['data'],
                "models_dir": paths['models'],
                "analysis_dir": paths['analysis'],
                "logs_dir": paths['logs']
            }
        }
        
        # Salva configurazione temporanea
        import json
        temp_config = os.path.join(paths['root'], 'temp_config.json')
        with open(temp_config, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        orchestrator = TradingAIOrchestrator(config_file=temp_config)
        
        # Mostra opzioni
        print("\n📋 MODALITÀ DISPONIBILI:")
        print("1. Chat interattiva completa")
        print("2. Pipeline completo automatico")
        print("3. Solo addestramento modello")
        print("4. Solo predizioni")
        print("5. Solo analisi")
        print("6. Test rapido")
        print("7. Predizione con ordine TP/SL")
        print("8. Predizione su file specifico")
        print("9. Predizione su file specifico con ordine TP/SL")
        print("0. Esci")
        
        while True:
            try:
                choice = input("\nScelta: ").strip()
                
                if choice == "0":
                    print("Arrivederci!")
                    break
                
                elif choice == "1":
                    # Modalità chat interattiva
                    print("\n" + "=" * 60)
                    print("💬 CHAT INTERATTIVA CON TRADING AI")
                    print("=" * 60)
                    
                    # Configura chatbot
                    api_key = os.getenv("DEEPSEEK_API_KEY")
                    if not api_key:
                        print("\n⚠️  DEEPSEEK_API_KEY non trovata nelle env var")
                        print("   Il chatbot userà la modalità simulazione")
                        print("   Per API reale, esporta la tua key:")
                        print("   export DEEPSEEK_API_KEY='la_tua_api_key'")
                    
                    orchestrator.setup_chatbot(api_key=api_key)
                    
                    # Avvia chat interattiva
                    orchestrator.interactive_chat()
                    break
                
                elif choice == "2":
                    # Pipeline completo
                    print("\n" + "=" * 60)
                    print("⚙️  PIPELINE COMPLETO AUTOMATICO")
                    print("=" * 60)
                    
                    # Verifica se ci sono dati
                    csv_files = [f for f in os.listdir(paths['data']) if f.endswith('.csv')]
                    
                    if not csv_files:
                        print("❌ Nessun file CSV trovato in data/")
                        print("   Verifica che i dati siano nella directory corretta")
                        continue
                    
                    print(f"\n✅ Trovati {len(csv_files)} file CSV in {paths['data']}:")
                    for i, file in enumerate(csv_files[:5], 1):
                        print(f"  {i}. {file}")
                    
                    if len(csv_files) > 1:
                        file_idx = input(f"\nScegli file (1-{min(5, len(csv_files))}): ").strip()
                        try:
                            idx = int(file_idx) - 1
                            if 0 <= idx < len(csv_files):
                                data_file = os.path.join(paths['data'], csv_files[idx])
                            else:
                                data_file = os.path.join(paths['data'], csv_files[0])
                        except:
                            data_file = os.path.join(paths['data'], csv_files[0])
                    else:
                        data_file = os.path.join(paths['data'], csv_files[0])
                    
                    print(f"\n🎯 Avvio pipeline con file: {data_file}")
                    
                    result = orchestrator.run_full_pipeline(data_file=data_file)
                    
                    if result.get("success"):
                        print("\n✅ PIPELINE COMPLETATO CON SUCCESSO!")
                        
                        # Mostra dove sono stati salvati i file
                        print(f"\n📁 OUTPUT CREATI:")
                        print(f"   Modelli: {paths['models']}")
                        print(f"   Analisi: {paths['analysis']}")
                        print(f"   Logs:    {paths['logs']}")
                    else:
                        print(f"\n❌ Errore nel pipeline: {result.get('error', 'Unknown error')}")
                    
                    break
                
                elif choice == "3":
                    # Solo addestramento
                    print("\n" + "=" * 60)
                    print("🧠 SOLO ADDESTRAMENTO MODELLO")
                    print("=" * 60)
                    
                    # Carica dati
                    data_result = orchestrator.load_data()
                    if not data_result["success"]:
                        print(f"❌ Errore nel caricamento dati: {data_result.get('error')}")
                        continue
                    
                    print(f"✅ Dati caricati: {data_result['data_info']['original_samples']} campioni")
                    
                    # Addestra modello
                    train_result = orchestrator.train_model()
                    
                    if train_result["success"]:
                        print(f"\n✅ Modello addestrato: {train_result['model_info']['model_name']}")
                        print(f"📊 Performance: MSE={train_result['performance'].get('final_test_mse', 'N/A'):.6f}")
                        print(f"💾 Salvato in: {paths['models']}")
                    else:
                        print(f"❌ Errore nell'addestramento: {train_result.get('error')}")
                    
                    break
                
                elif choice == "4":
                    # Solo predizioni
                    print("\n" + "=" * 60)
                    print("🔮 SOLO PREDIZIONI")
                    print("=" * 60)
                    
                    # Configura predictor
                    predictor_result = orchestrator.setup_predictor()
                    
                    if not predictor_result["success"]:
                        print("❌ Predictor non configurato. Addestra prima un modello o carica un modello esistente.")
                        print(f"   Modelli disponibili in: {paths['models']}")
                        continue
                    
                    print("✅ Predictor configurato")
                    
                    # Carica dati per predizione
                    data_result = orchestrator.load_data()
                    if data_result["success"]:
                        prediction = orchestrator.predict(generate_order=False)
                        
                        if prediction["success"]:
                            pred = prediction["prediction"]
                            print(f"\n🎯 PREDIZIONE:")
                            print(f"   Valore: {pred['prediction']:.4f}")
                            print(f"   Segnale: {pred['trading_signal']}")
                            print(f"   Confidenza: {pred['signal_confidence']:.2f}")
                            print(f"   Intervallo: [{pred['confidence_interval']['lower']:.4f}, "
                                  f"{pred['confidence_interval']['upper']:.4f}]")
                            
                            # Chiedi se analizzare con AI
                            analyze = input("\nAnalizzare con AI? (s/n): ").lower()
                            if analyze == 's':
                                # Configura chatbot
                                api_key = os.getenv("DEEPSEEK_API_KEY")
                                orchestrator.setup_chatbot(api_key=api_key)
                                
                                ai_result = orchestrator.chat_about_prediction(pred)
                                if ai_result["success"]:
                                    print("\n" + "=" * 40)
                                    print("🧠 ANALISI AI:")
                                    print("=" * 40)
                                    print(ai_result["ai_analysis"]["explanation"])
                        else:
                            print(f"❌ Errore nella predizione: {prediction.get('error')}")
                    else:
                        print(f"❌ Errore nel caricamento dati: {data_result.get('error')}")
                    
                    break
                
                elif choice == "5":
                    # Solo analisi
                    print("\n" + "=" * 60)
                    print("📊 SOLO ANALISI MODELLO")
                    print("=" * 60)
                    
                    # Verifica se esiste un modello
                    model_files = [f for f in os.listdir(paths['models']) if f.endswith('.h5')]
                    
                    if not model_files:
                        print(f"❌ Nessun modello trovato in {paths['models']}")
                        print("   Addestra prima un modello o copia un modello esistente.")
                        continue
                    
                    print(f"✅ Trovati {len(model_files)} modelli")
                    
                    # Carica dati
                    data_result = orchestrator.load_data()
                    if not data_result["success"]:
                        print(f"❌ Errore nel caricamento dati: {data_result.get('error')}")
                        continue
                    
                    # Configura predictor
                    orchestrator.setup_predictor()
                    
                    # Esegui analisi
                    analysis_result = orchestrator.analyze_model()
                    
                    if analysis_result["success"]:
                        print(f"\n✅ Analisi completata!")
                        if analysis_result.get("report_path"):
                            print(f"📄 Report generato: {analysis_result['report_path']}")
                    else:
                        print(f"❌ Errore nell'analisi: {analysis_result.get('error')}")
                    
                    break
                
                elif choice == "6":
                    # Test rapido
                    print("\n" + "=" * 60)
                    print("⚡ TEST RAPIDO DEL SISTEMA")
                    print("=" * 60)
                    
                    print("\nEsecuzione test rapido...")
                    
                    test_result = run_quick_test(paths)
                    
                    if test_result:
                        print("\n✅ TEST COMPLETATO CON SUCCESSO!")
                    else:
                        print("\n❌ TEST FALLITO")
                    
                    break
                
                elif choice == "7":
                    # Predizione con ordine TP/SL
                    print("\n" + "=" * 60)
                    print("🎯 PREDIZIONE CON ORDINE TP/SL")
                    print("=" * 60)
                    
                    # Configura predictor se non pronto
                    if not orchestrator.system_state["predictor_ready"]:
                        print("Predictor non configurato. Configuro...")
                        orchestrator.setup_predictor()
                    
                    # Carica dati
                    data_result = orchestrator.load_data()
                    if not data_result["success"]:
                        print(f"❌ Errore nel caricamento dati: {data_result.get('error')}")
                        continue
                    
                    print("Effettuo predizione e genero ordine con TP/SL...")
                    
                    try:
                        # Usa il metodo predict con generate_order=True
                        prediction_result = orchestrator.predict(generate_order=True)
                        
                        if prediction_result["success"]:
                            pred = prediction_result["prediction"]
                            
                            if "order" in pred:
                                order_result = pred["order"]
                                
                                if order_result.get("success", False):
                                    print("\n" + "=" * 60)
                                    print("✅ ORDINE CREATO CON SUCCESSO!")
                                    print("=" * 60)
                                    
                                    # Stampa summary
                                    if "execution_summary" in order_result:
                                        exec_summary = order_result["execution_summary"]
                                        print(f"\n📋 EXECUTION SUMMARY:")
                                        print(f"  Operazione: {exec_summary.get('action', 'N/A')}")
                                        print(f"  Entry Price: {exec_summary.get('entry', 'N/A')}")
                                        print(f"  Take Profit: {exec_summary.get('tp', 'N/A')}")
                                        print(f"  Stop Loss: {exec_summary.get('sl', 'N/A')}")
                                        print(f"  R/R Ratio: {exec_summary.get('rr_ratio', 'N/A')}:1")
                                        print(f"  Position Size: {exec_summary.get('position_size_lots', 'N/A')} lots")
                                    
                                    # Stampa display completo se disponibile
                                    if "display_text" in order_result:
                                        print("\n📊 ORDINE COMPLETO:")
                                        print(order_result["display_text"])
                                    
                                    # Chiedi se salvare
                                    save = input("\nSalvare ordine su file? (s/n): ").lower()
                                    if save == 's':
                                        import json
                                        from datetime import datetime
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        filename = f"order_{timestamp}.json"
                                        
                                        with open(filename, 'w') as f:
                                            json.dump(order_result, f, indent=2)
                                        
                                        print(f"✅ Ordine salvato in: {filename}")
                                else:
                                    print(f"❌ Errore nella creazione ordine: {order_result.get('error', 'Errore sconosciuto')}")
                                    if 'warnings' in order_result:
                                        print(f"Warnings: {order_result['warnings']}")
                            else:
                                print("❌ Ordine non generato nel risultato")
                                
                                # Mostra comunque la predizione base
                                print(f"\n📊 PREDIZIONE BASE:")
                                print(f"  Valore: {pred['prediction']:.4f}")
                                print(f"  Segnale: {pred['trading_signal']}")
                                print(f"  Confidenza: {pred['signal_confidence']:.2f}")
                        else:
                            print(f"❌ Errore nella predizione: {prediction_result.get('error', 'Errore sconosciuto')}")
                            
                    except Exception as e:
                        print(f"\n❌ Errore nell'esecuzione: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    break
                
                elif choice == "8" or choice == "9":
                    # Predizione su file specifico
                    print("\n" + "=" * 60)
                    print("🎯 PREDIZIONE SU FILE SPECIFICO")
                    print("=" * 60)
                    
                    # Configura predictor se non pronto
                    if not orchestrator.system_state["predictor_ready"]:
                        print("Predictor non configurato. Configuro...")
                        orchestrator.setup_predictor()
                    
                    # Chiedi il file
                    test_file = input("\nPercorso del file CSV di test (premi Invio per usare test.csv): ").strip()
                    if not test_file:
                        test_file = "test.csv"
                    
                    # Verifica che il file esista
                    if not os.path.exists(test_file):
                        print(f"❌ File non trovato: {test_file}")
                        print("   Il file deve essere nella directory corrente o specificare il percorso completo.")
                        continue
                    
                    print(f"File selezionato: {test_file}")
                    
                    # Determina se generare ordine
                    generate_order = (choice == "9")
                    
                    if generate_order:
                        print("\nEffettuo predizione e genero ordine con TP/SL...")
                    else:
                        print("\nEffettuo predizione...")
                    
                    try:
                        # Importa il metodo predict_from_file
                        from main import TradingAIOrchestrator as TAI
                        
                        # Usa reflection per chiamare il metodo
                        result = orchestrator.predict_from_file(test_file, generate_order=generate_order)
                        
                        if result["success"]:
                            pred = result["prediction"]
                            
                            print("\n" + "=" * 60)
                            print("✅ PREDIZIONE RIUSCITA!")
                            print("=" * 60)
                            print(f"File: {os.path.basename(result['test_file'])}")
                            print(f"Valore predetto: {pred['prediction']:.4f}")
                            print(f"Segnale: {pred['trading_signal']}")
                            print(f"Confidenza: {pred['signal_confidence']:.2f}")
                            
                            if 'confidence_interval' in pred:
                                ci = pred['confidence_interval']
                                print(f"Intervallo confidenza: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
                            
                            if generate_order and "order" in pred:
                                order_result = pred["order"]
                                
                                if order_result.get("success"):
                                    print("\n" + "=" * 60)
                                    print("✅ ORDINE GENERATO!")
                                    print("=" * 60)
                                    
                                    # Stampa summary
                                    if "execution_summary" in order_result:
                                        es = order_result["execution_summary"]
                                        print(f"\n📋 EXECUTION SUMMARY:")
                                        print(f"  Operazione: {es.get('action', 'N/A')}")
                                        print(f"  Entry Price: {es.get('entry', 'N/A')}")
                                        print(f"  Take Profit: {es.get('tp', 'N/A')}")
                                        print(f"  Stop Loss: {es.get('sl', 'N/A')}")
                                        print(f"  R/R Ratio: {es.get('rr_ratio', 'N/A')}:1")
                                        print(f"  Position Size: {es.get('position_size_lots', 'N/A')} lots")
                                    
                                    # Stampa display completo se disponibile
                                    if "display_text" in order_result:
                                        print("\n📊 ORDINE COMPLETO:")
                                        print(order_result["display_text"])
                                    
                                    # Chiedi se salvare
                                    save = input("\nSalvare ordine su file? (s/n): ").lower()
                                    if save == 's':
                                        import json
                                        from datetime import datetime
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        filename = f"order_{timestamp}.json"
                                        
                                        with open(filename, 'w') as f:
                                            json.dump(order_result, f, indent=2)
                                        
                                        print(f"✅ Ordine salvato in: {filename}")
                                else:
                                    print(f"\n⚠️  Ordine non generato: {order_result.get('error', 'Errore sconosciuto')}")
                        else:
                            print(f"\n❌ Errore: {result.get('error', 'Errore sconosciuto')}")
                        
                    except AttributeError:
                        print("\n⚠️  La funzionalità di predizione su file specifico non è disponibile.")
                        print("   Aggiorna il file main.py con il metodo predict_from_file().")
                    except Exception as e:
                        print(f"\n❌ Errore: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    break
                
                else:
                    print("Scelta non valida. Riprova.")
            
            except KeyboardInterrupt:
                print("\n\nOperazione interrotta dall'utente")
                break
            except Exception as e:
                print(f"\n❌ Errore: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Pulisci file temporaneo
        if os.path.exists(temp_config):
            os.remove(temp_config)
    
    except ImportError as e:
        logger.error(f"Errore nell'importazione dei moduli: {str(e)}")
        print(f"\n❌ IMPOSSIBILE IMPORTARE I MODULI:")
        print(f"   {str(e)}")
        print(f"\n📦 Verifica che i file siano in: {paths['src']}")
        print("   e che le dipendenze siano installate")
    
    except Exception as e:
        logger.error(f"Errore nell'avvio del sistema: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()