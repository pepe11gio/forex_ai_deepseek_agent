#!/usr/bin/env python3
"""
Script di avvio per il sistema di trading AI
Versione COMPLETA con tutte le funzionalità essenziali
"""

import os
import sys
import logging
from datetime import datetime

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Directory progetto
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_file_dir

print(f"📁 Directory progetto: {project_root}")

def setup_environment():
    """Configura le directory necessarie."""
    directories = {
        'models': 'Modelli addestrati',
        'data': 'Dati CSV',
        'analysis': 'Analisi e report', 
        'logs': 'Log sistema',
        'orders': 'Ordini generati'
    }
    
    print("\n📁 CREAZIONE STRUTTURA DIRECTORY:")
    print("=" * 50)
    
    paths = {}
    for dir_name, description in directories.items():
        dir_path = os.path.join(project_root, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        paths[dir_name] = dir_path
        print(f"  📂 {dir_name:15} → {description}")
    
    print("=" * 50)
    return paths

def main():
    """Avvia il sistema completo."""
    print("=" * 60)
    print("🤖 SISTEMA DI TRADING AI")
    print("=" * 60)
    
    # Configura ambiente
    paths = setup_environment()
    
    # Aggiungi src/ al path di Python
    src_dir = os.path.join(project_root, 'src')
    sys.path.insert(0, src_dir)
    
    try:
        from main import TradingAIOrchestrator
        
        print("\n" + "=" * 60)
        print("🚀 AVVIO SISTEMA")
        print("=" * 60)
        
        # Crea orchestratore
        orchestrator = TradingAIOrchestrator()
        
        print("\n📋 MODALITÀ DISPONIBILI:")
        print("=" * 40)
        print("1. Addestramento modello (TUTTI i file in data/)")
        print("2. Predizione con ordine TP/SL")
        print("3. Analisi modello")
        print("4. Chat interattiva con AI")
        print("5. Predizione su file specifico con TP/SL")
        print("6. Pipeline completo (training + predizione + analisi)")
        print("7. Training evoluto con self-learning") 
        print("0. Esci")
        print("=" * 40)
        
        while True:
            try:
                choice = input("\nScelta: ").strip()
                
                if choice == "0":
                    print("Arrivederci!")
                    break
                
                elif choice == "1":
                    # ADDESTRAMENTO SU TUTTI I FILE
                    print("\n" + "=" * 60)
                    print("🧠 ADDESTRAMENTO MODELLO")
                    print("=" * 60)
                    
                    data_dir = os.path.join(project_root, 'data')
                    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                    
                    if not csv_files:
                        print(f"❌ Nessun file CSV trovato in {data_dir}")
                        continue
                    
                    print(f"✅ Trovati {len(csv_files)} file CSV")
                    
                    result = orchestrator.run_training_pipeline()
                    
                    if result["success"]:
                        print(f"\n✅ MODELLO ADDESTRATO!")
                        print(f"   Modello salvato in: {paths['models']}")
                        print(f"   Scaler salvato: ✅ .pkl creato")
                        
                        # Verifica file creati
                        import glob
                        model_files = glob.glob(os.path.join(paths['models'], "*.h5"))
                        scaler_files = glob.glob(os.path.join(paths['models'], "*.pkl"))
                        print(f"   File creati: {len(model_files)} modelli, {len(scaler_files)} scaler")
                    else:
                        print(f"❌ Errore: {result.get('error', 'Unknown')}")
                
                elif choice == "2":
                    # PREDIZIONE CON TP/SL
                    print("\n" + "=" * 60)
                    print("🎯 PREDIZIONE CON TP/SL")
                    print("=" * 60)
                    
                    if not orchestrator.system_state["predictor_ready"]:
                        print("Configuro predictor...")
                        orchestrator.setup_predictor()
                    
                    print("Effettuo predizione con ultimi dati...")
                    result = orchestrator.predict(generate_order=True)
                    
                    if result["success"]:
                        pred = result["prediction"]
                        print(f"\n✅ PREDIZIONE:")
                        print(f"   Valore: {pred['prediction']:.6f}")
                        print(f"   Segnale: {pred['trading_signal']}")
                        
                        if "order" in pred and pred["order"].get("success"):
                            order = pred["order"]
                            print(f"\n💰 ORDINE GENERATO:")
                            print(f"   ID: {order.get('order_id')}")
                            print(f"   Operazione: {order.get('operation', 'N/A')}")
                            print(f"   TP: {order.get('take_profit', 'N/A')}")
                            print(f"   SL: {order.get('stop_loss', 'N/A')}")
                            
                            # Salva ordine
                            save = input("\nSalvare ordine? (s/n): ").lower()
                            if save == 's':
                                import json
                                order_file = os.path.join(paths['orders'], f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                                with open(order_file, 'w') as f:
                                    json.dump(order, f, indent=2)
                                print(f"✅ Ordine salvato: {order_file}")
                    else:
                        print(f"❌ Errore: {result.get('error')}")
                
                elif choice == "3":
                    # ANALISI MODELLO
                    print("\n" + "=" * 60)
                    print("📊 ANALISI MODELLO")
                    print("=" * 60)
                    
                    result = orchestrator.analyze_model()
                    
                    if result["success"]:
                        print(f"\n✅ ANALISI COMPLETATA")
                        if result.get("report_path"):
                            print(f"   Report: {result['report_path']}")
                    else:
                        print(f"❌ Errore: {result.get('error')}")
                
                elif choice == "4":
                    # CHAT INTERATTIVA
                    print("\n" + "=" * 60)
                    print("💬 CHAT INTERATTIVA CON AI")
                    print("=" * 60)
                    
                    api_key = os.getenv("DEEPSEEK_API_KEY")
                    if not api_key:
                        print("⚠️  DEEPSEEK_API_KEY non trovata")
                        print("   Il chatbot userà modalità simulazione")
                    
                    orchestrator.setup_chatbot(api_key=api_key)
                    orchestrator.interactive_chat()
                
                elif choice == "5":
                    # PREDIZIONE SU FILE SPECIFICO
                    print("\n" + "=" * 60)
                    print("🎯 PREDIZIONE SU FILE SPECIFICO")
                    print("=" * 60)
                    
                    # Chiedi file
                    test_file = input("\nPercorso file CSV (premi Invio per test.csv): ").strip()
                    if not test_file:
                        test_file = "test.csv"
                    
                    if not os.path.exists(test_file):
                        print(f"❌ File non trovato: {test_file}")
                        print("   Il file deve essere nella directory corrente")
                        continue
                    
                    print(f"File: {test_file}")
                    print("\nEffettuo predizione con TP/SL...")
                    
                    result = orchestrator.predict_from_file(test_file, generate_order=True)
                    
                    if result["success"]:
                        pred = result["prediction"]
                        print(f"\n✅ PREDIZIONE RIUSCITA!")
                        print(f"   File: {os.path.basename(result['test_file'])}")
                        print(f"   Valore: {pred['prediction']:.6f}")
                        print(f"   Segnale: {pred['trading_signal']}")
                        
                        if "order" in pred and pred["order"].get("success"):
                            order = pred["order"]
                            print(f"\n💰 ORDINE GENERATO:")
                            print(f"   Operazione: {order.get('operation', 'N/A')}")
                            
                            # Salva ordine
                            save = input("\nSalvare ordine? (s/n): ").lower()
                            if save == 's':
                                import json
                                order_file = os.path.join(paths['orders'], f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                                with open(order_file, 'w') as f:
                                    json.dump(order, f, indent=2)
                                print(f"✅ Ordine salvato: {order_file}")
                    else:
                        print(f"❌ Errore: {result.get('error')}")
                
                elif choice == "6":
                    # PIPELINE COMPLETO
                    print("\n" + "=" * 60)
                    print("⚙️  PIPELINE COMPLETO")
                    print("=" * 60)
                    
                    print("\nAvvio pipeline completo...")
                    result = orchestrator.run_full_pipeline()
                    
                    if result.get("success"):
                        print(f"\n✅ PIPELINE COMPLETATO!")
                        print(f"   Modello: {paths['models']}")
                        print(f"   Analisi: {paths['analysis']}")
                        print(f"   Logs: {paths['logs']}")
                    else:
                        print(f"❌ Errore: {result.get('error')}")
                elif choice == "7":
                    # TRAINING EVOLUTO CON SELF-LEARNING
                    print("\n" + "=" * 60)
                    print("🧠 TRAINING EVOLUTO CON SELF-LEARNING")
                    print("=" * 60)
                    
                    result = orchestrator.run_self_learning_training()
                    
                    if result["success"]:
                        print(f"\n✅ SELF-LEARNING COMPLETATO!")
                        
                        if 'improvement' in result:
                            print(f"   Miglioramento win rate: {result['improvement']:+.1%}p")
                            print(f"   Pattern errori trovati: {result.get('error_patterns_found', 0)}")
                        
                        print(f"   Modello: {result.get('model_name', 'N/A')}")
                    else:
                        print(f"❌ Errore: {result.get('error')}")
                else:
                    print("Scelta non valida. Riprova.")
                
            
            except KeyboardInterrupt:
                print("\n\nOperazione interrotta")
                break
            except Exception as e:
                print(f"\n❌ Errore: {str(e)}")
                import traceback
                traceback.print_exc()
    
    except ImportError as e:
        print(f"\n❌ ERRORE IMPORT MODULI:")
        print(f"   {str(e)}")
        print(f"\n📦 Verifica che tutti i file siano in: {src_dir}")
    
    except Exception as e:
        print(f"\n❌ ERRORE AVVIO: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()