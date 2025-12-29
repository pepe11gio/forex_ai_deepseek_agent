# verifica_predictor.ps1
Write-Host "🔍 Verifica del predictor e del modello" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Controlla se model_trainer.py è corretto
Write-Host "`n1. Controllo model_trainer.py..." -ForegroundColor Yellow

$pythonCode = @'
import re
with open("src/model_trainer.py", "r", encoding="utf-8") as f:
    content = f.read()

# Cerca metriche come stringhe (ERRATO)
pattern_stringhe = r"metrics\s*=\s*\[.*?['\"]mae['\"].*?['\"]mse['\"].*?\]"
match_stringhe = re.search(pattern_stringhe, content, re.IGNORECASE)

# Cerca metriche come oggetti (CORRETTO)
oggetto_mae = "MeanAbsoluteError()" in content
oggetto_mse = "MeanSquaredError()" in content

print("=== RISULTATI ===")
if match_stringhe:
    print("❌ TROVATO PROBLEMA:")
    print(f"   Linea con stringhe: {match_stringhe.group()[:100]}...")
    print("   Devi correggere questa linea!")
else:
    print("✅ Nessuna metrica stringa trovata")

if oggetto_mae and oggetto_mse:
    print("✅ Metriche a oggetti trovate (MeanAbsoluteError(), MeanSquaredError())")
else:
    print("⚠️  Metriche a oggetti NON trovate")

# Controlla anche l'import
if "from tensorflow.keras import metrics" in content:
    print("✅ Import delle metriche presente")
else:
    print("⚠️  Import delle metriche mancante")
'@

python -c $pythonCode

# 2. Lista modelli disponibili
Write-Host "`n2. Modelli disponibili:" -ForegroundColor Yellow

$pythonCode2 = @'
import os, glob, json

print("Cartella models/:")
models = glob.glob("models/*.h5")
if not models:
    print("   ❌ Nessun modello .h5 trovato")
else:
    print(f"   ✅ Trovati {len(models)} modelli:")
    for model in sorted(models, key=os.path.getmtime, reverse=True)[:5]:
        name = os.path.basename(model).replace(".h5", "")
        scaler = f"models/{name}_scaler.pkl"
        metadata = f"models/{name}_metadata.json"
        
        print(f"   • {name}")
        print(f"     - Modello: {os.path.exists(model)}")
        print(f"     - Scaler: {os.path.exists(scaler)}")
        print(f"     - Metadata: {os.path.exists(metadata)}")
        
        if os.path.exists(metadata):
            try:
                with open(metadata, "r") as f:
                    meta = json.load(f)
                print(f"     - Tipo: {meta.get('model_type', 'N/A')}")
                print(f"     - Data: {meta.get('training_date', 'N/A')}")
            except:
                pass
        print()
'@

python -c $pythonCode2

# 3. Test diretto del predictor
Write-Host "`n3. Test diretto del predictor:" -ForegroundColor Yellow

$pythonCode3 = @'
print("Test caricamento modello più recente...")
import glob, os, tensorflow as tf
from tensorflow.keras import metrics

# Patch per sicurezza
tf.keras.utils.get_custom_objects().update({
    "MeanSquaredError": metrics.MeanSquaredError,
    "MeanAbsoluteError": metrics.MeanAbsoluteError,
    "mae": metrics.MeanAbsoluteError,
    "mse": metrics.MeanSquaredError
})

models = glob.glob("models/*.h5")
if models:
    latest = max(models, key=os.path.getmtime)
    print(f"Modello da testare: {os.path.basename(latest)}")
    
    try:
        # Prova 1: Caricamento normale
        model = tf.keras.models.load_model(latest)
        print("✅ SUCCESSO - Modello caricato normalmente!")
        print(f"   Input shape: {model.input_shape}")
        
        # Prova 2: Predizione con dati dummy
        import numpy as np
        dummy_input = np.random.randn(1, model.input_shape[1], model.input_shape[2])
        prediction = model.predict(dummy_input, verbose=0)
        print(f"   Predizione dummy: {prediction[0, 0]:.6f}")
        print("✅ Modello FUNZIONANTE!")
        
    except Exception as e:
        print(f"❌ ERRORE nel caricamento: {str(e)[:200]}")
        
        # Prova alternativa
        try:
            print("Tento caricamento senza compilazione...")
            model = tf.keras.models.load_model(latest, compile=False)
            print("✅ Caricamento senza compilazione riuscito")
            print("   Ora ricompilo manualmente...")
            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
            )
            print("✅ Modello ricompilato con successo!")
        except Exception as e2:
            print(f"❌ Anche il fallback ha fallito: {str(e2)[:200]}")
else:
    print("❌ Nessun modello trovato per il test")
'@

python -c $pythonCode3

Write-Host "`n✅ Verifica completata!" -ForegroundColor Green