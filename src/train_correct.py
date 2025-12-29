
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
import os

print("🧠 Training modello pulito...")

# Crea dati dummy per test rapido
np.random.seed(42)
n_samples = 200
sequence_length = 20
n_features = 9

X_train = np.random.randn(n_samples, sequence_length, n_features)
y_train = np.random.randn(n_samples, 1)
X_test = np.random.randn(50, sequence_length, n_features)
y_test = np.random.randn(50, 1)

# Costruisci modello semplice ma corretto
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(sequence_length, n_features), return_sequences=True),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

# COMPILAZIONE CORRETTA - OGGETTI, NON STRINGHE!
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
)

print(f"✅ Modello compilato correttamente")
print(f"   Metriche: MeanAbsoluteError, MeanSquaredError (OGGETTI)")

# Training breve
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=3,
    batch_size=16,
    verbose=1
)

# Salva modello
model_name = "trading_model_CORRETTO"
model_path = f"src/models/{model_name}.h5"
model.save(model_path)

# Salva scaler dummy
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
scaler.fit(np.random.randn(100, n_features))
scaler_path = f"src/models/{model_name}_scaler.pkl"
joblib.dump(scaler, scaler_path)

# Metadata
import json
metadata = {
    "model_name": model_name,
    "model_type": "standard",
    "sequence_length": sequence_length,
    "n_features": n_features,
    "training_date": "2024-12-28",
    "metrics": "MeanAbsoluteError, MeanSquaredError",
    "note": "Modello creato con codice CORRETTO"
}

metadata_path = f"src/models/{model_name}_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ NUOVO MODELLO CREATO CON SUCCESSO!")
print(f"   Modello: {model_path}")
print(f"   Scaler: {scaler_path}")
print(f"   Metadata: {metadata_path}")

# Verifica che si possa caricare
print("\n🔍 Verifica caricamento...")
try:
    loaded_model = tf.keras.models.load_model(model_path)
    print(f"✅ Caricamento RIUSCITO!")
    print(f"   Input shape: {loaded_model.input_shape}")
    
    # Test predizione
    test_input = np.random.randn(1, sequence_length, n_features)
    prediction = loaded_model.predict(test_input, verbose=0)
    print(f"   Predizione test: {prediction[0, 0]:.6f}")
    
except Exception as e:
    print(f"❌ Errore caricamento: {e}")
