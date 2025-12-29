
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler

print("🧠 Training in corso...")

# Dati dummy
np.random.seed(42)
X = np.random.randn(200, 20, 9)
y = np.random.randn(200, 1)

# Modello
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(20, 9)),
    tf.keras.layers.Dense(1)
])

# COMPILAZIONE CORRETTA
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
)

# Training veloce
model.fit(X, y, epochs=2, batch_size=16, verbose=1)

# Salva
model_name = "trading_model_FIXED"
model_path = f"src/models/{model_name}.h5"
model.save(model_path)

# Scaler dummy
scaler = StandardScaler()
scaler.fit(np.random.randn(100, 9))
joblib.dump(scaler, f"src/models/{model_name}_scaler.pkl")

# Metadata
metadata = {
    "model_name": model_name,
    "sequence_length": 20,
    "n_features": 9,
    "fixed": True
}
with open(f"src/models/{model_name}_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Modello salvato: {model_path}")

# Test
loaded = tf.keras.models.load_model(model_path)
print(f"✅ Caricamento test: OK")
