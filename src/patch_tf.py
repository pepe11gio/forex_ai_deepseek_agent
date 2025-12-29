import tensorflow as tf
from tensorflow.keras import metrics

def apply_tensorflow_patch():
    print("🔧 Patch TensorFlow applicata")
    
    # Registra metriche
    tf.keras.utils.get_custom_objects().update({
        "MeanSquaredError": metrics.MeanSquaredError,
        "MeanAbsoluteError": metrics.MeanAbsoluteError,
        "mse": metrics.MeanSquaredError,
        "mae": metrics.MeanAbsoluteError
    })
    
    # Patch load_model
    original_load = tf.keras.models.load_model
    def patched_load(filepath, **kwargs):
        try:
            return original_load(filepath, **kwargs)
        except Exception as e:
            if "not a KerasSaveable subclass" in str(e):
                print(f"⚠️  Patch per: {filepath}")
                model = original_load(filepath, compile=False)
                model.compile(
                    optimizer="adam",
                    loss="mse",
                    metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
                )
                return model
            raise
    
    tf.keras.models.load_model = patched_load
    return True

apply_tensorflow_patch()
