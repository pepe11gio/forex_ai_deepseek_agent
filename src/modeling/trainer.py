# src/modeling/trainer.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import tensorflow as tf

from src.config import ProjectConfig
from src.features.normalizer import FeatureScaler, scaler_default_path


@dataclass
class TrainArtifacts:
    run_name: str
    model_export_dir: str
    scaler_path: str
    feature_columns_path: str
    history_path: str


def _export_paths(cfg: ProjectConfig, run_name: str) -> Dict[str, str]:
    export_dir = os.path.join(cfg.model_dir, "exports", run_name)
    return {
        "export_dir": export_dir,
        "scaler_path": os.path.join(export_dir, "feature_scaler.json"),
        "feature_columns_path": os.path.join(export_dir, "feature_columns.json"),
        "history_path": os.path.join(export_dir, "train_history.json"),
    }


def save_feature_columns(feature_columns: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_columns}, f, indent=2)


def load_feature_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload["feature_columns"])


def train_model(
    cfg: ProjectConfig,
    model: tf.keras.Model,
    scaler: FeatureScaler,
    feature_columns: List[str],
    X_train_list: List[np.ndarray],
    y_train: np.ndarray,
    X_val_list: List[np.ndarray],
    y_val: np.ndarray,
    run_name: str = "default",
    class_weight: Optional[Dict[int, float]] = None,
) -> TrainArtifacts:
    """
    Allena e salva artefatti:
    - modello (SavedModel / Keras)
    - scaler (json)
    - feature_columns (json)
    - train history (json)
    """
    tc = cfg.train
    paths = _export_paths(cfg, run_name)

    os.makedirs(paths["export_dir"], exist_ok=True)

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=tc.early_stopping_patience,
            min_delta=tc.early_stopping_min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(paths["export_dir"], "checkpoint.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    fit_kwargs: Dict[str, Any] = dict(
        x=X_train_list,
        y=y_train,
        validation_data=(X_val_list, y_val),
        epochs=tc.epochs,
        batch_size=tc.batch_size,
        callbacks=callbacks,
        verbose=1 if cfg.verbose else 0,
        shuffle=False,  # time-series: non mischiare
    )

    # Nota: class_weight in Keras si applica a single-output. Con output 2 è ambiguo.
    # Per ora lo disabilitiamo (lo possiamo fare in modo custom con sample_weight per target).
    if class_weight is not None:
        raise ValueError(
            "class_weight non supportato con output multi-label in questa versione. "
            "Se serve, implementiamo sample_weight custom."
        )

    history = model.fit(**fit_kwargs)

    # Export final model
    # Salviamo in formato Keras (più semplice e stabile) + SavedModel.
    model_keras_path = os.path.join(paths["export_dir"], "model.keras")
    model.save(model_keras_path)

    saved_model_dir = os.path.join(paths["export_dir"], "saved_model")
    model.export(saved_model_dir)  # TF 2.13+ ; se hai versioni più vecchie, sostituire con tf.saved_model.save

    # Save scaler + feature columns
    scaler.save(paths["scaler_path"])
    save_feature_columns(feature_columns, paths["feature_columns_path"])

    # Save training history
    hist_payload = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(paths["history_path"], "w", encoding="utf-8") as f:
        json.dump(hist_payload, f, indent=2)

    return TrainArtifacts(
        run_name=run_name,
        model_export_dir=paths["export_dir"],
        scaler_path=paths["scaler_path"],
        feature_columns_path=paths["feature_columns_path"],
        history_path=paths["history_path"],
    )


def load_trained_model(cfg: ProjectConfig, run_name: str = "default") -> tf.keras.Model:
    export_dir = os.path.join(cfg.model_dir, "exports", run_name)
    model_path = os.path.join(export_dir, "model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    return tf.keras.models.load_model(model_path)
