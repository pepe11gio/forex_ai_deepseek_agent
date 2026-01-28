# src/modeling/tf_model.py
from __future__ import annotations

from typing import List, Tuple

import tensorflow as tf

from src.config import ProjectConfig


def build_multiscale_model(
    cfg: ProjectConfig,
    n_features: int,
) -> tf.keras.Model:
    """
    Modello Keras multi-input:
      inputs: [X_w1, X_w2, X_w3] con shape (None, window, n_features)
      output: (None, 2) sigmoid => [p_win_long, p_win_short]

    Nota: le finestre sono cfg.windows.windows (es. 48,128,256).
    """
    mc = cfg.model
    windows = list(cfg.windows.windows)

    inputs: List[tf.keras.Input] = []
    branch_outputs: List[tf.Tensor] = []

    for w in windows:
        inp = tf.keras.Input(shape=(w, n_features), name=f"inp_w{w}")
        x = inp

        # Conv1D per pattern locali
        x = tf.keras.layers.Conv1D(
            filters=mc.conv_filters,
            kernel_size=mc.conv_kernel_size,
            padding="same",
            activation="relu",
            name=f"conv_w{w}",
        )(x)

        x = tf.keras.layers.Dropout(mc.conv_dropout, name=f"conv_dropout_w{w}")(x)

        # GRU per contesto
        x = tf.keras.layers.GRU(
            mc.gru_units,
            dropout=mc.gru_dropout,
            recurrent_dropout=0.0,
            return_sequences=False,
            name=f"gru_w{w}",
        )(x)

        # Proiezione branch
        x = tf.keras.layers.Dense(
            mc.dense_units,
            activation="relu",
            name=f"branch_dense_w{w}",
        )(x)

        inputs.append(inp)
        branch_outputs.append(x)

    # Unisci le branch
    if len(branch_outputs) == 1:
        x = branch_outputs[0]
    else:
        x = tf.keras.layers.Concatenate(name="concat_branches")(branch_outputs)

    x = tf.keras.layers.Dropout(mc.head_dropout, name="head_dropout")(x)

    x = tf.keras.layers.Dense(
        mc.dense_units,
        activation="relu",
        name="head_dense",
    )(x)

    x = tf.keras.layers.Dropout(mc.head_dropout, name="head_dropout2")(x)

    # Output 2 sigmoid: p_win_long, p_win_short
    out = tf.keras.layers.Dense(
        mc.output_dim,
        activation="sigmoid",
        name="p_win",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=out, name="multiscale_cnn_gru")

    return model


def compile_model(model: tf.keras.Model, cfg: ProjectConfig) -> tf.keras.Model:
    """
    Compila il modello con BCE (default). Focal loss opzionale (non ancora inclusa).
    """
    tc = cfg.train

    if tc.loss_type.lower() != "bce":
        raise ValueError("loss_type supportato in questa versione: 'bce' (focal lo aggiungiamo se serve).")

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    opt = tf.keras.optimizers.Adam(learning_rate=tc.learning_rate)

    # metriche: AUC per ciascun output può essere utile, ma Keras lo calcola sull'output intero.
    # Usiamo AUC generica + binary_accuracy come diagnostica.
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
