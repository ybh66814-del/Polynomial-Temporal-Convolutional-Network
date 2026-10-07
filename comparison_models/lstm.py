"""Local comparison architecture; see README and source_manifest.json."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build_lstm_wang(input_shape=(256, 1), num_classes=7):
    """Stacked LSTM (Wang et al., 2022, Energy).

    2-layer LSTM with BN after first layer.
    """
    inp = layers.Input(shape=input_shape)

    x = layers.LSTM(128, return_sequences=True, name='lstm1')(inp)
    x = layers.BatchNormalization(name='lstm1_bn')(x)

    x = layers.LSTM(64, return_sequences=False, name='lstm2')(x)

    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inp, out, name='LSTM_Wang')
