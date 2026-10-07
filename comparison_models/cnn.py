"""Local comparison architecture; see README and source_manifest.json."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build_classic_cnn(input_shape=(256, 1), num_classes=7):
    """Standard 1D CNN: Conv-BN-ReLU-Pool x3 + GAP + Dense."""
    inp = layers.Input(shape=input_shape)
    x = inp
    for i, (f, k, p) in enumerate([(64, 8, 2), (64, 5, 2), (64, 3, 2)]):
        x = layers.Conv1D(f, k, padding='same', name=f'conv{i+1}')(x)
        x = layers.BatchNormalization(name=f'bn{i+1}')(x)
        x = layers.Activation('relu', name=f'relu{i+1}')(x)
        x = layers.MaxPooling1D(p, name=f'pool{i+1}')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inp, out, name='ClassicCNN')
