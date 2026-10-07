"""Local comparison architecture; see README and source_manifest.json."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def _tcn_residual_block(x, filters, kernel_size, dilation_rate, block_id):
    """TCN residual block with dilated causal conv."""
    skip = x
    # Dilated causal conv 1
    c = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate,
                      name=f'tcn_b{block_id}_d{dilation_rate}_1')(x)
    c = layers.BatchNormalization(name=f'tcn_b{block_id}_d{dilation_rate}_1_bn')(c)
    c = layers.Activation('relu', name=f'tcn_b{block_id}_d{dilation_rate}_1_relu')(c)
    c = layers.Dropout(0.2, name=f'tcn_b{block_id}_d{dilation_rate}_1_do')(c)
    # Dilated causal conv 2
    c = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate,
                      name=f'tcn_b{block_id}_d{dilation_rate}_2')(c)
    c = layers.BatchNormalization(name=f'tcn_b{block_id}_d{dilation_rate}_2_bn')(c)
    c = layers.Activation('relu', name=f'tcn_b{block_id}_d{dilation_rate}_2_relu')(c)
    c = layers.Dropout(0.2, name=f'tcn_b{block_id}_d{dilation_rate}_2_do')(c)
    # Residual connection with 1x1 conv if needed
    if skip.shape[-1] != filters:
        skip = layers.Conv1D(filters, 1, padding='same',
                             name=f'tcn_b{block_id}_skip_conv')(skip)
    out = layers.Add(name=f'tcn_b{block_id}_res')([skip, c])
    return layers.Activation('relu', name=f'tcn_b{block_id}_out')(out)


def build_cnn_tcn(input_shape=(256, 1), num_classes=7):
    """CNN-TCN serial cascade (Liu et al., 2025).

    CNN frontend: wide first kernel (64, stride=8) for noise suppression
    TCN backend: 4 dilated residual blocks [1,2,4,8], kernel=3
    """
    inp = layers.Input(shape=input_shape)

    # ---- CNN frontend (spatial feature extraction) ----
    x = layers.Conv1D(16, kernel_size=64, strides=8, padding='same',
                      name='cnn_wide')(inp)
    x = layers.BatchNormalization(name='cnn_wide_bn')(x)
    x = layers.Activation('relu', name='cnn_wide_relu')(x)
    x = layers.MaxPooling1D(2, name='cnn_pool1')(x)

    x = layers.Conv1D(32, 3, padding='same', name='cnn_conv2')(x)
    x = layers.BatchNormalization(name='cnn_conv2_bn')(x)
    x = layers.Activation('relu', name='cnn_conv2_relu')(x)

    x = layers.Conv1D(64, 3, padding='same', name='cnn_conv3')(x)
    x = layers.BatchNormalization(name='cnn_conv3_bn')(x)
    x = layers.Activation('relu', name='cnn_conv3_relu')(x)

    x = layers.Conv1D(128, 3, padding='same', name='cnn_conv4')(x)
    x = layers.BatchNormalization(name='cnn_conv4_bn')(x)
    x = layers.Activation('relu', name='cnn_conv4_relu')(x)

    # ---- TCN backend (temporal dependency modeling) ----
    x = _tcn_residual_block(x, 64, 3, dilation_rate=1, block_id=1)
    x = _tcn_residual_block(x, 64, 3, dilation_rate=2, block_id=2)
    x = _tcn_residual_block(x, 64, 3, dilation_rate=4, block_id=3)
    x = _tcn_residual_block(x, 64, 3, dilation_rate=8, block_id=4)

    # ---- Classifier ----
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inp, out, name='CNN_TCN')
