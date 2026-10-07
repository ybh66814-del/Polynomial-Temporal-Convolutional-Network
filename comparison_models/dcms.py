"""Local comparison architecture; see README and source_manifest.json."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def _ms_block(x, block_id, kernel_sizes, filters):
    """Multi-Scale block: parallel branches with different kernels + residual."""
    branches = []
    for ks in kernel_sizes:
        b = layers.Conv1D(filters, ks, padding='same',
                          name=f'ms_b{block_id}_k{ks}')(x)
        b = layers.BatchNormalization(name=f'ms_b{block_id}_k{ks}_bn')(b)
        b = layers.Activation('relu', name=f'ms_b{block_id}_k{ks}_relu')(b)
        branches.append(b)

    fused = layers.Add(name=f'ms_b{block_id}_fuse')(branches)
    # Residual: project input if channel mismatch
    if x.shape[-1] != filters:
        x_proj = layers.Conv1D(filters, 1, padding='same',
                               name=f'ms_b{block_id}_proj')(x)
    else:
        x_proj = x
    out = layers.Add(name=f'ms_b{block_id}_res')([x_proj, fused])
    return layers.Activation('relu', name=f'ms_b{block_id}_out')(out)


def build_dcms1dcnn(input_shape=(256, 1), num_classes=7):
    """DCMS1DCNN (Li et al., 2025).

    Uses MS-blocks with residual connections and dual-channel structure.
    Kernels follow universal rule: k_i = 2*i+1 for i=0,1,2,... up to K_max.

    Channel A (fine): kernels [3, 5, 7, 9]
    Channel B (coarse): kernels [13, 15, 17, 19]
    """
    inp = layers.Input(shape=input_shape)

    # ---- Stem ----
    x = layers.Conv1D(16, 7, padding='same', name='stem_conv')(inp)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('relu', name='stem_relu')(x)

    # ---- Dual-channel multi-scale blocks ----
    # Channel A: fine-scale (small kernels)
    ch_a = _ms_block(x, 1, kernel_sizes=(3, 5, 7, 9), filters=32)
    ch_a = layers.MaxPooling1D(2, name='chA_pool')(ch_a)
    ch_a = _ms_block(ch_a, 2, kernel_sizes=(3, 5, 7), filters=64)

    # Channel B: coarse-scale (large kernels)
    ch_b = _ms_block(x, 3, kernel_sizes=(13, 15, 17, 19), filters=32)
    ch_b = layers.MaxPooling1D(2, name='chB_pool')(ch_b)
    ch_b = _ms_block(ch_b, 4, kernel_sizes=(13, 15, 17), filters=64)

    # ---- Fusion ----
    fused = layers.Concatenate(name='dual_fuse')([ch_a, ch_b])
    fused = layers.Conv1D(128, 1, padding='same', name='fuse_conv')(fused)
    fused = layers.BatchNormalization(name='fuse_bn')(fused)
    fused = layers.Activation('relu', name='fuse_relu')(fused)

    # ---- Classifier ----
    x = layers.GlobalAveragePooling1D(name='gap')(fused)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inp, out, name='DCMS1DCNN')
