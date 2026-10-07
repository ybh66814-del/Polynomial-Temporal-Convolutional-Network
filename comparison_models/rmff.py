"""Local comparison architecture; see README and source_manifest.json."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def _rmff_module_1d(x, block_id, module_id, filters_1x1, reduce_3x3, filters_3x3,
                    reduce_5x5, filters_5x5):
    """1-D adaptation of the RMFFCNN residual multiscale fusion module.

    The paper defines the model on 28x28x3 kurtosis images. This baseline keeps
    the published multiscale/residual topology but maps 2-D kernels to 1-D
    kernels so it can train on the local 256x1 signal format.
    """
    name = f'rmff_b{block_id}_m{module_id}'

    p1 = layers.Conv1D(filters_1x1, 1, padding='same', name=f'{name}_p1_1x1')(x)
    p1 = layers.BatchNormalization(name=f'{name}_p1_bn')(p1)
    p1 = layers.Activation('relu', name=f'{name}_p1_relu')(p1)

    p2 = layers.Conv1D(reduce_3x3, 1, padding='same', name=f'{name}_p2_reduce')(x)
    p2 = layers.BatchNormalization(name=f'{name}_p2_reduce_bn')(p2)
    p2 = layers.Activation('relu', name=f'{name}_p2_reduce_relu')(p2)
    p2 = layers.Conv1D(filters_3x3, 3, padding='same', name=f'{name}_p2_3x3')(p2)
    p2 = layers.BatchNormalization(name=f'{name}_p2_bn')(p2)
    p2 = layers.Activation('relu', name=f'{name}_p2_relu')(p2)

    p3 = layers.Conv1D(reduce_5x5, 1, padding='same', name=f'{name}_p3_reduce')(x)
    p3 = layers.BatchNormalization(name=f'{name}_p3_reduce_bn')(p3)
    p3 = layers.Activation('relu', name=f'{name}_p3_reduce_relu')(p3)
    p3 = layers.Conv1D(filters_5x5, 5, padding='same', name=f'{name}_p3_5x5')(p3)
    p3 = layers.BatchNormalization(name=f'{name}_p3_bn')(p3)
    p3 = layers.Activation('relu', name=f'{name}_p3_relu')(p3)

    p4 = layers.MaxPooling1D(3, strides=1, padding='same', name=f'{name}_p4_pool')(x)
    p4 = layers.Conv1D(filters_1x1, 1, padding='same', name=f'{name}_p4_1x1')(p4)
    p4 = layers.BatchNormalization(name=f'{name}_p4_bn')(p4)
    p4 = layers.Activation('relu', name=f'{name}_p4_relu')(p4)

    fused = layers.Concatenate(name=f'{name}_concat')([p1, p2, p3, p4])
    if x.shape[-1] != fused.shape[-1]:
        shortcut = layers.Conv1D(int(fused.shape[-1]), 1, padding='same',
                                 name=f'{name}_shortcut_proj')(x)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
    else:
        shortcut = x
    out = layers.Add(name=f'{name}_residual')([shortcut, fused])
    return layers.Activation('relu', name=f'{name}_out')(out)


def build_rmffcnn(input_shape=(256, 1), num_classes=7):
    """RMFFCNN signal baseline adapted from Zhu et al. (2023).

    Original paper input: 28x28x3 images generated from kurtosis features.
    Local adaptation: direct 1-D signals with the same multiscale fusion idea.
    """
    inp = layers.Input(shape=input_shape)

    input_branches = []
    for kernel_size in (1, 2, 3, 4, 5):
        branch = layers.Conv1D(2, kernel_size, padding='same',
                               name=f'rmff_input_k{kernel_size}')(inp)
        branch = layers.BatchNormalization(name=f'rmff_input_k{kernel_size}_bn')(branch)
        branch = layers.Activation('relu', name=f'rmff_input_k{kernel_size}_relu')(branch)
        input_branches.append(branch)
    x = layers.Concatenate(name='rmff_input_multiscale')(input_branches)

    block_configs = [
        (1, [(16, 24, 32, 4, 8), (16, 24, 32, 4, 8)]),
        (2, [(32, 48, 64, 8, 16), (32, 48, 64, 8, 16), (32, 48, 64, 8, 16)]),
        (3, [(64, 96, 128, 16, 32), (64, 96, 128, 13, 32)]),
    ]
    for block_id, modules in block_configs:
        for module_id, config in enumerate(modules, start=1):
            x = _rmff_module_1d(x, block_id, module_id, *config)
        x = layers.MaxPooling1D(2, name=f'rmff_b{block_id}_pool')(x)
        x = layers.BatchNormalization(name=f'rmff_b{block_id}_pool_bn')(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inp, out, name='RMFFCNN_1D')
