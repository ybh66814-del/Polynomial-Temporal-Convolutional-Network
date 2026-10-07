"""Dilated-Kronecker KCFP-BiLSTM baseline.

This module implements a compact reproduction of the paper model
"Physics-inspired deep learning network using dilated kronecker convolution
for rotary machines under variable operating conditions" for the local
PTCN-MGF signal format.
"""

from __future__ import annotations

import tensorflow as tf


class AdaptiveMedianFilter1D(tf.keras.layers.Layer):
    """Windowed median spike suppression for 1-D vibration signals."""

    def __init__(self, window_size: int = 5, threshold: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        self.window_size = int(window_size)
        self.threshold = float(threshold)

    def call(self, inputs, training=None):
        del training
        x = tf.convert_to_tensor(inputs)
        pad = self.window_size // 2
        padded = tf.pad(x, [[0, 0], [pad, pad], [0, 0]], mode="REFLECT")
        windows = tf.signal.frame(
            tf.transpose(padded, [0, 2, 1]),
            frame_length=self.window_size,
            frame_step=1,
            axis=-1,
        )
        windows = tf.transpose(windows, [0, 2, 3, 1])
        median = tf.sort(windows, axis=2)[:, :, pad, :]
        residual = tf.abs(x - median)
        scale = tf.reduce_mean(tf.abs(x - tf.reduce_mean(x, axis=1, keepdims=True)), axis=1, keepdims=True)
        mask = residual > self.threshold * (scale + 1e-6)
        return tf.where(mask, median, x)

    def get_config(self):
        config = super().get_config()
        config.update({"window_size": self.window_size, "threshold": self.threshold})
        return config


class WaveletVolterraExpansion(tf.keras.layers.Layer):
    """Fixed db6-like multiresolution front-end with third-order Volterra terms."""

    _DB6_LO = (
        0.001077301085308479,
        0.004777257510945511,
        -0.0005538422011614961,
        -0.03158203931748603,
        0.027522865530305728,
        0.09750160558732247,
        -0.12976686756709563,
        -0.22626469396544,
        0.3152503517091982,
        0.7511339080210954,
        0.4946238903984531,
        0.11154074335008017,
    )

    def __init__(self, levels: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.levels = int(levels)

    def build(self, input_shape):
        lo = tf.constant(self._DB6_LO, dtype=self.dtype)
        hi = tf.reverse(lo * tf.constant([(-1.0) ** i for i in range(len(self._DB6_LO))], dtype=self.dtype), axis=[0])
        self.lo_filter = tf.reshape(lo, [-1, 1, 1])
        self.hi_filter = tf.reshape(hi, [-1, 1, 1])
        super().build(input_shape)

    def call(self, inputs, training=None):
        del training
        x = tf.convert_to_tensor(inputs)
        approx = x
        features = [x]
        original_length = tf.shape(x)[1]
        for level in range(1, self.levels + 1):
            smooth = tf.nn.conv1d(approx, self.lo_filter, stride=1, padding="SAME")
            detail = tf.nn.conv1d(approx, self.hi_filter, stride=1, padding="SAME")
            approx = tf.nn.avg_pool1d(smooth, ksize=2, strides=2, padding="SAME")
            repeat = 2 ** (level - 1)
            detail_up = tf.repeat(detail, repeats=repeat, axis=1)[:, :original_length, :]
            approx_up = tf.repeat(approx, repeats=repeat * 2, axis=1)[:, :original_length, :]
            features.extend([approx_up, detail_up])
        base = tf.concat(features, axis=-1)
        return tf.concat([base, tf.square(base), tf.pow(base, 3)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"levels": self.levels})
        return config


class KroneckerFeaturePyramid(tf.keras.layers.Layer):
    """Multi-branch dilated convolution pyramid with Kronecker-style interaction."""

    def __init__(
        self,
        filters: int = 32,
        kernel_size: int = 3,
        dilation_rates: tuple[int, ...] = (2, 4, 6),
        negative_slope: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.dilation_rates = tuple(int(rate) for rate in dilation_rates)
        self.negative_slope = float(negative_slope)
        self.branches = []
        self.mix = tf.keras.layers.Conv1D(self.filters, 1, padding="same")
        self.norm = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.LeakyReLU(alpha=self.negative_slope)

    def build(self, input_shape):
        self.branches = [
            tf.keras.layers.Conv1D(
                self.filters,
                self.kernel_size,
                padding="same",
                dilation_rate=rate,
                use_bias=False,
                name=f"dilated_conv_d{rate}",
            )
            for rate in self.dilation_rates
        ]
        super().build(input_shape)

    def call(self, inputs, training=None):
        branch_outputs = [branch(inputs) for branch in self.branches]
        summed = tf.add_n(branch_outputs)
        if len(branch_outputs) >= 3:
            interaction = branch_outputs[0] * branch_outputs[1] + branch_outputs[1] * branch_outputs[2]
            summed = summed + self.mix(interaction)
        return self.act(self.norm(summed, training=training))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rates": list(self.dilation_rates),
                "negative_slope": self.negative_slope,
            }
        )
        return config


def build_dk_kcfp_bilstm(
    num_classes: int,
    input_shape: tuple[int, int] = (256, 1),
    filters: int = 32,
    lstm_units: int = 150,
    attention_heads: int = 4,
    dropout: float = 0.5,
) -> tf.keras.Model:
    """Build the paper-inspired DK/KCFP-BiLSTM classifier."""

    inputs = tf.keras.Input(shape=input_shape, name="signal")
    x = AdaptiveMedianFilter1D(name="adaptive_median_filter")(inputs)
    x = WaveletVolterraExpansion(name="wavelet_volterra_expansion")(x)
    x = KroneckerFeaturePyramid(filters=filters, name="kcfp")(x)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=max(8, filters // attention_heads),
        name="self_attention",
    )(x, x)
    x = tf.keras.layers.LayerNormalization(name="attention_norm")(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=False),
        name="bilstm",
    )(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name="dk_kcfp_bilstm")


CUSTOM_OBJECTS = {
    "AdaptiveMedianFilter1D": AdaptiveMedianFilter1D,
    "WaveletVolterraExpansion": WaveletVolterraExpansion,
    "KroneckerFeaturePyramid": KroneckerFeaturePyramid,
}


__all__ = [
    "AdaptiveMedianFilter1D",
    "WaveletVolterraExpansion",
    "KroneckerFeaturePyramid",
    "build_dk_kcfp_bilstm",
    "CUSTOM_OBJECTS",
]
