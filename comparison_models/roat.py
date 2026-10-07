"""Local comparison architecture; see README and source_manifest.json."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable(package="baselines")
class ROATEnvelopeSpectrum(layers.Layer):
    """Single-phase ROAT frontend followed by magnitude spectrum.

    Hajnayeb et al. define S-ROAT as the upper envelope of the original current
    and its Hilbert-transform orthogonal axis. Algorithmically, C[k] is the
    larger of abs(x[k]) and abs(H(x)[k]), then the spectrum of C is used for
    fault-component detection. This layer implements that deterministic
    preprocessing for 256-point local signals.
    """

    def call(self, inputs):
        x = tf.squeeze(inputs, axis=-1)
        n = tf.shape(x)[-1]
        spectrum = tf.signal.fft(tf.cast(x, tf.complex64))

        half = n // 2
        h = tf.concat(
            [
                tf.ones((1,), dtype=tf.float32),
                tf.ones((half - 1,), dtype=tf.float32) * 2.0,
                tf.ones((1,), dtype=tf.float32),
                tf.zeros((n - half - 1,), dtype=tf.float32),
            ],
            axis=0,
        )
        analytic = tf.signal.ifft(spectrum * tf.cast(h, tf.complex64))
        hilbert = tf.math.imag(analytic)
        envelope = tf.maximum(tf.abs(x), tf.abs(hilbert))
        magnitude = tf.abs(tf.signal.rfft(envelope))
        magnitude = tf.math.log1p(magnitude)
        return tf.expand_dims(magnitude, axis=-1)

    def get_config(self):
        return super().get_config()


def build_roat_spectrum_cnn(input_shape=(256, 1), num_classes=7):
    """S-ROAT feature frontend plus lightweight CNN classifier.

    The paper proposes ROAT as a signal-processing preprocessing method, not a
    trained classifier. For the local seven-class comparison, we reproduce the
    single-phase ROAT transform and attach a compact classifier so the method can
    be trained/evaluated under the same one-epoch protocol as the other models.
    """
    inp = layers.Input(shape=input_shape)
    x = ROATEnvelopeSpectrum(name='roat_spectrum')(inp)
    x = layers.BatchNormalization(name='roat_spectrum_norm')(x)

    x = layers.Conv1D(32, 5, padding='same', name='roat_conv1')(x)
    x = layers.BatchNormalization(name='roat_bn1')(x)
    x = layers.Activation('relu', name='roat_relu1')(x)
    x = layers.MaxPooling1D(2, name='roat_pool1')(x)

    x = layers.Conv1D(64, 3, padding='same', name='roat_conv2')(x)
    x = layers.BatchNormalization(name='roat_bn2')(x)
    x = layers.Activation('relu', name='roat_relu2')(x)
    x = layers.MaxPooling1D(2, name='roat_pool2')(x)

    x = layers.Conv1D(64, 3, padding='same', name='roat_conv3')(x)
    x = layers.BatchNormalization(name='roat_bn3')(x)
    x = layers.Activation('relu', name='roat_relu3')(x)

    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(64, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inp, out, name='ROAT_Spectrum_CNN')
