"""Seven architecture-only baselines matching the supplied comparison table."""

from .cnn import build_classic_cnn
from .dcms import build_dcms1dcnn
from .lstm import build_lstm_wang
from .rmff import build_rmffcnn
from .roat import ROATEnvelopeSpectrum, build_roat_spectrum_cnn
from .tcn import build_cnn_tcn
from .v_bilstm import (
    AdaptiveMedianFilter1D,
    KroneckerFeaturePyramid,
    WaveletVolterraExpansion,
    build_dk_kcfp_bilstm,
)

__version__ = "0.2.0"


def build_v_bilstm_table(input_shape=(256, 1), num_classes=7):
    """Historical median + wavelet/KCFP + attention + BiLSTM classifier."""
    return build_dk_kcfp_bilstm(
        num_classes=num_classes, input_shape=input_shape,
        filters=32, lstm_units=150, attention_heads=4, dropout=0.5,
    )


MODEL_BUILDERS = {
    "tcn": build_cnn_tcn,
    "cnn": build_classic_cnn,
    "dcms": build_dcms1dcnn,
    "rmff": build_rmffcnn,
    "lstm": build_lstm_wang,
    "v_bilstm": build_v_bilstm_table,
    "roat": build_roat_spectrum_cnn,
}

CUSTOM_OBJECTS = {
    cls.__name__: cls for cls in (
        ROATEnvelopeSpectrum, AdaptiveMedianFilter1D,
        WaveletVolterraExpansion, KroneckerFeaturePyramid,
    )
}


def build_model(name):
    """Build the fixed table configuration: float32 input (256, 1), 7 classes.

    Models are uncompiled and randomly initialized. Training is out of scope.
    Use the named builders directly only for explicitly different experiments.
    """
    if name not in MODEL_BUILDERS:
        raise ValueError("Unknown model {!r}; choose {}".format(name, ", ".join(MODEL_BUILDERS)))
    return MODEL_BUILDERS[name](input_shape=(256, 1), num_classes=7)


__all__ = ["build_model", "MODEL_BUILDERS", "CUSTOM_OBJECTS", "build_v_bilstm_table"]
