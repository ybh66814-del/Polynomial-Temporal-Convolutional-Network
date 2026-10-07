"""Dataset-free checks for the fixed comparison-table configurations."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from comparison_models import CUSTOM_OBJECTS, MODEL_BUILDERS, build_model


EXPECTED_PARAMS = {
    "tcn": 165111, "cnn": 43463, "dcms": 211079, "rmff": 2690312,
    "lstm": 125703, "v_bilstm": 233227, "roat": 24011,
}


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

    def test_inventory_and_parameters(self):
        self.assertEqual(set(MODEL_BUILDERS), set(EXPECTED_PARAMS))
        for name, expected in EXPECTED_PARAMS.items():
            with self.subTest(model=name):
                model = build_model(name)
                self.assertEqual(model.input_shape, (None, 256, 1))
                self.assertEqual(model.output_shape, (None, 7))
                self.assertEqual(model.count_params(), expected)
                trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
                fixed = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
                self.assertEqual(trainable + fixed, expected)
                self.assertTrue(all(w.dtype == tf.float32 for w in model.weights))

    def test_cnn_and_lstm_contracts(self):
        cnn = build_model("cnn")
        convs = [x for x in cnn.layers if isinstance(x, tf.keras.layers.Conv1D)]
        self.assertEqual([(x.filters, x.kernel_size) for x in convs],
                         [(64, (8,)), (64, (5,)), (64, (3,))])
        self.assertEqual(sum(isinstance(x, tf.keras.layers.MaxPooling1D) for x in cnn.layers), 3)
        self.assertEqual(cnn.get_layer("fc1").units, 128)
        self.assertEqual(cnn.get_layer("dropout").rate, 0.3)
        lstm = build_model("lstm")
        self.assertEqual(lstm.get_layer("lstm1").units, 128)
        self.assertTrue(lstm.get_layer("lstm1").return_sequences)
        self.assertEqual(lstm.get_layer("lstm2").units, 64)
        self.assertFalse(lstm.get_layer("lstm2").return_sequences)
        self.assertEqual([type(x).__name__ for x in lstm.layers],
                         ["InputLayer", "LSTM", "BatchNormalization", "LSTM", "Dense", "Dropout", "Dense"])

    def test_tcn_contract(self):
        model = build_model("tcn")
        wide = model.get_layer("cnn_wide")
        self.assertEqual((wide.filters, wide.kernel_size, wide.strides), (16, (64,), (8,)))
        for block, rate in enumerate((1, 2, 4, 8), 1):
            for branch in (1, 2):
                prefix = "tcn_b{}_d{}_{}".format(block, rate, branch)
                layer = model.get_layer(prefix)
                self.assertEqual((layer.filters, layer.kernel_size, layer.dilation_rate, layer.padding),
                                 (64, (3,), (rate,), "causal"))
                self.assertEqual(model.get_layer(prefix + "_do").rate, 0.2)
            self.assertIsInstance(model.get_layer("tcn_b{}_res".format(block)), tf.keras.layers.Add)

    def test_dcms_and_rmff_contracts(self):
        dcms = build_model("dcms")
        for block, kernels, filters in ((1, (3, 5, 7, 9), 32), (2, (3, 5, 7), 64),
                                        (3, (13, 15, 17, 19), 32), (4, (13, 15, 17), 64)):
            for kernel in kernels:
                layer = dcms.get_layer("ms_b{}_k{}".format(block, kernel))
                self.assertEqual((layer.filters, layer.kernel_size), (filters, (kernel,)))
        rmff = build_model("rmff")
        for stage, count in enumerate((2, 3, 2), 1):
            residuals = [x for x in rmff.layers if x.name.startswith("rmff_b{}_m".format(stage))
                         and x.name.endswith("_residual")]
            self.assertEqual(len(residuals), count)
            self.assertIsInstance(rmff.get_layer("rmff_b{}_pool".format(stage)), tf.keras.layers.MaxPooling1D)
        self.assertIsInstance(rmff.get_layer("flatten"), tf.keras.layers.Flatten)
        self.assertEqual(rmff.get_layer("fc1").units, 256)
        for kernel in range(1, 6):
            layer = rmff.get_layer("rmff_input_k{}".format(kernel))
            self.assertEqual((layer.filters, layer.kernel_size), (2, (kernel,)))

    def test_v_bilstm_matches_historical_architecture(self):
        model = build_model("v_bilstm")
        self.assertEqual([type(x).__name__ for x in model.layers], [
            "InputLayer", "AdaptiveMedianFilter1D", "WaveletVolterraExpansion", "KroneckerFeaturePyramid",
            "MultiHeadAttention", "LayerNormalization", "Bidirectional", "Dropout", "Dense",
        ])
        self.assertEqual(model.get_layer("adaptive_median_filter").window_size, 5)
        self.assertEqual(model.get_layer("wavelet_volterra_expansion").levels, 3)
        self.assertEqual(model.get_layer("kcfp").dilation_rates, (2, 4, 6))
        attention = model.get_layer("self_attention").get_config()
        self.assertEqual((attention["num_heads"], attention["key_dim"], attention["value_dim"]), (4, 8, 8))
        self.assertEqual(attention["dropout"], 0.0)
        self.assertEqual(model.get_layer("attention_norm").epsilon, 0.001)
        bilstm = model.get_layer("bilstm")
        self.assertEqual(bilstm.merge_mode, "concat")
        self.assertEqual(bilstm.output_shape, (None, 300))
        for layer in (bilstm.forward_layer, bilstm.backward_layer):
            self.assertEqual(layer.units, 150)
            self.assertFalse(layer.return_sequences)
            self.assertEqual(layer.dropout, 0.0)
            self.assertEqual(layer.recurrent_dropout, 0.0)
        self.assertFalse(bilstm.forward_layer.go_backwards)
        self.assertTrue(bilstm.backward_layer.go_backwards)
        self.assertEqual(model.get_layer("classifier").units, 7)
        self.assertEqual(model.get_layer("dropout").rate, 0.5)

    def test_roat_against_numpy_reference(self):
        x = np.random.RandomState(7).normal(size=(2, 256)).astype("float32")
        h = np.concatenate(([1.], np.full(127, 2.), [1.], np.zeros(127)))
        hilbert = np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1).imag
        envelope = np.maximum(np.abs(x), np.abs(hilbert))
        expected = np.log1p(np.abs(np.fft.rfft(envelope, axis=-1)))
        model = build_model("roat")
        got = model.get_layer("roat_spectrum")(x[..., None]).numpy()[..., 0]
        self.assertEqual(got.shape, (2, 129))
        # TensorFlow complex64 FFT vs NumPy complex128 FFT accumulates roundoff.
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-4)

    def test_inference_and_roundtrip(self):
        rng = np.random.RandomState(123)
        x = rng.normal(0, 0.2, size=(3, 256, 1)).astype("float32")
        x[0] = 0
        x[1] = 0
        x[1, 128, 0] = 1
        with tempfile.TemporaryDirectory(prefix="architecture-check-") as directory:
            for name in MODEL_BUILDERS:
                with self.subTest(model=name):
                    model = build_model(name)
                    expected = model(x, training=False).numpy()
                    self.assertEqual(expected.shape, (3, 7))
                    self.assertTrue(np.isfinite(expected).all())
                    self.assertTrue((expected >= 0).all())
                    np.testing.assert_allclose(expected.sum(axis=1), 1, atol=1e-6)
                    self.assertEqual(model(x[:1], training=False).shape, (1, 7))
                    traced = tf.function(lambda value: model(value, training=False),
                                         input_signature=[tf.TensorSpec([None, 256, 1], tf.float32)])
                    np.testing.assert_allclose(traced(x).numpy(), expected, rtol=1e-5, atol=1e-6)
                    path = str(Path(directory) / (name + ".keras"))
                    model.save(path)
                    restored = tf.keras.models.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)
                    self.assertEqual(restored.count_params(), model.count_params())
                    np.testing.assert_allclose(restored(x, training=False).numpy(), expected,
                                               rtol=1e-5, atol=1e-6)

    def test_copied_definitions_match_manifest(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
        for module, record in manifest["modules"].items():
            loaded = importlib.import_module("comparison_models." + module)
            content = Path(loaded.__file__).read_text(encoding="utf-8")
            lines = content.splitlines(keepends=True)
            parsed = ast.parse(content)
            nodes = {n.name: n for n in parsed.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            for definition in record["definitions"]:
                node = nodes[definition["name"]]
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                # Source hashes remain stable across Python AST schema versions.
                block = "".join(lines[start - 1:node.end_lineno]).rstrip()
                digest = hashlib.sha256(block.encode()).hexdigest()
                self.assertEqual(digest, definition["source_sha256"])

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            build_model("historical_bilstm")


if __name__ == "__main__":
    unittest.main()
