# Signal Comparison Model Architectures

Companion repository for the Polynomial Temporal Convolutional Network project.
Repository package prepared on **2026-10-08**.

- [Supplementary Material (PDF)](Supplementary%20Material.pdf)
- [Comparison architecture table](ARCHITECTURE_TABLE.md)
- [Validation report](VALIDATION.md)

The supplied PDF is distributed unchanged. The code below contains the seven
comparison architectures, not the PTCN main model or its training pipeline.

Architecture-only TensorFlow/Keras release, version **0.2.0 (2026-10-05)**.
See [Chinese documentation](README.zh-CN.md).

This repository contains the seven local implementations corresponding to the
corrected comparison architecture table. It includes **no dataset, trained
weights, training loop, evaluation pipeline, or claimed classification scores**.
It does not depend on the original project directory.

## Important model identities

- **TCN** in the table means **CNN-TCN**, not the standalone TCN ablation.
- **CNN** means the three-layer Classic CNN, not the 155K parameter-matched CNN.
- **V-BiLSTM** restores the historical **DK-KCFP-BiLSTM**: median filter,
  wavelet/polynomial frontend, KCFP, self-attention, LayerNorm, and BiLSTM.
  It has **233,227 parameters**. The CNN head from v0.1.0 is not included.
  The archived source and a saved historical checkpoint were checked locally;
  this is not a claim of exact equivalence to the referenced paper.
  See [the corrected architecture table](ARCHITECTURE_TABLE.md).
- **RMFF** retains the existing **three stages with 2, 3, and 2 modules**
  (seven residual multi-scale modules in total), as explicitly confirmed by the
  user. It is not reduced to three individual modules.

## Installation and inspection

The locally verified environment is Python 3.8.20, TensorFlow 2.13.0,
Keras 2.13.1, NumPy 1.24.3 on Windows CPU. Dependencies are pinned to this
TensorFlow/Keras generation; Keras 3 is not claimed compatible. Python
3.8-3.11 is the declared installation range; only the recorded local runtime
has been executed here. The included GitHub workflow targets Python 3.10 on
Linux and has not been run on GitHub as part of this export.

From this repository directory, preferably in a new virtual environment:

```bash
python -m pip install .
python -m comparison_models --model all
python -m comparison_models --model tcn --summary
python -m unittest discover -s tests -v
```

If the pinned dependencies already exist, the inspection and test commands
also work directly from this directory, without installation or network access.

```python
import tensorflow as tf
from comparison_models import build_model, CUSTOM_OBJECTS

model = build_model("v_bilstm")  # Historical attention + BiLSTM, median enabled
x = tf.zeros((1, 256, 1), dtype=tf.float32)
probabilities = model(x, training=False)  # shape (1, 7)
model.summary()

# Optional serialization of a randomly initialized architecture:
# model.save("example.keras")
# restored = tf.keras.models.load_model(
#     "example.keras", compile=False, custom_objects=CUSTOM_OBJECTS
# )
```

`build_model(name)` fixes input shape `(256, 1)` and seven output classes.
The batch dimension is variable. The copied lower-level builders permit some
overrides for research, but those overrides are **not** the table configuration.
Use the default float32 Keras policy. Models are returned uncompiled.

## Model inventory

| Table label | Key | File | Builder used by the public factory |
|---|---|---|---|
| TCN | `tcn` | [tcn.py](comparison_models/tcn.py) | `build_cnn_tcn` |
| CNN | `cnn` | [cnn.py](comparison_models/cnn.py) | `build_classic_cnn` |
| DCMS | `dcms` | [dcms.py](comparison_models/dcms.py) | `build_dcms1dcnn` |
| RMFF | `rmff` | [rmff.py](comparison_models/rmff.py) | `build_rmffcnn` |
| LSTM | `lstm` | [lstm.py](comparison_models/lstm.py) | `build_lstm_wang` |
| V-BiLSTM | `v_bilstm` | [v_bilstm.py](comparison_models/v_bilstm.py) | `build_v_bilstm_table` calls historical `build_dk_kcfp_bilstm` |
| ROAT | `roat` | [roat.py](comparison_models/roat.py) | `build_roat_spectrum_cnn` |

All seven source modules are in **one directory, `comparison_models/`**.
The package includes only the requested builders and their supporting layers.

## Architecture contract

`C(f,k,s)` denotes a 1-D convolution; omitted stride is 1. Unless specified
otherwise, convolution padding is `same`. `BN` is BatchNormalization, `GAP`
is global average pooling, and all classifiers end in Dense(7, softmax).

| Model | Architecture |
|---|---|
| TCN | C(16,64,8)-BN-ReLU-MaxPool(2); C(32,3)-BN-ReLU; C(64,3)-BN-ReLU; C(128,3)-BN-ReLU; four 64-channel causal TCN residual blocks at dilation 1/2/4/8 and kernel 3; GAP-Dense(128,ReLU)-Dropout(0.3)-classifier. Each residual block has two Conv-BN-ReLU-Dropout(0.2) branches in sequence and a projected shortcut if necessary. |
| CNN | Three Conv-BN-ReLU-MaxPool(2) blocks, each 64 channels and kernels 8/5/3; GAP-Dense(128,ReLU)-Dropout(0.3)-classifier. |
| DCMS | Stem C(16,7)-BN-ReLU; A: kernels 3/5/7/9, 32 channels, then MaxPool(2), then kernels 3/5/7, 64 channels; B: kernels 13/15/17/19, 32 channels, then MaxPool(2), then kernels 13/15/17, 64 channels; concatenate A/B; C(128,1)-BN-ReLU; GAP-Dense(128,ReLU)-Dropout(0.3)-classifier. MS branches are added and have residual shortcuts. |
| RMFF | Five parallel C(2,k), k=1..5, each BN-ReLU; concatenate; three stages with 2/3/2 residual Inception-style modules, each stage ending in MaxPool(2)-BN; Flatten-Dense(256,ReLU)-Dropout(0.3)-classifier. Modules use 1-D kernels 1/3/5 and a pooled branch. |
| LSTM | LSTM(128,return_sequences=True)-BN-LSTM(64)-Dense(128,ReLU)-Dropout(0.3)-classifier. |
| V-BiLSTM | AdaptiveMedianFilter1D(window=5)-WaveletVolterraExpansion(levels=3)-KroneckerFeaturePyramid(filters=32,dilations=2/4/6)-MultiHeadAttention(heads=4,key_dim=8,value_dim=8)-LayerNormalization(epsilon=0.001)-Bidirectional(LSTM(150,return_sequences=False),merge_mode=concat)-Dropout(0.5)-classifier. |
| ROAT | Hilbert orthogonal axis; max(abs(x),abs(Hilbert(x))); abs(real FFT); log1p; BN; C(32,5)-BN-ReLU-MaxPool(2); C(64,3)-BN-ReLU-MaxPool(2); C(64,3)-BN-ReLU; GAP-Dense(64,ReLU)-Dropout(0.3)-classifier. |

### Details the abbreviated table does not specify

These are preserved from the existing implementation, not independently
inferred from the table:

- BatchNorm uses Keras defaults (momentum 0.99, epsilon 0.001). Standard
  Conv1D layers use bias; KCFP dilated branches explicitly use
  `use_bias=False`. Other initializer/default values
  are those of the pinned Keras version and the copied code.
- TCN residual addition is followed by ReLU. Its first shortcut projects
  128 channels to 64 using a 1x1 convolution.
- RMFF module widths are tuples `(pointwise, reduce3, out3, reduce5, out5)`:
  stage 1: `(16,24,32,4,8)` twice; stage 2: `(32,48,64,8,16)` three times;
  stage 3: `(64,96,128,16,32)` then `(64,96,128,13,32)`.
  The pooled branch outputs `pointwise` channels. Shortcut projection uses BN.
- The median filter only replaces samples whose deviation from the window
  median exceeds `3 * (mean absolute deviation from the signal mean + 1e-6)`;
  reflection padding is used. This is not unconditional median smoothing.
- BiLSTM uses 150 units **per direction**, concatenating to 300 features.
  Its internal dropout and recurrent dropout are zero; the following separate
  Dropout layer has rate 0.5. The classifier is directly Dense(7), with no
  Dense(128), GAP, or three-layer CNN head. Attention has no additional external
  residual connection and uses dropout 0.0.
- Wavelet expansion is a **db6-like local implementation**, not a claim of
  exact equivalence to a standard wavelet library. Raw/approximation/detail
  channels are concatenated with their squares and cubes. There are no
  general cross-lag third-order Volterra tensor terms.
- KCFP combines three dilated convolutions and pointwise interactions
  `b0*b1 + b1*b2`, mixed with a 1x1 convolution, BN and LeakyReLU(0.1).
  The name does not imply an explicit full Kronecker-product tensor.
- ROAT takes raw 256-point time-domain input and returns a 129-bin spectrum
  internally. Its envelope is a maximum of two absolute axes, not the usual
  analytic-signal magnitude. No external spectrum preprocessing is required.

## Training settings: metadata only

The supplied table specifies Adam, learning rate 0.001, categorical
cross-entropy, batch size 32, and 1 epoch for all seven models. These settings
are recorded in [table_protocol.json](table_protocol.json); there is no
training pipeline in this release. One epoch does not define a step count
without a dataset and batching protocol. Data splitting, augmentation,
normalization, seeds, and checkpoint selection are outside this package.

This is **architecture correspondence**, not a reproduction of historical
accuracy, FLOPs, serialized model sizes, or dataset-dependent results.

## Validation and provenance

- Tests check declared parameter counts, structural contracts, finite
  probability outputs, variable batches, traced inference, and Keras save/load.
- AST hashes in [source_manifest.json](source_manifest.json) trace the copied
  functions/classes to the local source snapshot. Definitions were extracted
  unchanged; only imports, package layout, and the table-specific factory were
  added. The restored archived builder always includes the DK median filter.
- [VALIDATION.md](VALIDATION.md) records the executed checks, their scope, and
  limitations. No original project is required to run the shipped tests.
- Original code and historical experiment outputs were not overwritten.
- The v0.1.0 folder and ZIP remain unchanged. Six other architecture modules
  are byte-identical to v0.1.0. Only the V-BiLSTM model identity changes.

## Publishing

This folder is the repository root. It contains README files, model modules,
dependency metadata, architecture tests, source provenance, and a GitHub CI
workflow. Generated weights, caches, datasets, and output directories are
ignored. The earlier Appendix.pdf and README were replaced in the current
branch; their prior versions remain accessible in Git history.

**Before a public release, confirm ownership and choose a license.** No
license grant was invented. See [NOTICE.md](NOTICE.md) for attribution and
the difference between these local adaptations and official implementations.
