# Validation of the Historical BiLSTM Restoration

Release **0.2.0**, dated **2026-10-05**. No training or accuracy evaluation.

## Historical evidence

- Source: `archive/source_backups/dk_kcfp_bilstm_backup_20260630_100747.py`
  relative to the original project. The release's `v_bilstm.py` is a direct
  copy of this archived implementation, not a newly attached recurrent head.
- Checkpoint: `outputs_baselines/no_aug_single_epoch/dk_kcfp_bilstm/models/`
  `dk_kcfp_bilstm_no_aug_ep1_run1_final.keras`.
- Checkpoint SHA-256:
  `7ba8736fb1d183887ee29563b9d2f6d8b70720456760ce90cefb7097976bd4b7`.
- The checkpoint contains median filtering, WaveletVolterraExpansion, KCFP,
  4-head self-attention (key/value dimension 8), LayerNormalization
  (epsilon 0.001), a bidirectional 150-unit LSTM per direction, Dropout(0.5),
  and a direct seven-class classifier. It has no CNN classification head.
- Serialized per-layer configurations agree after ignoring layer names;
  every weight tensor shape agrees. Transferring historical checkpoint
  weights into the restored model gives **maximum output difference 0.0**
  over five probes (zero, impulse, sinusoid, higher-amplitude noise, noise).
- The checkpoint's compile metadata confirms Adam, learning rate 0.001,
  and categorical cross-entropy. Its adjacent training config records batch
  size 32 and one epoch. Generic unused configuration fields were not used to
  infer the architecture; the saved model and archived builder are authoritative.

The checkpoint was used only for local verification and is **not distributed**.
Detailed machine-readable evidence is in `validation_results.json`.

## Parameter and source parity checks

All seven models match their source layer configurations and produce identical
outputs with the same weights on the five probes. Counts include scalar Keras
model variables, exclude optimizer state, and exclude constants not registered
as Keras weights. These are not FLOPs or model-file-size measurements.

| Key | Total | Trainable | Non-trainable | Source output max difference |
|---|---:|---:|---:|---:|
| tcn | 165,111 | 163,607 | 1,504 | 0.0 |
| cnn | 43,463 | 43,079 | 384 | 0.0 |
| dcms | 211,079 | 209,511 | 1,568 | 0.0 |
| rmff | 2,690,312 | 2,685,082 | 5,230 | 0.0 |
| lstm | 125,703 | 125,447 | 256 | 0.0 |
| v_bilstm | 233,227 | 233,163 | 64 | 0.0 |
| roat | 24,011 | 23,689 | 322 | 0.0 |

## Standalone architecture tests

`python -m unittest discover -s tests -v`: **9 tests passed**.

Tests cover model inventory, parameters, input/output shape, float32 weights,
TCN/CNN/DCMS/RMFF/LSTM structural settings, the restored BiLSTM head, ROAT's
independent NumPy formula, source definition hashes, unknown-key rejection,
finite normalized outputs, eager variable batches, traced inference, and
save/reload round-trips for all seven models with custom layer registration
provided via `CUSTOM_OBJECTS`.

ROAT uses the pre-existing float32 FFT comparison tolerance (`atol=1e-4`,
`rtol=1e-5`). TensorFlow emitted retracing warnings because tests trace separate
functions for different architectures; assertions passed. Temporary random
weight files are not part of this release.

Environment: Windows CPU, Python 3.8.20, TensorFlow 2.13.0, Keras 2.13.1,
NumPy 1.24.3. GitHub-hosted CI, GPUs and Keras 3 are not locally verified.

An offline wheel build and `--no-deps --target` installation into a separate
directory also succeeded. The same **nine tests passed again under Python
isolated mode (`-I`)**, importing the installed package without the original
project. Dependencies came from the existing pinned environment, not a fresh
download. See `distribution_validation.json`.

## Preservation and scope

- Six non-BiLSTM architecture modules are byte-identical to v0.1.0.
- All previous release files and its ZIP retain their recorded SHA-256 values.
- Archived source, historical checkpoint, and its training configuration are
  unchanged. Original working model sources have not been edited.
- The corrected table and suggested manuscript wording are in
  `ARCHITECTURE_TABLE.md`; original manuscript/appendix PDFs are not modified.
- This verifies restoration of the **local historical implementation**, not
  exact agreement with reference [38] or an official author repository.
- No accuracy, robustness, FLOPs, or runtime result is newly claimed. License
  selection remains with the owner; nothing has been pushed to a remote.
