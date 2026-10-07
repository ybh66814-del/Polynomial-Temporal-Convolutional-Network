# Changelog

## Repository publication - 2026-10-08

- Prepared the v0.2.0 comparison architecture package and user-supplied
  Supplementary Material.pdf without modifying the model definitions or PDF.
- Replaced the previous current-tree Appendix.pdf and README; Git history
  and a local pre-publication backup preserve the old contents.
- Added supplementary-document links to the English and Chinese READMEs.

## 0.2.0 - 2026-10-05

- Restored the historical median/Wavelet/KCFP/attention/BiLSTM implementation.
- Corrected the V-BiLSTM architecture table and documentation: four attention
  heads, key dimension 8, LayerNorm epsilon 0.001, 150 LSTM units per direction,
  concatenation, Dropout(0.5), and direct Dense(7) classification.
- V-BiLSTM now has 233,227 parameters, replacing the 72,775-parameter CNN variant.
- Six other model modules are unchanged; v0.1.0 folder and ZIP are preserved.
- Historical checkpoint validation is reported separately; no trained weights
  or datasets are included and no new training or performance evaluation is claimed.

## 0.1.0 - 2026-10-04

- Initial architecture-only export with seven models in one module directory.
- Preserved copied function/class implementations and recorded source hashes.
- Added a median-enabled factory for the table's CNN-headed V-BiLSTM entry.
- Preserved RMFF's user-confirmed three-stage, seven-module architecture.
- Added English/Chinese documentation, pinned runtime requirements, protocol
  metadata, CLI summaries, and dataset-free architecture tests.
- Passed nine tests both directly and from an isolated wheel installation.
- Confirmed identical outputs against all seven local source implementations
  with shared weights on synthetic probes.
- No data, trained weights, training pipeline, or historical accuracy claims.
- Public license selection remains pending; no remote publishing performed.
