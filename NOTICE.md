# Scope, provenance, and licensing

This repository packages local TensorFlow comparison implementations, not
official source code released by the cited paper authors. Architectural
adaptations are documented in README.md. Publication attribution does not
establish source-code ownership or grant redistribution rights.

The source owner has not specified a distribution license for this export.
No MIT, Apache, or other license has been assigned on their behalf. Confirm
ownership and any third-party obligations, and add an appropriate LICENSE
before making a public open-source release. This is a packaging readiness
condition, not a claim that any particular license restriction was found.

Source descriptions retained from the local implementation:

- CNN-TCN: Liu et al. (2025), Measurement 256, 118482;
  DOI: 10.1016/j.measurement.2025.118482.
- DCMS1DCNN: Li et al. (2025), IEEE Transactions on Instrumentation and
  Measurement; DOI: 10.1109/TIM.2025.3527087.
- LSTM: Wang et al. (2022), Energy 239, 122298;
  DOI: 10.1016/j.energy.2021.122298.
- RMFFCNN: Zhu et al. (2023), IEEE Internet of Things Journal 10(8),
  7393-7404. The local model uses 1-D signals instead of kurtosis images.
- ROAT: Hajnayeb et al. (2024), IEEE Transactions on Instrumentation and
  Measurement 73. The local implementation adds a trainable CNN classifier.
- DK-KCFP: local adaptation associated with the title
  "Physics-inspired deep learning network using dilated kronecker convolution
  for rotary machines under variable operating conditions". This release
  restores the historical attention/BiLSTM head from the local source backup.
- Classic CNN: local general-purpose three-layer 1-D CNN baseline.

Bibliographic metadata above is preserved from source comments, not newly
verified against publisher records. Supplementary Material.pdf is the
user-provided companion document and is included unchanged. No datasets,
trained weights, private absolute paths, or credentials are included.
