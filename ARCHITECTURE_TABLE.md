# Corrected Comparison Architecture Table

Release **0.2.0 (2026-10-05)**. This table replaces the earlier CNN-headed
V-BiLSTM row. The other six model implementations are unchanged. Input is
`(256, 1)`, float32, seven classes. BN = BatchNormalization, GAP = global
average pooling. All convolution kernels are one-dimensional.

| Method | Complete architecture and hyperparameter settings |
|---|---|
| TCN (CNN-TCN) | Conv1D(16,k=64,s=8) -> BN -> ReLU -> MaxPool(2) -> Conv1D(32,k=3) -> BN -> ReLU -> Conv1D(64,k=3) -> BN -> ReLU -> Conv1D(128,k=3) -> BN -> ReLU -> 4 TCN residual blocks (dilations 1,2,4,8; filters 64; kernel 3; causal padding; two Conv-BN-ReLU-Dropout(0.2) sub-blocks per residual block) -> GAP -> Dense(128,ReLU) -> Dropout(0.3) -> Dense(7,Softmax). |
| CNN | Conv1D(64,k=8) -> BN -> ReLU -> MaxPool(2) -> Conv1D(64,k=5) -> BN -> ReLU -> MaxPool(2) -> Conv1D(64,k=3) -> BN -> ReLU -> MaxPool(2) -> GAP -> Dense(128,ReLU) -> Dropout(0.3) -> Dense(7,Softmax). |
| DCMS | Stem Conv1D(16,k=7) -> BN -> ReLU -> dual-channel MS blocks (A: kernels 3,5,7,9, filters 32; B: kernels 13,15,17,19, filters 32) -> MaxPool(2) per channel -> second MS blocks (A: kernels 3,5,7, filters 64; B: kernels 13,15,17, filters 64) -> concatenate -> Conv1D(128,k=1) -> BN -> ReLU -> GAP -> Dense(128,ReLU) -> Dropout(0.3) -> Dense(7,Softmax). MS blocks use additive branch fusion and residual shortcuts. |
| RMFF | Five parallel Conv1D(2,k=1..5)-BN-ReLU input branches -> concatenate -> 3 residual multi-scale fusion stages with 2,3,2 Inception-style modules, respectively (1/3/5 kernels and MaxPool branches; residual connections); each stage ends in MaxPool(2) -> BN -> Flatten after the final stage -> Dense(256,ReLU) -> Dropout(0.3) -> Dense(7,Softmax). Module widths are specified in README.md. |
| LSTM | LSTM(128,return_sequences=True) -> BN -> LSTM(64,return_sequences=False) -> Dense(128,ReLU) -> Dropout(0.3) -> Dense(7,Softmax). |
| V-BiLSTM (historical DK-KCFP-BiLSTM) | AdaptiveMedianFilter1D(window=5,threshold=3) -> WaveletVolterraExpansion(db6-like 3-level decomposition; componentwise terms up to degree 3) -> KroneckerFeaturePyramid(filters=32,k=3,dilation rates=2,4,6) -> MultiHeadAttention(num_heads=4,key_dim=8,value_dim=8,dropout=0) -> LayerNormalization(epsilon=0.001) -> Bidirectional(LSTM(150 units per direction,return_sequences=False),merge_mode=concat) -> Dropout(0.5) -> Dense(7,Softmax). |
| ROAT (ROAT-Spectrum-CNN) | ROATEnvelopeSpectrum(Hilbert orthogonal axis -> max(abs(x),abs(Hilbert(x))) -> real FFT magnitude spectrum -> log1p) -> BN -> Conv1D(32,k=5) -> BN -> ReLU -> MaxPool(2) -> Conv1D(64,k=3) -> BN -> ReLU -> MaxPool(2) -> Conv1D(64,k=3) -> BN -> ReLU -> GAP -> Dense(64,ReLU) -> Dropout(0.3) -> Dense(7,Softmax). |

**Settings for every row:** Adam (learning rate = 0.001), categorical
cross-entropy, batch size 32, 1 epoch. These are protocol metadata, not a
training pipeline in this release. The historical V-BiLSTM checkpoint's
compile configuration and adjacent training config confirm these settings.

## V-BiLSTM shapes and parameters

Batch dimension omitted. Checked against the historical saved model.

| Layer | Output shape | Parameters |
|---|---|---:|
| Input | 256 x 1 | 0 |
| AdaptiveMedianFilter1D | 256 x 1 | 0 |
| WaveletVolterraExpansion | 256 x 21 | 0 |
| KroneckerFeaturePyramid | 256 x 32 | 7,232 |
| MultiHeadAttention | 256 x 32 | 4,224 |
| LayerNormalization | 256 x 32 | 64 |
| Bidirectional LSTM | 300 | 219,600 |
| Dropout | 300 | 0 |
| Dense softmax | 7 | 2,107 |
| **Total** | | **233,227** |

Total = 233,163 trainable + 64 non-trainable BatchNorm running statistics.
LSTM count: `2 * 4 * 150 * (32 + 150 + 1) = 219600`.
Classifier count: `300 * 7 + 7 = 2107`. Fixed wavelet constants are not
registered Keras weights and are not counted by count_params.

## Suggested manuscript wording

The following retains the user's citation numbering; reference [38] itself
was not independently rechecked in this code-restoration task.

> A locally implemented Volterra-inspired nonlinear expansion and BiLSTM
> classification baseline (V-BiLSTM) [38] was also included. Its frontend
> comprises adaptive median filtering, a db6-like multiresolution polynomial
> expansion, and a dilated feature pyramid, followed by self-attention and a
> bidirectional LSTM classifier. In addition, ROAT [8] was implemented as a
> ROAT-Spectrum-CNN baseline using its envelope-spectrum representation
> followed by a lightweight CNN classifier.

The local expansion uses componentwise powers, not a full third-order Volterra
tensor with all cross-lag interactions. This historical architecture has no
three-layer CNN head, GAP, or Dense(128) hidden classifier. Source/checkpoint
agreement does not prove exact reproduction of a paper's official code.
This standalone table does not modify the manuscript or appendix PDF.
