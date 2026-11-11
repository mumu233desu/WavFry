# WavFry v1.0.0: Pretrained Checkpoints & Dual-Task Ablation Suite

This release publishes the pretrained checkpoints corresponding to the paper:
> **WavFry: Noise-Robust Phonation Detection and Functional Representation Auditing in Conversational Speech**

## 📦 Released Model Weights

All weights are stored as delta checkpoints. The underlying self-supervised backbones (`WavLM`, `HuBERT`, `wav2vec 2.0`) should be automatically loaded from Hugging Face Hub during inference.

| File Name | Base Backbone | Role / Architecture Description | Creaky PR-AUC | Creaky ROC-AUC | Optimal Thresholds |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `wavfry_wavlm_baseplus.pt` | `microsoft/wavlm-base-plus` | Primary Dual-Task Model (Weak-On) **(Primary)** | 0.9133 | 0.9841 | `creaky: 0.560 / breathy: 0.440` |
| `wavfry_wavlm_base.pt` | `microsoft/wavlm-base` | Cross-Backbone Comparison (WavLM-Base) | 0.9132 | 0.9850 | `creaky: 0.549` |
| `wavfry_hubert_base.pt` | `facebook/hubert-base-ls960` | Cross-Backbone Comparison (HuBERT-Base) | 0.9135 | 0.9851 | `creaky: 0.570` |
| `wavfry_wav2vec2_base.pt` | `facebook/wav2vec2-base` | Cross-Backbone Comparison (wav2vec 2.0-Base) | 0.9143 | 0.9844 | `creaky: 0.505 / breathy: 0.470` |
| `wavfry_weak_off.pt` | `microsoft/wavlm-base-plus` | Ablation Baseline (Weak-Off, Supervised Only) | 0.9122 | 0.9844 | `creaky: 0.580 / breathy: 0.500` |
| `wavfry_weak_mil_positive_only.pt` | `microsoft/wavlm-base-plus` | Ablation Variant (Weak MIL: Positive Bags Only) | 0.9132 | 0.9839 | `creaky: 0.560 / breathy: 0.480` |
| `wavfry_weak_mil_soft_weight.pt` | `microsoft/wavlm-base-plus` | Ablation Variant (Weak MIL: Soft Instance Weighting) | 0.9133 | 0.9842 | `creaky: 0.555 / breathy: 0.440` |

---

## 🔒 SHA-256 Checksums

```
1356529f2fac48669165a3e26f68730c5e7adcd2ef20e38d633b0e3d461795fd  wavfry_wavlm_baseplus.pt
bd4b17b88ead24f45cb57cab33f43ad1ebd77ca94e49847378f0d4f70ec9f609  wavfry_wavlm_base.pt
b2b30c1b81e40cc7e530c62c48ebb627a8f00d1859a1a6540118eb96a19ddce6  wavfry_hubert_base.pt
8d808e9ce58900d3dfb8166d3b3c5393782f1eb63c34e6c4dfd06355ddde7814  wavfry_wav2vec2_base.pt
256d9f042cd33a1ba84d4b953a55400b8f7297fdd24898f935d1ce0021879ae5  wavfry_weak_off.pt
cf0d0bf5d004fa569e63998227e86515d3a9bacd9a3f33d2234cc20cface9ced  wavfry_weak_mil_positive_only.pt
a4e693f7070ec99b532a7239922acbe80c3466f2ecdbb0b845bdb5f3ddbece80  wavfry_weak_mil_soft_weight.pt
```
