# WavFry

Noise-robust phonation detection framework for conversational speech, supporting dual-task creaky voice (vocal fry) and breathy voice detection across multiple self-supervised speech backbones.

---

## 🌐 Interactive Demo

Audio examples and model predictions across 18 acoustic degradation conditions:  
👉 [https://mumu233desu.github.io/wavfry-demo/](https://mumu233desu.github.io/wavfry-demo/)

---

## 📦 Pretrained Checkpoints (Release v1.0.0)

Pretrained delta checkpoints are available on [GitHub Releases (v1.0.0)](https://github.com/mumu233desu/WavFry/releases/tag/v1.0.0):

| Checkpoint | SSL Backbone | Tasks / Role | Creaky PR-AUC | Creaky ROC-AUC | Optimal Thresholds |
| :--- | :--- | :--- | :---: | :---: | :---: |
| [`wavfry_wavlm_baseplus.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_wavlm_baseplus.pt) | `microsoft/wavlm-base-plus` | Primary model (Creaky + Breathy, VSDC-Net) | **0.9133** | **0.9841** | `creaky: 0.560 / breathy: 0.440` |
| [`wavfry_wavlm_base.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_wavlm_base.pt) | `microsoft/wavlm-base` | Backbone comparison (WavLM-Base) | 0.9132 | 0.9850 | `creaky: 0.549` |
| [`wavfry_hubert_base.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_hubert_base.pt) | `facebook/hubert-base-ls960` | Backbone comparison (HuBERT-Base) | 0.9135 | 0.9851 | `creaky: 0.570` |
| [`wavfry_wav2vec2_base.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_wav2vec2_base.pt) | `facebook/wav2vec2-base` | Backbone comparison (wav2vec 2.0-Base) | 0.9143 | 0.9844 | `creaky: 0.505 / breathy: 0.470` |
| [`wavfry_weak_off.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_weak_off.pt) | `microsoft/wavlm-base-plus` | Supervised baseline (No weak supervision) | 0.9122 | 0.9844 | `creaky: 0.580 / breathy: 0.500` |

*Base SSL backbones are automatically downloaded from Hugging Face Hub during inference.*

---

## Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/mumu233desu/WavFry.git
cd WavFry
uv sync
```

### 2. Python API
```python
import torch
from wavfry.model.wavlm_classifier import WavFryClassifier

# Load checkpoint and instantiate model
ckpt = torch.load("deployments/checkpoints_v1.0.0/wavfry_wavlm_baseplus.pt", map_location="cpu")
model = WavFryClassifier(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

# Forward pass on 16kHz audio waveform
waveform = torch.randn(1, 16000)  # [1, T]
with torch.no_grad():
    outputs = model(waveform)
    prob_creaky = torch.sigmoid(outputs["creaky"]).item()
    prob_breathy = torch.sigmoid(outputs["breathy"]).item()
    print(f"P(creaky)={prob_creaky:.3f}, P(breathy)={prob_breathy:.3f}")
```

### 3. CLI Prediction
```bash
# Predict Creaky Voice from audio files & TextGrid vowel intervals
uv run python scripts/predict.py \
  --checkpoint deployments/checkpoints_v1.0.0/wavfry_wavlm_baseplus.pt \
  --audio-dir /path/to/wav_files \
  --textgrid-dir /path/to/textgrid_files \
  --output-csv predictions_creaky.csv \
  --task creaky

# Predict Breathy Voice
uv run python scripts/predict.py \
  --checkpoint deployments/checkpoints_v1.0.0/wavfry_wavlm_baseplus.pt \
  --audio-dir /path/to/wav_files \
  --textgrid-dir /path/to/textgrid_files \
  --output-csv predictions_breathy.csv \
  --task breathy
```

### 4. Export to TextGrid (Optional)
```bash
uv run python scripts/export_textgrid.py \
  --csv predictions_creaky.csv \
  --tg-dir /path/to/textgrid_files \
  --output-dir /path/to/output_textgrids
```

---

## Training

```bash
# 1. Build segment manifests
uv run python scripts/build_manifest.py

# 2. Train with canonical frozen-backbone configuration
uv run python scripts/train.py --config config/ablation_backbone_wavlm_baseplus.yaml

# 3. Monitor with TensorBoard
uv run tensorboard --logdir runs/
```

---

## Documentation

Detailed methodology, theoretical limits, and evaluation specifications are available in [`docs/`](docs/README.md):
- [Theoretical limits of phonation decoupling](docs/discussion_unidentifiability_of_phonation_decoupling.md)
- [BreathyTTS synthesis methodology](docs/breathy_tts_synthesis_methodology.md)
- [Weak supervision & VSDC-Net specification](docs/weak_label_training_technical_specification.md)
- [Frame-level TAD Multi-IoU evaluation](docs/notes_frame_level_event_evaluation_methodology.md)
- [Data processing pipeline](docs/data_collection_and_processing_pipeline.md)
- [Representation & layer audit](docs/representation_cka_and_calibration_audit.md)
