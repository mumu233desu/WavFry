# PRD: WavFry (WavLM Creaky Voice Detection)

## 1. Introduction/Overview
WavFry is a sub-project designed to detect creaky voice (vocal fry) phenomena within vowel segments of the CS-Dialogue bilingual corpus (English and Mandarin). By fine-tuning the WavLM pre-trained model on manually annotated data from 5 speakers, WavFry aims to provide a robust, deep learning-based alternative to existing rule-based detection pipelines (V1/V2) and external tools (DeepFry, creapy).

## 2. Goals
- Develop a deep learning model to accurately classify vowel segments as creaky or non-creaky based on audio and TextGrid inputs.
- Support both WAV and FLAC audio formats natively.
- Implement a flexible architecture supporting multi-model switching (WavLM base, base-plus, large).
- Optimize training for limited hardware (e.g., 8GB local GPU) and cloud environments (Google Colab).
- Provide a clear, manageable project structure with comprehensive YAML configuration files.

## 3. User Stories

### US-001: Data Preparation and Manifest Generation
**Description:** As a researcher, I want to automatically generate a data manifest from TextGrid annotations and audio files, so that I don't have to manually match files and parse intervals during training.

**Acceptance Criteria:**
- [x] Script successfully parses TextGrids to extract vowel intervals and labels.
- [x] Manifest CSV includes file paths, start/end times, phoneme, speaker ID, language, label, and dataset split.
- [x] Automatically filters out segments shorter than the configured minimum duration (e.g., 40ms).
- [x] Supports cross-validation splitting by speaker (e.g., Leave-One-Speaker-Out).

### US-002: Flexible Model Fine-Tuning
**Description:** As a machine learning engineer, I want to selectively fine-tune specific layers of WavLM and use a scalar mix, so that I can capture phonetic-level features without catastrophic forgetting.

**Acceptance Criteria:**
- [x] Configuration supports specifying layers to fine-tune (e.g., "4-8" for 0-indexed Transformer layers 4 to 8).
- [x] CNN feature extractor is frozen.
- [x] Learnable scalar mix is applied across all layers.
- [x] Supports switching between WavLM base, base-plus, and large via configuration.
- [x] Employs differential learning rates for the backbone and classification head.

### US-003: Memory-Efficient Training
**Description:** As a user with an 8GB GPU, I want to train the model locally or via Colab without Out-Of-Memory (OOM) errors.

**Acceptance Criteria:**
- [x] Training supports mixed precision (FP16/BF16).
- [x] Gradient checkpointing is supported and toggleable.
- [x] Gradient accumulation is supported for larger effective batch sizes.
- [x] TensorBoard logging includes training loss, validation F1, PR curves, and audio samples (TP/FP/FN/TN) for manual review.

### US-004: Delta Checkpointing
**Description:** As a researcher, I want to save only the fine-tuned weights, so that I can minimize storage usage.

**Acceptance Criteria:**
- [x] Checkpoint files contain only the weights of the fine-tuned layers, scalar mix, and classification head.
- [x] Delta checkpoints are under 50MB for the base-plus model.
- [x] Model can be successfully restored from a delta checkpoint for inference by downloading the base model from HF.

### US-005: Batch Inference and Output
**Description:** As a researcher, I want to run the model on a batch of files and export the results to a CSV, so that I can compare them with baseline methods.

**Acceptance Criteria:**
- [x] CLI script accepts audio directory, TextGrid directory, and checkpoint path.
- [x] Outputs a CSV with predictions and confidence scores aligned with the V2 pipeline format.
- [x] (Optional) Can export predictions as a new tier in the original TextGrid files.

## 4. Functional Requirements
- **FR-1:** The system must process both `.wav` and `.flac` audio formats using `torchaudio.load` and the `soundfile` backend.
- **FR-2:** The data loader must dynamically adjust `num_workers` based on the OS (0 for Windows, 4 for Linux/Colab) to prevent deadlocks.
- **FR-3:** The training loop must support Focal Loss and Weighted BCE for handling class imbalance.
- **FR-4:** The training pipeline must implement early stopping based on the validation F1 score.
- **FR-5:** The system must parse configuration from YAML files following the `{backbone}_{loss}_{split_strategy}.yaml` naming convention, using `_defaults.yaml` for fallback values.
- **FR-6:** The data pipeline must include a resampling mechanism (to 16kHz) and an optional context padding mechanism (e.g., ±50ms) around vowel segments.

## 5. Non-Goals
- Real-time or streaming creaky voice detection.
- Building a new standalone GUI application (WavFry will output CSVs that can be consumed by existing analysis pipelines or GUIs).
- Fine-tuning the CNN feature extractor of WavLM.
- Supporting audio sampling rates other than 16kHz without automatic resampling.

## 6. Design Considerations
- **Architecture:** WavLM Encoder -> Scalar Mix -> Mean Pooling -> Dropout -> Linear -> GELU -> Dropout -> Linear -> Sigmoid.
- **Configuration:** Use `pyyaml` for configuration management to keep parameters centralized, hierarchical, and easily interchangeable.
- **Project Structure:** Clear separation of data loading (`data/`), model definition (`model/`), training loop (`train/`), and inference scripts (`inference/`) within `06_wavfry/`.

## 7. Technical Considerations
- **Data Imbalance:** Creaky vowels are a minority class (estimated 10-30%). Metrics must prioritize F1 and AUC-PR over accuracy.
- **Cross-Validation:** With only 5 annotated speakers, Leave-One-Speaker-Out cross-validation is highly recommended to assess generalization properly.
- **Colab Integration:** Provide a ready-to-use Jupyter Notebook (`notebooks/colab_train.ipynb`) that handles Google Drive mounting, dependency installation, and checkpoint saving.

## 8. Success Metrics
- Achieve a validation F1 score higher than the standalone V2 pipeline baseline.
- Successfully train on an 8GB GPU (or Colab T4) without OOM errors.
- Delta checkpoint size does not exceed 100MB for base/base-plus models.

## 9. Open Questions
- What are the exact tier names and label formats used in the manual TextGrid annotations for the 5 speakers?
- Are the 5 annotated speakers exclusively English, Mandarin, or a mix?
- Should the validation speaker be fixed, or is Leave-One-Speaker-Out strictly required for the first release?
