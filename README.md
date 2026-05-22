# WavFry: WavLM-based Creaky Voice Detection

WavFry is a sub-project for high-precision creaky voice detection in the CS-Dialogue corpus using fine-tuned WavLM models.

## Quick Start

1. **Build Manifest**:
```bash
uv run python scripts/build_manifest.py
```
2. **Train**:
```bash
# Standard training (using config's val_speakers)
uv run python scripts/train.py --config config/_defaults.yaml --resume

# Leave-One-Speaker-Out Cross Validation
# (Set 'val_speakers: loocv' in config, or use --loocv flag)
# uv run python scripts/train.py --override data.val_speakers=loocv
```
3. **Predict**:
```bash
uv run python scripts/predict.py --checkpoint checkpoints/best_model.pt --audio-dir <path> --textgrid-dir <path> --output-csv results.csv
```
4. **Archive Results**:
To package your experiment (runs, reports, and checkpoints) for backup:
```bash
uv run python scripts/package_experiment.py
```
This will create a `wavfry_exp_YYYYMMDD_{N}spks.zip` in the project root.

5. **Export to TextGrid** (Optional):
```bash
uv run python scripts/export_textgrid.py --csv results.csv --tg-dir <path> --output-dir <path>
```

## Monitoring

Visualize training progress, confusion matrices, and audio samples:
```bash
# Outputs (runs/ and checkpoints/) are kept inside the project root
uv run tensorboard --logdir runs/
```

## Key Features
- **WavLM Fine-tuning**: Selective layer fine-tuning (Layer 4-8) with Scalar Mix.
- **Efficiency**: Delta checkpoints (~40MB), Mixed Precision (FP16), Resampler caching.
- **Robustness**: Focal Loss for class imbalance, speaker-based data splitting.
- **Observability**: TensorBoard integration with loss, metrics, and error analysis.
- **Modular Design**: Proper Python package structure (`wavfry/`) with clean imports.

## Structure
- `wavfry/`: Core package containing data loaders, model definitions, and trainer logic.
- `scripts/`: Ready-to-use entry point scripts for the pipeline.
- `config/`: YAML configurations.
- `manifests/`: Generated segment CSVs for training.
- `checkpoints/`: Model weights (Delta or Full).
- `runs/`: TensorBoard logs.
