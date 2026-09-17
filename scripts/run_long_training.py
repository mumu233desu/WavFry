import subprocess
import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    config_path = "config/_defaults.yaml"
    
    # 40 epochs parameter settings
    epochs = 40
    annotated_speakers = ["U1001", "U0046", "U2013", "U1089", "U2002", "U2004", "U1067", "U0037"]
    spk_str = "[" + ",".join(annotated_speakers) + "]"
    
    overrides = [
        f"training.epochs={epochs}",
        f"data.annotated_speakers={spk_str}",
        "training.lr_schedule=cosine",                 # Force standard cosine schedule (no restart)
        "eval.early_stopping_patience=0",              # Disable early stopping to run full 40 epochs
        "tensorboard.log_dir=R:/runs/long_training",
        "checkpoint.dir=R:/checkpoints/long_training"
    ]
    
    # Clean up pre-existing best_model to ensure fresh checkpoints
    best_path = Path("R:/checkpoints/long_training/best_model.pt")
    if best_path.exists():
        try: best_path.unlink()
        except: pass
        
    cmd = ["uv", "run", "python", "scripts/train.py", "--config", config_path, "--override"] + overrides
    
    print("="*60)
    print(f"Starting 40-Epoch Long Training Convergence Test on Fold U1001")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    try:
        # We run it synchronously or capture progress
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*60)
        print("Long training completed successfully!")
        print("="*60)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Training exited with code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
