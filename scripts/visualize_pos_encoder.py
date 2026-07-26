import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    checkpoint_path = Path("R:/checkpoints/full_ablation/full/best_model.pt")
    if not checkpoint_path.exists():
        checkpoint_path = Path("runs/full_ablation/full/best_model.pt") # fallback
    
    if not checkpoint_path.exists():
        # Let's search E drive
        checkpoint_path = Path(__file__).resolve().parents[1] / "runs" / "full_ablation" / "full" / "best_model.pt"
        
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = ckpt['model_state_dict']
    
    pos_scorer_key = "pos_scorer.weight"
    if pos_scorer_key not in state_dict:
        print(f"Error: {pos_scorer_key} not found in state_dict. Available keys: {list(state_dict.keys())}")
        return
        
    weight = state_dict[pos_scorer_key] # shape: [1, 32]
    weight = weight.squeeze(0).numpy() # shape: [32]
    
    # Compute relative positions 0.0 to 1.0 (100 steps)
    rel_pos = np.linspace(0.0, 1.0, 100)
    
    # Replicate Sinusoidal encoding
    d_model = len(weight)
    i = np.arange(d_model // 2)
    freq = 1.0 / (10000.0 ** (2.0 * i / d_model))
    
    pos_bias_list = []
    for p in rel_pos:
        angles = p * freq
        pos_emb = np.concatenate([np.sin(angles), np.cos(angles)])
        bias = np.dot(weight, pos_emb)
        pos_bias_list.append(bias)
        
    # Plotting with professional styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(9, 5))
    plt.plot(rel_pos * 100, pos_bias_list, label="Attention Bias Contribution", color="#1f77b4", linewidth=3.0)
    plt.axhline(0, color="#d3d3d3", linestyle="--", linewidth=1.5)
    
    # Highlight the peaks / trends
    min_bias_idx = np.argmin(pos_bias_list)
    max_bias_idx = np.argmax(pos_bias_list)
    plt.scatter([rel_pos[min_bias_idx]*100], [pos_bias_list[min_bias_idx]], color="red", zorder=5)
    plt.scatter([rel_pos[max_bias_idx]*100], [pos_bias_list[max_bias_idx]], color="green", zorder=5)
    
    plt.text(rel_pos[min_bias_idx]*100, pos_bias_list[min_bias_idx] - 0.05, f"Min: {pos_bias_list[min_bias_idx]:.3f}", 
             ha='center', va='top', color="red", fontweight='semibold')
    plt.text(rel_pos[max_bias_idx]*100, pos_bias_list[max_bias_idx] + 0.05, f"Max: {pos_bias_list[max_bias_idx]:.3f}", 
             ha='center', va='bottom', color="green", fontweight='semibold')
    
    plt.title("Learned Attention Position Bias vs. Relative Position in Vowel Segment", fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("Relative Temporal Position within Vowel (%)", fontsize=12)
    plt.ylabel("Attention Score Bias (Additive)", fontsize=12)
    plt.xlim(-2, 102)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    # Save path
    save_dir = Path("runs/full_ablation")
    save_dir.mkdir(parents=True, exist_ok=True)
    img_name = "positional_bias_curve.png"
    img_path = save_dir / img_name
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved plot to {img_path}")
    
    # Also save to artifacts directory for markdown rendering
    artifact_dir = Path("C:/Users/mumu/.gemini/antigravity/brain/459c349c-51ab-4f3a-8d7c-57b96fe1e0a4")
    if artifact_dir.exists():
        artifact_path = artifact_dir / img_name
        import shutil
        shutil.copy(img_path, artifact_path)
        print(f"Copied plot to artifact directory: {artifact_path}")

if __name__ == "__main__":
    main()
