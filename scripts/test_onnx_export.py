import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Add 06_wavfry to path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from wavfry.model import wavlm_classifier as model_mod


def parse_args():
    parser = argparse.ArgumentParser(description="Test and benchmark exported ONNX and full PyTorch models.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="deployments",
        help="Directory containing exported wavfry_full.pt and wavfry_model.onnx",
    )
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations for RTF.")
    parser.add_argument("--duration-sec", type=float, default=1.0, help="Audio duration in seconds for benchmark.")
    return parser.parse_args()


def test_export():
    args = parse_args()
    model_dir = Path(args.model_dir) if Path(args.model_dir).is_absolute() else (root / args.model_dir)
    if not model_dir.exists():
        fallback = root / "exp_results" / "checkpoints"
        if fallback.exists():
            model_dir = fallback

    pt_path = model_dir / "wavfry_full.pt"
    onnx_path = model_dir / "wavfry_model.onnx"

    if not pt_path.exists():
        raise FileNotFoundError(f"PyTorch full model not found: {pt_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print(f"1. Loading PyTorch model from {pt_path}...")
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    pt_model = model_mod.WavFryClassifier(config)
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()

    print(f"2. Loading ONNX model from {onnx_path}...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output_names = [o.name for o in sess.get_outputs()]
    print(f"   ONNX Outputs detected: {onnx_output_names}")

    # 3. Create dummy input
    num_samples = int(16000 * args.duration_sec)
    np.random.seed(42)
    dummy_audio_np = np.random.uniform(-1.0, 1.0, size=(1, num_samples)).astype(np.float32)
    pt_audio = torch.from_numpy(dummy_audio_np)
    pt_lengths = torch.tensor([num_samples], dtype=torch.long)

    print(f"\n3. Verifying mathematical consistency (1 batch, {args.duration_sec:.2f}s audio)...")
    with torch.no_grad():
        pt_out = pt_model(pt_audio, pt_lengths)

    inputs = {"waveform": dummy_audio_np, "lengths": np.array([num_samples], dtype=np.int64)}
    onnx_out_raw = sess.run(None, inputs)
    onnx_out_dict = {name: val for name, val in zip(onnx_output_names, onnx_out_raw)}

    # Verify matching tasks
    verified_tasks = []
    max_diffs = {}
    mses = {}

    tasks = getattr(pt_model, "tasks", ["creaky"])
    for t in tasks:
        target_name = f"{t}_logits" if f"{t}_logits" in onnx_out_dict else "logits"
        if target_name in onnx_out_dict:
            pt_val = pt_out[t].numpy() if isinstance(pt_out, dict) else pt_out.numpy()
            onnx_val = onnx_out_dict[target_name]
            pt_p = 1.0 / (1.0 + np.exp(-pt_val))
            onnx_p = 1.0 / (1.0 + np.exp(-onnx_val))
            diff = float(np.max(np.abs(pt_p - onnx_p)))
            mse = float(np.mean((pt_p - onnx_p) ** 2))
            max_diffs[t] = diff
            mses[t] = mse
            verified_tasks.append(t)
            print(f"   Task [{t}]: MSE = {mse:.2e}, Max Absolute Diff = {diff:.2e}")

    # 4. Latency & RTF Benchmark
    print(f"\n4. Benchmarking Edge Inference Latency across {args.iterations} iterations on CPU...")
    # Warmup
    for _ in range(5):
        sess.run(None, inputs)

    latencies_ms = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_lat_ms = float(np.mean(latencies_ms))
    std_lat_ms = float(np.std(latencies_ms))
    p50_lat_ms = float(np.percentile(latencies_ms, 50))
    p95_lat_ms = float(np.percentile(latencies_ms, 95))
    rtf = (mean_lat_ms / 1000.0) / args.duration_sec

    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    pt_size_mb = pt_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("      Table 9: Edge Deployment & RTF Benchmark Summary")
    print("=" * 65)
    print(f"  PyTorch Full Model Size : {pt_size_mb:.2f} MB")
    print(f"  ONNX Exported Model Size: {onnx_size_mb:.2f} MB")
    print(f"  Audio Slice Duration    : {args.duration_sec:.2f} s")
    print(f"  Mean Latency (CPU)      : {mean_lat_ms:.2f} ± {std_lat_ms:.2f} ms")
    print(f"  P50 / P95 Latency       : {p50_lat_ms:.2f} ms / {p95_lat_ms:.2f} ms")
    print(f"  Real-Time Factor (RTF)  : {rtf:.4f} (Threshold: < 0.05)")
    print("=" * 65)

    benchmark_summary = {
        "pytorch_model_size_mb": round(pt_size_mb, 2),
        "onnx_model_size_mb": round(onnx_size_mb, 2),
        "audio_duration_sec": args.duration_sec,
        "iterations": args.iterations,
        "mean_latency_ms": round(mean_lat_ms, 2),
        "std_latency_ms": round(std_lat_ms, 2),
        "p50_latency_ms": round(p50_lat_ms, 2),
        "p95_latency_ms": round(p95_lat_ms, 2),
        "rtf": round(rtf, 5),
        "rtf_status": "Ultra-Fast (RTF < 0.05)" if rtf < 0.05 else "Standard",
        "mathematical_fidelity": {
            "verified_tasks": verified_tasks,
            "max_absolute_diff": max_diffs,
            "mse": mses,
        },
    }

    # Save to deployments
    json_path = model_dir / "table9_edge_runtime_benchmark.json"
    md_path = model_dir / "table9_edge_runtime_benchmark.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 9: Edge Runtime & Real-Time Factor (RTF) Benchmark\n\n")
        f.write(f"- **ONNX Model Size**: `{onnx_size_mb:.2f} MB`\n")
        f.write(f"- **PyTorch Full Model**: `{pt_size_mb:.2f} MB`\n")
        f.write(f"- **Benchmark Slice**: `{args.duration_sec:.2f}s` (16 kHz mono)\n")
        f.write(f"- **CPU Provider Latency**: `{mean_lat_ms:.2f} ± {std_lat_ms:.2f} ms` (P95: `{p95_lat_ms:.2f} ms`)\n")
        f.write(f"- **Real-Time Factor (RTF)**: **`{rtf:.4f}`** (Target `< 0.05`)\n\n")
        f.write("| Platform / Runtime | Latency (Mean ± Std) | P50 (ms) | P95 (ms) | RTF | Status |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **ONNX Runtime (CPU)** | {mean_lat_ms:.2f} ± {std_lat_ms:.2f} ms | {p50_lat_ms:.2f} | {p95_lat_ms:.2f} | **{rtf:.4f}** | {'🟢 RTF < 0.05 Pass' if rtf < 0.05 else '🟡 Standard'} |\n")

    print(f"\n[Saved] Benchmark JSON saved to: {json_path}")
    print(f"[Saved] Table 9 Markdown saved to: {md_path}")

    all_mse_pass = all(m < 1e-4 for m in mses.values())
    if all_mse_pass:
        print("\n[SUCCESS] ONNX export verified! Mathematical parity with PyTorch confirmed.")
    else:
        print("\n[WARNING] Mathematical parity difference exceeded 1e-4.")


if __name__ == "__main__":
    test_export()
