# Table 9: Edge Runtime & Real-Time Factor (RTF) Benchmark

- **ONNX Model Size**: `362.14 MB`
- **PyTorch Full Model**: `364.49 MB`
- **Benchmark Slice**: `1.00s` (16 kHz mono)
- **CPU Provider Latency**: `44.50 ± 1.75 ms` (P95: `46.70 ms`)
- **Real-Time Factor (RTF)**: **`0.0445`** (Target `< 0.05`)

| Platform / Runtime | Latency (Mean ± Std) | P50 (ms) | P95 (ms) | RTF | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ONNX Runtime (CPU)** | 44.50 ± 1.75 ms | 44.96 | 46.70 | **0.0445** | 🟢 RTF < 0.05 Pass |
