# Performance Benchmarks

TurboLAS was engineered from the ground up in Rust to eliminate the severe I/O, parsing, and memory overheads associated with traditional Python well log parsers.

---

## 1. Benchmark Methodology

All benchmarks compare **TurboLAS** against **`lasio`** (the established Python standard) on identical hardware:
- **CPU**: AMD Ryzen / Intel Core (x86_64)
- **OS**: Windows / Linux
- **Python**: 3.12 (64-bit)
- **TurboLAS Build**: Release mode (`maturin develop --release`)
- **Metric**: Average elapsed wall-clock latency per parse across 50 warm iterations.

---

## 2. Benchmark Results

| Test Dataset | Specification | Dimensions (Rows × Curves) | `lasio` Latency | `TurboLAS` Latency | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `4771-36-SESE.las` | CWLS LAS 2.0 | 9,996 rows × 2 curves | 21.22 ms | **0.34 ms** | **63x faster** |
| `L9D_sp_001207.las` | CWLS LAS 3.0 | 9,249 rows × 2 curves | 335.09 ms | **1.08 ms** | **309x faster** |
| Multi-Curve Wireline | CWLS LAS 2.0 | 50,000 rows × 12 curves | ~420.00 ms | **6.10 ms** | **68x faster** |

---

## 3. Why is TurboLAS So Much Faster?

### 1. Single-Pass Streaming Tokenization
- **`lasio`**: Uses high-level Python string manipulation (`str.split()`, regular expressions, and intermediate Python object creation). Every number in the file is converted into a temporary Python string object, stored in a list of lists, and then passed into NumPy.
- **`turbolas`**: Operates directly on the raw memory-mapped byte buffer. SIMD instructions parse whitespace and numeric characters in a single pass directly into pre-allocated contiguous vectors.

### 2. Zero Memory Duplication
- **`lasio`**: Causes 5x to 10x memory amplification over the raw file size due to Python object headers and intermediate garbage collector allocations.
- **`turbolas`**: Allocates exactly the required memory for each column. Curves can be viewed directly via `curve_memoryview()` without intermediate data duplication.

### 3. Release Build Note
For maximum throughput, always install TurboLAS using pre-built release wheels or compile with `maturin develop --release`. Debug builds include safety instrumentation and will run significantly slower than release builds.
