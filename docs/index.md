# TurboLAS Documentation

Welcome to the official documentation for **TurboLAS**, an ultra-high-performance, zero-dependency Python extension module for reading, processing, and writing CWLS LAS (Log ASCII Standard) well log data, implemented in pure Rust via PyO3.

---

## Navigation & Guides

- [**Installation**](installation.md): How to install pre-built wheels or build from source using Maturin.
- [**Quickstart Guide**](quickstart.md): Get up and running in under 5 minutes with practical examples.
- [**Reading LAS Files**](reading.md): Reading LAS 1.2, 2.0, and 3.0 files, encoding fallbacks, and validation modes.
- [**Writing & Serializing LAS Files**](writing.md): High-throughput buffered writing, version selection, wrapped data, and custom delimiters.
- [**DataFrame & Array Interoperability**](dataframe_interop.md): Zero-copy handoff to Polars, Pandas, NumPy, and pure Python datatypes.
- [**CWLS LAS 3.0 Support**](las3_support.md): Compound section headers, custom delimiters, and textual / categorical lithology curves.
- [**Permissive Parsing & Dirty Data Recovery**](permissive_parsing.md): Heuristics for gracefully repairing malformed legacy well logs.
- [**API Reference**](api_reference.md): Detailed specification of all modules, classes, functions, and parameters.
- [**Performance Benchmarks**](benchmarks.md): Comparative throughput and memory benchmarks against `lasio`.

---

## Key Highlights

- **Blazing Fast**: Up to **60x to 300x+ faster** than traditional Python LAS parsers (`lasio`). Single-pass SIMD-accelerated columnar tokenization and zero-copy byte parsing.
- **Zero Heavy Dependencies**: TurboLAS exports directly to native Python data types (`dict`, `list`, `memoryview`). No mandatory runtime dependencies on PyArrow, Pandas, or Polars.
- **Universal Interoperability**: Direct zero-copy bridge to **Polars**, **Pandas**, and **NumPy** (`np.frombuffer`).
- **Complete CWLS Standard Compliance**:
  - Full support for **CWLS LAS 1.2**, **2.0**, and **3.0**.
  - Compound sections (`~Log_Definition`, `~Log_Data`, `~Core_Definition`, `~Core_Data`, etc.).
  - Textual / categorical curves (e.g. Lithology, Stratigraphy).
  - Delimiter customization (`DLM . COMMA`, `SPACE`, `TAB`).
  - Wrapped record handling (`WRAP. YES`).
  - Robust character encoding detection (ASCII, UTF-8, Windows-1252 / CP-1252).
- **Industrial Permissive Heuristics**: Automatic recovery for legacy files with missing dots, missing colons, non-standard comments, and corrupted data tokens (`***`, `ERR`, `BAD`, `ND`).
