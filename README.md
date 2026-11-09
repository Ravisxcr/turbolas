# TurboLAS

[![CI](https://github.com/Ravisxcr/turbolas/actions/workflows/ci.yml/badge.svg)](https://github.com/Ravisxcr/turbolas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Rust: 1.75+](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

**TurboLAS** is an ultra-high-performance, zero-dependency Python extension module for reading and processing CWLS LAS (Log ASCII Standard) well log data (versions 1.2, 2.0, and 3.0), implemented in pure Rust via PyO3.

---

## Key Highlights

- **Blazing Fast**: **60x to 300x+ faster** than traditional Python LAS parsers (e.g., `lasio`). Single-pass SIMD-accelerated columnar tokenization and zero-copy byte parsing.
- **Zero Heavy Dependencies**: TurboLAS exports directly to native Python data types (`dict`, `list`, `memoryview`). No mandatory dependencies on PyArrow, Pandas, or Polars.
- **Universal Interoperability**: Ingest data into any downstream DataFrame or array library with zero friction:
  - **Polars**: `pl.DataFrame(well.to_dict())`
  - **Pandas**: `pd.DataFrame(well.to_dict())`
  - **NumPy**: `np.frombuffer(well.curve_memoryview("DEPT"), dtype=np.float64)`
- **Complete CWLS Specification Compliance**:
  - Full support for **CWLS LAS 1.2**, **2.0**, and **3.0**.
  - Handles compound 3.0 sections (`~Log_Definition`, `~Log_Data`, `~Core_Definition`, `~Core_Data`, etc.).
  - Automatic delimiter detection (`DLM . COMMA`, `SPACE`, `TAB`).
  - Supports wrapped records (`WRAP. YES`), CRLF/LF line endings, and character encodings (ASCII, UTF-8, Windows-1252 / CP-1252).
  - Configurable strict and permissive parsing modes with automatic null sentinel detection (`-999.25`, etc.).

---

## Performance Benchmark

Benchmarked against `lasio` on real-world petrophysical well files:

| Dataset / File | Rows × Curves | `lasio` Latency | `TurboLAS` Latency | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| `4771-36-SESE.las` (LAS 2.0) | 9,996 × 2 | 21.22 ms | **0.34 ms** | **63x faster** |
| `L9D_sp_001207.las` (LAS 3.0) | 9,249 × 2 | 335.09 ms | **1.08 ms** | **309x faster** |

> *Note: For optimal parsing throughput, build in release mode via `maturin develop --release`.*

---

## Installation & Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ravisxcr/turbolas.git
cd turbolas

# Build & install in release mode
maturin develop --release
```

### Basic Usage

```python
import turbolas

# Read a LAS file (LAS 1.2, 2.0, or 3.0)
well = turbolas.read("path/to/well.las")

# Inspect well metadata
print(f"Well: {well.well}")
print(f"Company: {well.company}")
print(f"Start / Stop Depth: {well.start} -> {well.stop} (step: {well.step})")
print(f"Curves: {well.curve_mnemonics}")

# Access curve data
dept_curve = well.curves["DEPT"]
print(f"Unit: {dept_curve.unit}, Description: {dept_curve.descr}")
print(f"Min: {dept_curve.min}, Max: {dept_curve.max}, Null count: {dept_curve.null_count}")
```

---

## Interoperability (Zero Dependencies)

TurboLAS produces clean, native Python datatypes so you can load them directly into your favorite analytics framework:

### 1. Polars

```python
import polars as pl
import turbolas

well = turbolas.read("well.las")

# Ingest via native dictionary (preserves exact column ordering)
df = pl.DataFrame(well.to_dict())
print(df)
```

### 2. Pandas

```python
import pandas as pd
import turbolas

well = turbolas.read("well.las")

df = pd.DataFrame(well.to_dict())
print(df.head())
```

### 3. NumPy / Zero-Copy Memoryview

```python
import numpy as np
import turbolas

well = turbolas.read("well.las")

# Direct zero-copy buffer view of IEEE 754 float64 curve data
dept_array = np.frombuffer(well.curve_memoryview("DEPT"), dtype=np.float64)
```

### 4. Pure Python (No Libraries Needed)

```python
import turbolas

well = turbolas.read("well.las")

# Native Python dictionary: dict[str, list[float] | list[str]]
data_dict = well.to_dict()

# Native 2D Python list: list[list[float]] (numeric columns)
data_columns = well.to_list()
```

---

## Fast Rust-Backed LAS Writer

TurboLAS provides an ultra-fast buffered writer (`BufWriter`) capable of exporting wells to standard CWLS LAS 1.2, 2.0, or 3.0 text files or strings:

```python
import turbolas

well = turbolas.read("path/to/input.las")

# Serialize directly to a formatted string
las_str = well.write(version=2.0)

# Or write to a file path with custom formatting options
turbolas.write(
    well,
    "path/to/output.las",
    version=2.0,
    wrap=False,  # Wrap data lines (WRAP. YES)
    float_precision=4,  # Decimal places for float curves
    null_value=-999.25,  # Custom null sentinel value
)

# Export as CWLS LAS 3.0 with custom delimiter
turbolas.write(well, "output_las3.las", version=3.0, delimiter="COMMA")
```

---

## CWLS LAS 3.0 Support & Categorical Curves

CWLS LAS 3.0 introduces compound sections, custom delimiters, multiple data sets, and textual curves (e.g. Lithology, Stratigraphy, Core descriptions):

```python
import polars as pl
import turbolas

well = turbolas.read("las3_file.las")

if well.is_las3:
    print(f"LAS 3.0 Delimiter: {well.delimiter}")
    print(f"Available sections: {well.section_names}")

    # Textual / categorical curve support
    lith_curve = well.curves["LITH"]
    if lith_curve.is_string:
        descriptions = lith_curve.string_data()  # list[str]
        print(descriptions[:5])  # ['SANDSTONE', 'SHALE', 'LIMESTONE', ...]

    # Direct ingestion into Polars preserves pl.String type
    df = pl.DataFrame(well.to_dict())
    print(df.schema)  # {'DEPT': Float64, 'LITH': String, 'GR': Float64}

    # Access secondary sections (e.g., Core data)
    if well.has_section("CORE"):
        core_curves = well.section_curves("CORE")
        core_data = well.section_data("CORE")  # dict[str, list[float] | list[str]]
        print(f"Core section curves: {core_curves}")
```

---

## Resilient Permissive Parsing (Dirty Header Recovery)

Real-world historical well log files often violate official CWLS standards. TurboLAS includes robust permissive heuristics that gracefully recover from common corruptions:

```python
from turbolas import ValidationMode
import turbolas

# Read malformed or non-compliant legacy files
well = turbolas.read("corrupt_legacy.las", mode=ValidationMode.PERMISSIVE)
```

**Permissive recovery features**:
- **Missing Dot in Mnemonics**: Handles lines like `WELL ACME #1 : Well Name`.
- **Missing Colons**: Gracefully parses lines omitting the `:` description separator.
- **Corrupt & Asterisk Data Tokens**: Automatically converts `***`, `****`, `ERR`, `BAD`, `ND`, `#N/A`, `-`, `?` into `NaN`.
- **Non-Standard Comments & Delimiters**: Accepts `//`, `--`, `!` comments and spaces after `~` (e.g. `~ WELL INFORMATION`).
- **Duplicate Mnemonics**: Resolves duplicate curve names using indexed suffixes (`GR`, `GR_2`).
- **Header-less `~A` Data**: Extracts curve definitions embedded directly on the `~A` line.

---

## Development & Quality Assurance

```bash
# Rust tests (94 unit & integration tests)
cargo test --target-dir target

# Rust linter & formatter
cargo clippy --target-dir target --all-targets
cargo fmt --all --check

# Python tests (201 pytest tests)
pytest

# Python linter & type checks
ruff check .
ruff format --check .
mypy python/turbolas tests
```

---

## License

This project is licensed under the [MIT License](LICENSE).
