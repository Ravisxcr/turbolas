# TurboLAS

[![CI](https://github.com/Ravisxcr/turbolas/actions/workflows/ci.yml/badge.svg)](https://github.com/Ravisxcr/turbolas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Rust: 1.75+](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

**TurboLAS** is a blazing-fast Python extension module for LAS (Log ASCII Standard) well log data, powered by Rust and PyO3.

---

## Project Structure

A clean, single-crate Python extension layout powered by [Maturin](https://www.maturin.rs/):

```
turbolas/
├── Cargo.toml               # Rust package manifest (cdylib)
├── pyproject.toml           # PEP 621 packaging & Maturin configuration
├── rustfmt.toml             # Rust formatting configuration
├── src/
│   └── lib.rs               # Rust extension module implementation (PyO3)
├── python/
│   └── turbolas/
│       ├── __init__.py      # Python package entry & version export
│       ├── py.typed         # PEP 561 inline types marker
│       └── _turbolas.pyi    # Type stubs for the native Rust extension
├── tests/
    └── test_basic.py        # Python test suite
```

---

## Development Setup

### Prerequisites

- **Rust toolchain** (1.75+): [rustup.rs](https://rustup.rs/)
- **Python** (3.9+)
- **Maturin**: Build tool for Rust-based Python extensions

### Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ravisxcr/turbolas.git
   cd turbolas
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install development tools:**
   ```bash
   pip install maturin ruff pytest mypy
   ```

4. **Build and install in editable mode:**
   ```bash
   maturin develop
   ```

---

## Testing & Quality Assurance

```bash
# Rust formatting
cargo fmt --all --check

# Rust linter
cargo clippy --all-targets

# Python lint and formatting
ruff check .
ruff format --check .

# Run tests
pytest
```

---

## License

This project is licensed under the [MIT License](LICENSE).
