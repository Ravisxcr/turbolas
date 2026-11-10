# Installation Guide

TurboLAS is distributed as pre-compiled binary wheels for major operating systems and architectures, as well as an editable source distribution.

---

## Prerequisites

- **Python**: Version `3.9` or higher (`3.9`, `3.10`, `3.11`, `3.12`, `3.13`).
- **Dependencies**: TurboLAS has **zero mandatory third-party runtime dependencies**. All parsing and serialization algorithms are compiled in pure Rust.

---

## 1. Installing via Pip

Once published on PyPI, install TurboLAS using standard pip:

```bash
pip install turbolas
```

Or using modern package managers like `uv`:

```bash
uv pip install turbolas
```

---

## 2. Building from Source

To build TurboLAS from source for maximum platform optimization:

### Step 1: Install Rust Toolchain

Make sure the stable Rust toolchain is installed (Rust 1.75+ recommended):

```bash
# On Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# On Windows
# Download and run rustup-init.exe from https://rustup.rs/
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/Ravisxcr/turbolas.git
cd turbolas
```

### Step 3: Build & Install via Maturin

We recommend using a Python virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS
source .venv/bin/activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install maturin build tool
pip install maturin

# Build and install optimized release wheel
maturin develop --release
```

---

## 3. Verifying Installation

Verify that TurboLAS is installed and functional:

```bash
python -c "import turbolas; print(turbolas.__doc__)"
```
