# DataFrame & Array Interoperability

TurboLAS was engineered with a **zero-dependency philosophy**: it exports data directly to native Python data types (`dict`, `list`, `memoryview`), removing the need for mandatory third-party runtimes like PyArrow, Polars, or Pandas.

Downstream analytics libraries can ingest TurboLAS data with zero friction.

---

## 1. Polars Integration

Polars seamlessly constructs a `pl.DataFrame` from the native dictionary exported by `well.to_dict()`:

```python
import polars as pl
import turbolas

well = turbolas.read("data/well.las")

# Ingest into Polars DataFrame
df = pl.DataFrame(well.to_dict())

print(df)
# Output:
# shape: (10000, 4)
# ┌─────────┬──────────┬──────────┬──────────┐
# │ DEPT    ┆ GR       ┆ NPHI     ┆ RHOB     │
# │ ---     ┆ ---      ┆ ---      ┆ ---      │
# │ f64     ┆ f64      ┆ f64      ┆ f64      │
# ╞═════════╪══════════╪══════════╪══════════╡
# │ 1000.0  ┆ 45.2     ┆ 0.15     ┆ 2.45     │
# │ 1000.5  ┆ 52.1     ┆ NaN      ┆ 2.48     │
# └─────────┴──────────┴──────────┴──────────┘
```

### Textual & Categorical Columns in Polars

If the well contains string curves (such as LAS 3.0 lithology or facies descriptions), Polars automatically assigns the `pl.String` dtype:

```python
df = pl.DataFrame(well.to_dict())
print(df.schema)
# {'DEPT': Float64, 'FACIES': String, 'GR': Float64}
```

---

## 2. Pandas Integration

Pandas constructs a `pd.DataFrame` directly from `well.to_dict()`:

```python
import pandas as pd
import turbolas

well = turbolas.read("data/well.las")

df = pd.DataFrame(well.to_dict())
print(df.head())
```

---

## 3. NumPy Zero-Copy Memoryview

For petrophysical algorithms, neural networks, or signal filtering that require NumPy arrays, `well.curve_memoryview()` provides a **zero-copy view** into the underlying Rust float64 memory buffer:

```python
import numpy as np
import turbolas

well = turbolas.read("data/well.las")

# Direct zero-copy memory view
mv = well.curve_memoryview("DEPT")

# Zero-copy NumPy array creation (O(1) time and memory overhead)
dept_np = np.frombuffer(mv, dtype=np.float64)

print("NumPy Array Shape:", dept_np.shape)
print("Mean Depth:", np.nanmean(dept_np))
```

> **Performance Note**: `np.frombuffer()` on a memoryview does not copy bytes. It creates a NumPy array header referencing the contiguous memory allocated in Rust.

---

## 4. Pure Python Export (No Libraries Required)

When deploying to lightweight serverless environments (AWS Lambda, GCP Cloud Run) or embedded devices without NumPy or Pandas:

```python
well = turbolas.read("data/well.las")

# Dictionary mapping curve names to Python lists
# dict[str, list[float] | list[str]]
data_dict = well.to_dict()

# 2D list of numeric columns
# list[list[float]]
data_columns = well.to_list()
```
