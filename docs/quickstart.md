# Quickstart Guide

This guide introduces the core workflows in TurboLAS: reading a well log file, accessing curve data, inspecting metadata, and writing files back to disk.

---

## 1. Reading a Well File

Use `turbolas.read()` to parse a LAS file from a local filesystem path or raw LAS text in memory:

```python
import turbolas

# Read from a file path
well = turbolas.read("path/to/well.las")

# Or read directly from a string or bytes
with open("path/to/well.las", "r", encoding="utf-8") as f:
    raw_content = f.read()
well = turbolas.read(raw_content)
```

---

## 2. Inspecting Well Metadata

TurboLAS exposes convenient properties for all standard CWLS header fields:

```python
# Well identification
print("Well Name:", well.well_name)
print("Company:", well.company)
print("Field:", well.field)
print("Location:", well.location)
print("UWI / API:", well.uwi)

# Depth and step intervals
print(f"Depth Interval: {well.start_depth} -> {well.stop_depth} (step: {well.step})")
print("Null Sentinel Value:", well.null_value)

# LAS specification details
print("Version:", well.version)
print("Is LAS 3.0:", well.is_las3)
print("Is Wrapped:", well.is_wrapped)
if well.is_las3:
    print("Delimiter:", well.delimiter)
```

---

## 3. Accessing Curve Data

Access curves by name, index, or iterate over the entire set:

```python
# List all available curve mnemonics
print("Available Curves:", well.curve_names)
print(f"Total Rows: {well.num_rows}, Total Curves: {well.num_curves}")

# Access a specific curve via dict-like indexing
gamma_ray = well["GR"]  # Returns list[float] or list[str]

# Or access rich LasCurve metadata objects
dept_curve = well.curve("DEPT")
print(f"Mnemonic: {dept_curve.mnemonic}")
print(f"Unit: {dept_curve.unit}")
print(f"Description: {dept_curve.description}")
print(f"Min: {dept_curve.min}, Max: {dept_curve.max}")
print(f"Null Count: {dept_curve.null_count}")

# Extract numerical data as a Python list
depth_values = dept_curve.data
```

---

## 4. Converting to DataFrames

TurboLAS exports directly to native Python structures without mandatory dependencies:

```python
# Direct conversion to Polars DataFrame
import polars as pl

df_pl = pl.DataFrame(well.to_dict())
print(df_pl.head())

# Direct conversion to Pandas DataFrame
import pandas as pd

df_pd = pd.DataFrame(well.to_dict())
print(df_pd.head())

# Zero-copy NumPy view of curve data
import numpy as np

dept_array = np.frombuffer(well.curve_memoryview("DEPT"), dtype=np.float64)
```

---

## 5. Serializing & Writing LAS Files

Write a `LasWell` object back to disk or serialize it into a string:

```python
# Export to a formatted string
las_text = well.write(version=2.0)

# Write directly to disk
turbolas.write(well, "exported_well.las", version=2.0, float_precision=4)
```
