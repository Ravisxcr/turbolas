# Writing & Serializing LAS Files

TurboLAS provides an ultra-fast, Rust-backed LAS writer capable of serializing `LasWell` objects into standard-compliant CWLS LAS 1.2, 2.0, and 3.0 files or strings.

---

## 1. Top-Level `write()` Function

The `turbolas.write()` function serializes a well to a file or returns the formatted string:

```python
import turbolas

well = turbolas.read("path/to/source.las")

# Write directly to disk
turbolas.write(well, "exported_well.las", version=2.0)

# Return formatted LAS text as a Python string
las_string = turbolas.write(well, version=2.0)
```

Alternatively, invoke `well.write()` directly:

```python
# Write to file
well.write("output.las", version=2.0)

# Return string
content = well.write(version=2.0)
```

---

## 2. Function Signature & Parameters

```python
def write(
    well: LasWell,
    destination: str | os.PathLike | None = None,
    *,
    version: float = 2.0,
    wrap: bool = False,
    delimiter: str | None = None,
    float_precision: int = 4,
    null_value: float | None = None,
) -> str | None:
```

### Parameters

- **`well`**: The `LasWell` instance to serialize.
- **`destination`**: File path to write to, or `None` to return the serialized LAS content as a `str`.
- **`version`**: Target CWLS LAS version:
  - `1.2`: Standard LAS 1.2 format with `~V`, `~W`, `~C`, `~A`.
  - `2.0` (default): Standard LAS 2.0 format with full section descriptions.
  - `3.0`: CWLS LAS 3.0 compound format (`~Log_Definition`, `~Log_Data`).
- **`wrap`**: Whether data lines should be wrapped (`WRAP. YES`). Default is `False`.
- **`delimiter`**: Delimiter for LAS 3.0 data columns (`"SPACE"`, `"COMMA"`, `"TAB"`).
- **`float_precision`**: Number of decimal places to format floating-point curve values. Default is `4`.
- **`null_value`**: The null sentinel value to write when a float value is `NaN`. Defaults to the well's existing `NULL` value or `-999.25`.

---

## 3. Serialization Examples

### Example 1: Exporting with Custom Precision and Null Value

```python
turbolas.write(
    well,
    "high_precision.las",
    version=2.0,
    float_precision=6,
    null_value=-9999.0,
)
```

### Example 2: Exporting as CWLS LAS 3.0 with Comma Delimiter

```python
turbolas.write(
    well,
    "well_las3.las",
    version=3.0,
    delimiter="COMMA",
    float_precision=4,
)
```

The resulting file uses standard LAS 3.0 headers and comma separators:

```text
~Version Information
 VERS.                        3.0 : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.                        NO   : ONE LINE PER DEPTH STEP
 DLM .                      COMMA : DELIMITING CHARACTER BETWEEN DATA COLUMNS

~Well Information
...
~Log_Data | Log_Definition
1200.0000,"FINE SANDSTONE",0.1800,45.2000
1200.5000,SHALE,0.0500,52.1000
```

### Example 3: Writing Wrapped Mode (LAS 1.2 / 2.0)

When targeting legacy systems that require wrapped lines (`WRAP. YES`), set `wrap=True`:

```python
turbolas.write(well, "wrapped.las", version=2.0, wrap=True)
```
In wrapped mode, row 1 contains the depth step and subsequent curves wrap neatly across 4 columns per line.
