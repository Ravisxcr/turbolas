# API Reference

This document provides a comprehensive reference for all modules, classes, methods, and functions in TurboLAS.

---

## 1. Top-Level Functions

### `turbolas.read`

```python
def read(
    source: str | bytes | os.PathLike,
    *,
    mode: ValidationMode = ValidationMode.STRICT,
    encoding: str | None = None,
) -> LasWell:
```

Parses a CWLS LAS file into a [`LasWell`](#laswell) instance.

- **`source`**: Filepath (`str` or `PathLike`), raw LAS text (`str`), or file bytes (`bytes`).
- **`mode`**: Validation mode ([`ValidationMode.STRICT`](#validationmode) or [`ValidationMode.PERMISSIVE`](#validationmode)). Default is `STRICT`.
- **`encoding`**: Character encoding override. If `None`, detects UTF-8 with automatic CP-1252 fallback.
- **Returns**: A [`LasWell`](#laswell) object representing the parsed well.
- **Raises**: `ValueError` or `RuntimeError` if parsing fails in strict mode.

---

### `turbolas.write`

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

Serializes a [`LasWell`](#laswell) instance to a CWLS LAS file or formatted text string.

- **`well`**: The `LasWell` object to serialize.
- **`destination`**: Filepath to write to, or `None` to return the formatted string.
- **`version`**: Target LAS version (`1.2`, `2.0`, or `3.0`). Default is `2.0`.
- **`wrap`**: Whether to wrap records across multiple lines (`WRAP. YES`). Default is `False`.
- **`delimiter`**: Delimiter for LAS 3.0 data columns (`"SPACE"`, `"COMMA"`, `"TAB"`).
- **`float_precision`**: Number of decimal places to format float values. Default is `4`.
- **`null_value`**: Sentinel value to write when a float is `NaN`. Defaults to well's existing `NULL` or `-999.25`.
- **Returns**: `str` if `destination` is `None`, or `None` if written to disk.

---

## 2. Classes

### `ValidationMode`

Enum specifying validation and error recovery behavior.

```python
class ValidationMode(enum.Enum):
    STRICT = 0
    PERMISSIVE = 1
```

- **`STRICT`**: Enforces strict CWLS LAS standard compliance.
- **`PERMISSIVE`**: Enables robust heuristics to parse malformed or dirty legacy files.

---

### `LasWell`

Represents a parsed CWLS LAS well document.

#### Header Properties

| Property | Type | Description |
| :--- | :--- | :--- |
| `well_name` | `str | None` | The `WELL` name from the `~Well` section. |
| `company` | `str | None` | The `COMP` company name. |
| `field` | `str | None` | The `FLD` field name. |
| `location` | `str | None` | The `LOC` location string. |
| `service_company` | `str | None` | The `SRVC` service company name. |
| `date` | `str | None` | The `DATE` logging date string. |
| `uwi` | `str | None` | The `UWI` unique well identifier / API number. |
| `start_depth` | `float | None` | The `STRT` depth value. |
| `stop_depth` | `float | None` | The `STOP` depth value. |
| `step` | `float | None` | The `STEP` depth step increment. |
| `null_value` | `float | None` | The `NULL` sentinel value declared in the header. |
| `version` | `float` | The LAS specification version (e.g. `1.2`, `2.0`, `3.0`). |
| `is_las3` | `bool` | Whether the file is CWLS LAS 3.0. |
| `is_wrapped` | `bool` | Whether the data section is in wrapped mode (`WRAP. YES`). |
| `delimiter` | `str | None` | Delimiter string for LAS 3.0 (`"SPACE"`, `"COMMA"`, `"TAB"`). |

#### Curve & Data Properties

| Property | Type | Description |
| :--- | :--- | :--- |
| `num_rows` | `int` | Total number of depth rows in the primary data section. |
| `num_curves` | `int` | Total number of curves in the primary data section. |
| `curve_names` | `list[str]` | List of all primary curve mnemonics. |
| `curves` | `dict[str, LasCurve]` | Mapping of curve mnemonics to [`LasCurve`](#lascurve) objects. |
| `section_names` | `list[str]` | List of all available section names. |

#### Methods

- **`curve(key: str | int) -> LasCurve`**: Retrieves a curve by name or index.
- **`curve_memoryview(key: str | int) -> memoryview`**: Returns a zero-copy memoryview of the curve's IEEE 754 float64 buffer.
- **`to_dict() -> dict[str, list[float] | list[str]]`**: Exports all curves as a native dictionary suitable for Polars or Pandas.
- **`to_list() -> list[list[float]]`**: Exports numeric curves as a 2D Python list of columns.
- **`has_section(name: str) -> bool`**: Returns whether a specific section exists.
- **`section_curves(name: str) -> list[str]`**: Returns curve mnemonics defined in a secondary section.
- **`section_data(name: str) -> dict[str, list[float] | list[str]]`**: Exports secondary section data as a dictionary.
- **`write(destination=None, version=2.0, wrap=False, delimiter=None, float_precision=4, null_value=None) -> str | None`**: Serializes the well.

---

### `LasCurve`

Represents a single curve / column in a LAS well.

#### Properties

| Property | Type | Description |
| :--- | :--- | :--- |
| `mnemonic` | `str` | The curve mnemonic name (e.g. `"DEPT"`, `"GR"`). |
| `unit` | `str` | The engineering unit of measurement (e.g. `"M"`, `"GAPI"`). |
| `value` | `str` | The value field from the curve header line. |
| `description` | `str` | The descriptive text for the curve. |
| `is_string` | `bool` | `True` if this is a textual / categorical curve, `False` if numeric float. |
| `data` | `list[float]` | Float curve data as a Python list (raises error if curve is string). |
| `min` | `float` | Minimum non-null value of the curve (NaN if string). |
| `max` | `float` | Maximum non-null value of the curve (NaN if string). |
| `null_count` | `int` | Number of null / NaN values in the curve. |

#### Methods

- **`string_data() -> list[str]`**: Returns curve values as a list of strings if `is_string` is `True`.
- **`memoryview() -> memoryview`**: Returns a zero-copy buffer view of numeric float data.
- **`__len__() -> int`**: Number of samples in the curve.
- **`__getitem__(index: int) -> float | str`**: Returns the value at `index`.
