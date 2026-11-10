# Reading LAS Files

TurboLAS provides an ultra-fast streaming parser implemented in Rust that reads CWLS LAS 1.2, 2.0, and 3.0 files.

---

## 1. The `read()` Function

The primary entry point is `turbolas.read()`:

```python
import turbolas
from turbolas import ValidationMode

# Basic read
well = turbolas.read("data/well.las")

# Read with permissive mode for dirty legacy files
well = turbolas.read("data/corrupt_well.las", mode=ValidationMode.PERMISSIVE)

# Read directly from in-memory string or bytes
well = turbolas.read(las_content_str)
well = turbolas.read(las_content_bytes)
```

### Signature

```python
def read(
    source: str | bytes | os.PathLike,
    *,
    mode: ValidationMode = ValidationMode.STRICT,
    encoding: str | None = None,
) -> LasWell:
```

### Arguments

- **`source`**: A file path (`str` or `PathLike`), a LAS text string (`str`), or raw LAS file bytes (`bytes`).
- **`mode`**: Validation mode (`ValidationMode.STRICT` or `ValidationMode.PERMISSIVE`). Default is `STRICT`.
- **`encoding`**: Character encoding override (e.g. `"utf-8"`, `"windows-1252"`, `"latin1"`). If `None`, TurboLAS automatically detects UTF-8 with automatic fallback to Windows-1252 (CP-1252).

---

## 2. Validation Modes

TurboLAS supports two validation modes:

| Mode | Enum Value | Behavior |
| :--- | :--- | :--- |
| **Strict** | `ValidationMode.STRICT` | Enforces official CWLS specifications strictly. Rejects files with missing mandatory sections, missing dots in mnemonics, duplicate curve names, or ragged data rows. |
| **Permissive** | `ValidationMode.PERMISSIVE` | Resilient error recovery designed for dirty historical field logs. Forgives missing dots in mnemonics, missing colons, non-standard section headers, unknown sections, and corrupted data tokens. |

### Example: Handling Legacy Files

```python
from turbolas import ValidationMode
import turbolas

# If a file has syntax errors, strict mode raises ValueError:
try:
    well = turbolas.read("legacy_dirty.las", mode=ValidationMode.STRICT)
except ValueError as e:
    print(f"Strict validation rejected file: {e}")

# Permissive mode repairs and parses it gracefully:
well = turbolas.read("legacy_dirty.las", mode=ValidationMode.PERMISSIVE)
print(f"Parsed {well.num_curves} curves across {well.num_rows} rows.")
```

---

## 3. Character Encoding Detection

Well logs often contain non-ASCII characters such as degree symbols (`°C`, `°F`), euro symbols (`€`), or copyright marks (`©`).

TurboLAS incorporates zero-overhead encoding handling:
1. Fast-paths pure ASCII and valid UTF-8 without intermediate copying.
2. Gracefully detects Windows-1252 / CP-1252 and decodes non-UTF8 byte sequences deterministically.
3. Automatically strips UTF-8 Byte Order Marks (BOM `\xef\xbb\xbf`).

---

## 4. Null Value Sentinel Handling

CWLS files define a null value sentinel (typically `-999.25` or `-9999`) in the `~Well` section under the `NULL` mnemonic.

TurboLAS automatically:
- Parses the well's declared null sentinel value.
- Replaces null sentinel tokens in data columns with standard IEEE 754 `NaN` (`float('nan')`).
- Maps common corrupt tokens (`***`, `****`, `ERR`, `BAD`, `ND`, `#N/A`, `-`, `?`) to `NaN` in permissive mode.
- Computes `null_count` per curve in $O(1)$ time.

```python
well = turbolas.read("data/sample.las")
gr_curve = well.curve("GR")

print("Null value sentinel:", well.null_value)
print("NaN count in GR:", gr_curve.null_count)
```
