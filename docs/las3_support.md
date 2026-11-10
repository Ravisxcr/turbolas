# CWLS LAS 3.0 Support & Categorical Curves

CWLS LAS 3.0 was designed to expand the LAS standard beyond 1-dimensional numeric wireline data. It introduced:
- **Compound Sections**: Associating definition sections with corresponding data sections (e.g. `~Log_Data | Log_Definition`, `~Core_Data | Core_Definition`).
- **Custom Delimiters**: Supporting comma (`COMMA`), space (`SPACE`), or tab (`TAB`) separated values.
- **Heterogeneous & Textual Data**: Columns containing string descriptions (Lithology, Facies, Stratigraphy) alongside numeric measurements.

TurboLAS provides native, full-featured support for the complete CWLS 3.0 specification.

---

## 1. Reading LAS 3.0 Files

When TurboLAS encounters a LAS 3.0 file, it automatically parses the version, delimiter, and compound section relationships:

```python
import turbolas

well = turbolas.read("las3_canonical.las")

print("Is LAS 3.0:", well.is_las3)
print("Version:", well.version)  # 3.0
print("Delimiter:", well.delimiter)  # 'COMMA', 'SPACE', or 'TAB'
print("Available Sections:", well.section_names)
```

---

## 2. Textual & Categorical Curves

In LAS 3.0, curves are not restricted to numbers. Curves can contain text, categorical codes, and quoted strings (e.g. `"CHANNEL SAND"`, `"SHALE"`).

### Checking if a Curve is String

```python
lith_curve = well.curve("LITHOLOGY")

if lith_curve.is_string:
    print("Curve type: String / Categorical")
    # Extract string values as a Python list
    descriptions = lith_curve.string_data()
    print("Top 5 intervals:", descriptions[:5])
else:
    print("Curve type: Floating-Point Numeric")
    numeric_data = lith_curve.data
```

### Accessing via Dict Syntax

Accessing a string curve directly through `well[name]` returns the native list of strings:

```python
facies = well["FACIES"]
print(facies)  # ['SANDSTONE', 'SHALE', 'LIMESTONE', 'DOLOMITE']
```

### Automatic String Column Detection

TurboLAS detects string columns automatically:
- When a column is defined in a section with non-numeric data.
- When quoted tokens (`"..."` or `'...'`) are encountered.
- When non-numeric tokens (that are not standard null sentinels) appear in the column data.

---

## 3. Multi-Section Access

LAS 3.0 wells can contain multiple independent data blocks (e.g., Wireline Log Data and Core Data):

```python
well = turbolas.read("multi_section.las")

# Check for presence of secondary section
if well.has_section("CORE"):
    core_curves = well.section_curves("CORE")
    print("Core curves:", core_curves)

    # Extract core section as a dictionary of columns
    core_data = well.section_data("CORE")
    print("Core data columns:", list(core_data.keys()))
```

---

## 4. Roundtrip LAS 3.0 Serialization

TurboLAS can write LAS 3.0 files with any supported delimiter:

```python
# Write as LAS 3.0 with COMMA delimiter
turbolas.write(well, "exported_las3.las", version=3.0, delimiter="COMMA")

# Or with TAB delimiter
turbolas.write(well, "exported_las3_tab.las", version=3.0, delimiter="TAB")
```

When serializing text columns containing commas, spaces, or quotes, TurboLAS automatically applies smart quoting to maintain syntax compliance.
