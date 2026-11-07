//! CWLS LAS well log high-level representation and ergonomic Python mapping interface.
//!
//! Provides the [`LasWell`] struct, representing a complete parsed CWLS LAS well log
//! with metadata sections (~V, ~W, ~C, ~P, ~O) and columnar numerical data (~A).
//!
//! Implements PyO3's mapping protocol (`__getitem__`, `__contains__`, `__len__`, `__iter__`,
//! `keys`, `values`, `items`, `get`) with $O(1)$ case-insensitive ASCII mnemonic lookup
//! pointing to column offsets in contiguous numeric buffers.

#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::trivially_copy_pass_by_ref,
    clippy::single_match_else,
    clippy::map_unwrap_or,
    clippy::if_not_else,
    clippy::manual_midpoint
)]

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;

use crate::columnar::{ColumnarBuffer, ColumnarParser, PyColumnarBuffer};
use crate::encoding::decode_buffer;
use crate::error::{LASDataError, LASHeaderError, LASSectionError};
use crate::header::HeaderItem;
use crate::null::{NullPolicy, NullSentinel};
use crate::section::SectionType;
use crate::state_machine::{SectionStateMachine, ValidationMode};
use crate::unwrapper;

/// Output representation format for exported curve data.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Export curve data as an exported 1D NumPy array (`float64`).
    #[default]
    NumPy,
    /// Export curve data as an Apache Arrow `Array` (`pa.float64()`).
    Arrow,
    /// Export raw contiguous IEEE 754 64-bit little-endian bytes (`bytes`).
    Bytes,
    /// Export curve data as a standard Python `list[float]`.
    List,
}

#[pymethods]
impl OutputFormat {
    fn __repr__(&self) -> String {
        match self {
            Self::NumPy => "OutputFormat.NumPy".to_string(),
            Self::Arrow => "OutputFormat.Arrow".to_string(),
            Self::Bytes => "OutputFormat.Bytes".to_string(),
            Self::List => "OutputFormat.List".to_string(),
        }
    }
}

/// Normalizes a curve mnemonic string for case-insensitive ASCII lookup.
#[inline]
pub(crate) fn normalize_mnemonic(s: &str) -> String {
    s.trim().to_ascii_uppercase()
}

/// Rich representation of an individual well log curve.
///
/// Contains curve metadata (mnemonic, unit, API value, description)
/// alongside contiguous numeric sample data.
#[pyclass(name = "LasCurve")]
#[derive(Debug, Clone)]
pub struct LasCurve {
    mnemonic: String,
    unit: String,
    value: String,
    description: String,
    column_index: usize,
    data: Vec<f64>,
    output_format: OutputFormat,
}

impl LasCurve {
    /// Creates a new `LasCurve` instance.
    #[must_use]
    pub fn new(
        mnemonic: impl Into<String>,
        unit: impl Into<String>,
        value: impl Into<String>,
        description: impl Into<String>,
        column_index: usize,
        data: Vec<f64>,
    ) -> Self {
        Self {
            mnemonic: mnemonic.into(),
            unit: unit.into(),
            value: value.into(),
            description: description.into(),
            column_index,
            data,
            output_format: OutputFormat::NumPy,
        }
    }

    /// Sets the output format for `.data`.
    #[must_use]
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }
}

#[pymethods]
impl LasCurve {
    /// Curve mnemonic identifier (e.g. `"GR"`, `"DEPT"`).
    #[getter]
    pub fn mnemonic(&self) -> &str {
        &self.mnemonic
    }

    /// Unit of measurement (e.g. `"GAPI"`, `"M"`, `"OHMM"`).
    #[getter]
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Value / curve code field from ~C line.
    #[getter]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Descriptive text for this curve.
    #[getter]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Zero-based column offset in the raw columnar matrix.
    #[getter]
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Number of sample rows in this curve.
    #[getter]
    pub fn num_rows(&self) -> usize {
        self.data.len()
    }

    /// Shape of the curve vector as a tuple `(num_rows,)`.
    #[getter]
    pub fn shape(&self) -> (usize,) {
        (self.data.len(),)
    }

    /// Number of `NaN` / missing values in this curve.
    pub fn null_count(&self) -> usize {
        self.data.iter().filter(|v| v.is_nan()).count()
    }

    /// Computes `(min, max)` of non-NaN values in this curve.
    pub fn min_max(&self) -> Option<(f64, f64)> {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut found = false;
        for &val in &self.data {
            if !val.is_nan() {
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
                found = true;
            }
        }
        if found {
            Some((min, max))
        } else {
            None
        }
    }

    /// Computes the arithmetic mean of non-NaN values in this curve.
    pub fn mean(&self) -> Option<f64> {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &val in &self.data {
            if !val.is_nan() {
                sum += val;
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    /// Returns the raw contiguous float data as a Python `bytes` object (IEEE 754 little-endian).
    #[allow(unsafe_code)]
    pub fn as_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        // SAFETY: self.data is a valid contiguous [f64].
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().cast::<u8>(),
                self.data.len().saturating_mul(std::mem::size_of::<f64>()),
            )
        };
        PyBytes::new(py, byte_slice)
    }

    /// Exports curve data as a zero-copy 1D NumPy array view.
    pub fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let dtype = np.getattr("float64")?;
        let py_bytes = self.as_bytes(py);
        let arr = np.call_method1("frombuffer", (py_bytes, &dtype))?;
        Ok(arr.into())
    }

    /// Exports curve data as a zero-copy Apache Arrow `Array`.
    pub fn to_arrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pa = py.import("pyarrow")?;
        let py_bytes = self.as_bytes(py);
        let py_buf = pa.call_method1("py_buffer", (py_bytes,))?;
        let float64_type = pa.call_method0("float64")?;
        let none_val = py.None();
        let buffers = (none_val, py_buf);
        let arr = pa
            .getattr("Array")?
            .call_method1("from_buffers", (float64_type, self.data.len(), buffers))?;
        Ok(arr.into())
    }

    /// Exports curve data as a Python `list[float]`.
    pub fn to_list(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Surfaces curve data based on the active output format.
    #[getter]
    pub fn data(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.output_format {
            OutputFormat::NumPy => match self.to_numpy(py) {
                Ok(arr) => Ok(arr),
                Err(_) => match self.to_arrow(py) {
                    Ok(arr) => Ok(arr),
                    Err(_) => Ok(self.as_bytes(py).into()),
                },
            },
            OutputFormat::Arrow => match self.to_arrow(py) {
                Ok(arr) => Ok(arr),
                Err(_) => self.to_numpy(py),
            },
            OutputFormat::Bytes => Ok(self.as_bytes(py).into()),
            OutputFormat::List => {
                let py_list = PyList::new(py, &self.data)?;
                Ok(py_list.into())
            }
        }
    }

    /// Provides NumPy `__array_interface__` dictionary for zero-copy wrapping.
    #[getter]
    pub fn __array_interface__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ptr = self.data.as_ptr() as usize;
        let dict = PyDict::new(py);
        dict.set_item("shape", (self.data.len(),))?;
        dict.set_item("typestr", "<f8")?;
        dict.set_item("data", (ptr, true))?;
        dict.set_item("version", 3)?;
        Ok(dict.into())
    }

    /// Implements `__array__` protocol for `numpy.asarray(curve)`.
    #[pyo3(signature = (dtype = None, copy = None))]
    pub fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        let arr = self.to_numpy(py)?;
        let py_arr = arr.bind(py);
        let mut result = py_arr.clone();
        if let Some(dt) = dtype {
            result = result.call_method1("astype", (dt,))?;
        }
        if copy == Some(true) {
            result = result.call_method0("copy")?;
        }
        Ok(result.into())
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn __getitem__(&self, idx: isize) -> PyResult<f64> {
        let len = self.data.len() as isize;
        let norm = if idx < 0 { idx + len } else { idx };
        if norm >= 0 && (norm as usize) < self.data.len() {
            Ok(self.data[norm as usize])
        } else {
            Err(PyIndexError::new_err(format!(
                "Curve sample index {idx} out of range (curve has {} samples)",
                self.data.len()
            )))
        }
    }

    pub fn __repr__(&self) -> String {
        let unit_part = if self.unit.is_empty() {
            String::new()
        } else {
            format!(" [{}]", self.unit)
        };
        format!(
            "<LasCurve {}{}: {} samples>",
            self.mnemonic,
            unit_part,
            self.data.len()
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Rust-backed CWLS LAS well log object exposing an ergonomic Python mapping interface.
///
/// Under the hood, maintains an internal case-insensitive ASCII lookup index
/// mapping mnemonics to column offsets in contiguous numeric buffers.
///
/// Lookups like `well["GR"]` resolve in $O(1)$ time, surfacing target curve data
/// as a zero-copy PyArrow array or an exported NumPy array via Python buffer protocol
/// bindings, raising a standard Python `KeyError` if the curve does not exist.
#[pyclass(name = "LasWell")]
#[derive(Debug, Clone)]
pub struct LasWell {
    /// Columnar numeric data containing contiguous Vec<f64> per curve.
    pub(crate) data: ColumnarBuffer,
    /// Fast case-insensitive ASCII lookup map: uppercase trimmed mnemonic -> column index.
    pub(crate) lookup: HashMap<String, usize>,
    /// Curve metadata items from ~C (Curve) section, in column order.
    pub(crate) curve_items: Vec<HeaderItem>,
    /// Well information items from ~W (Well) section.
    pub(crate) well_items: Vec<HeaderItem>,
    /// Version information items from ~V (Version) section.
    pub(crate) version_items: Vec<HeaderItem>,
    /// Parameter information items from ~P (Parameter) section.
    pub(crate) parameter_items: Vec<HeaderItem>,
    /// Other non-structured comments / text lines from ~O (Other) section.
    pub(crate) other_lines: Vec<String>,
    /// Output format preference for curve lookups.
    pub(crate) output_format: OutputFormat,
}

impl LasWell {
    /// Creates a new `LasWell` from a columnar buffer and metadata collections.
    #[must_use]
    pub fn new(
        data: ColumnarBuffer,
        curve_items: Option<Vec<HeaderItem>>,
        well_items: Option<Vec<HeaderItem>>,
        version_items: Option<Vec<HeaderItem>>,
        parameter_items: Option<Vec<HeaderItem>>,
        other_lines: Option<Vec<String>>,
        output_format: OutputFormat,
    ) -> Self {
        let mut lookup = HashMap::with_capacity(data.num_columns());
        for (idx, name) in data.column_names().iter().enumerate() {
            lookup.insert(normalize_mnemonic(name), idx);
        }

        let curves = curve_items.unwrap_or_else(|| {
            data.column_names()
                .iter()
                .map(|name| HeaderItem::new_item(name, "", "", "", 0, ""))
                .collect()
        });

        // Also index mnemonics from curve items if present
        for (idx, item) in curves.iter().enumerate() {
            let norm = normalize_mnemonic(item.mnemonic());
            lookup.entry(norm).or_insert(idx);
        }

        Self {
            data,
            lookup,
            curve_items: curves,
            well_items: well_items.unwrap_or_default(),
            version_items: version_items.unwrap_or_default(),
            parameter_items: parameter_items.unwrap_or_default(),
            other_lines: other_lines.unwrap_or_default(),
            output_format,
        }
    }

    /// Resolves a curve mnemonic (case-insensitive ASCII) to a column index in $O(1)$ time.
    #[must_use]
    pub fn resolve_mnemonic(&self, name: &str) -> Option<usize> {
        let norm = normalize_mnemonic(name);
        if let Some(&idx) = self.lookup.get(&norm) {
            return Some(idx);
        }
        // Fallback: check if user passed "MNEM.UNIT"
        if let Some((mnem, _)) = norm.split_once('.') {
            if let Some(&idx) = self.lookup.get(mnem) {
                return Some(idx);
            }
        }
        None
    }

    /// Returns `true` if the well contains a curve matching the given mnemonic.
    #[must_use]
    pub fn has_curve(&self, name: &str) -> bool {
        self.resolve_mnemonic(name).is_some()
    }

    /// Returns a slice of the column float data at the specified index.
    #[must_use]
    pub fn curve_data(&self, index: usize) -> Option<&[f64]> {
        self.data.column(index)
    }

    /// Returns a slice of the column float data by curve mnemonic.
    #[must_use]
    pub fn curve_data_by_name(&self, name: &str) -> Option<&[f64]> {
        self.resolve_mnemonic(name)
            .and_then(|idx| self.curve_data(idx))
    }

    /// Returns raw contiguous float bytes at the specified index.
    #[must_use]
    pub fn raw_curve_bytes(&self, index: usize) -> Option<&[u8]> {
        self.data.column_bytes(index)
    }

    /// Helper to find a header item value in a section.
    #[must_use]
    pub fn get_header_value(&self, section: SectionType, mnemonic: &str) -> Option<&str> {
        let items = match section {
            SectionType::Version => &self.version_items,
            SectionType::Well => &self.well_items,
            SectionType::Curve => &self.curve_items,
            SectionType::Parameter => &self.parameter_items,
            _ => return None,
        };
        let norm = normalize_mnemonic(mnemonic);
        items
            .iter()
            .find(|item| normalize_mnemonic(item.mnemonic()) == norm)
            .map(HeaderItem::value)
    }

    /// Helper to find a full `HeaderItem` in a section.
    #[must_use]
    pub fn get_header_item(&self, section: SectionType, mnemonic: &str) -> Option<&HeaderItem> {
        let items = match section {
            SectionType::Version => &self.version_items,
            SectionType::Well => &self.well_items,
            SectionType::Curve => &self.curve_items,
            SectionType::Parameter => &self.parameter_items,
            _ => return None,
        };
        let norm = normalize_mnemonic(mnemonic);
        items
            .iter()
            .find(|item| normalize_mnemonic(item.mnemonic()) == norm)
    }

    /// Resolves a key (str mnemonic or int index) to a column index.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn resolve_column_index(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(name) = key.extract::<&str>() {
            let trimmed = name.trim();
            if trimmed.is_empty() {
                return Err(PyKeyError::new_err("Curve mnemonic cannot be empty"));
            }
            self.resolve_mnemonic(trimmed).ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "Curve mnemonic '{trimmed}' not found; available curves: {:?}",
                    self.curve_names()
                ))
            })
        } else if let Ok(idx) = key.extract::<isize>() {
            let num_cols = self.data.num_columns() as isize;
            let normalized = if idx < 0 { idx + num_cols } else { idx };
            if normalized >= 0 && (normalized as usize) < self.data.num_columns() {
                Ok(normalized as usize)
            } else {
                Err(PyIndexError::new_err(format!(
                    "Curve index {idx} out of range (well has {} curves)",
                    self.data.num_columns()
                )))
            }
        } else {
            Err(PyTypeError::new_err(
                "Curve identifier must be a string mnemonic or integer index",
            ))
        }
    }

    /// Exports a column as a NumPy array view.
    pub(crate) fn export_column_numpy(&self, py: Python<'_>, col_idx: usize) -> PyResult<PyObject> {
        let bytes = self
            .data
            .column_bytes(col_idx)
            .ok_or_else(|| PyIndexError::new_err(format!("Column index {col_idx} out of range")))?;

        // Attempt zero-copy NumPy array creation via numpy.frombuffer
        if let Ok(np) = py.import("numpy") {
            let dtype = np.getattr("float64")?;
            let py_bytes = PyBytes::new(py, bytes);
            let arr = np.call_method1("frombuffer", (py_bytes, &dtype))?;
            return Ok(arr.into());
        }

        // Fallback: try PyArrow if numpy is not installed
        if let Ok(pa) = py.import("pyarrow") {
            let py_bytes = PyBytes::new(py, bytes);
            let py_buf = pa.call_method1("py_buffer", (py_bytes,))?;
            let float64_type = pa.call_method0("float64")?;
            let none_val = py.None();
            let buffers = (none_val, py_buf);
            let arr = pa.getattr("Array")?.call_method1(
                "from_buffers",
                (float64_type, self.data.num_rows(), buffers),
            )?;
            return Ok(arr.into());
        }

        // If neither is installed, return PyBytes buffer
        let py_bytes = PyBytes::new(py, bytes);
        Ok(py_bytes.into())
    }

    /// Exports a column as a zero-copy PyArrow `Array`.
    pub(crate) fn export_column_arrow(&self, py: Python<'_>, col_idx: usize) -> PyResult<PyObject> {
        let bytes = self
            .data
            .column_bytes(col_idx)
            .ok_or_else(|| PyIndexError::new_err(format!("Column index {col_idx} out of range")))?;
        let num_rows = self.data.num_rows();

        let Ok(pa) = py.import("pyarrow") else {
            // Fallback to NumPy if PyArrow is not installed
            if let Ok(np) = py.import("numpy") {
                let dtype = np.getattr("float64")?;
                let py_bytes = PyBytes::new(py, bytes);
                let arr = np.call_method1("frombuffer", (py_bytes, &dtype))?;
                return Ok(arr.into());
            }
            return Err(pyo3::exceptions::PyImportError::new_err(
                "pyarrow is required for arrow output; install with 'pip install pyarrow'",
            ));
        };

        let py_bytes = PyBytes::new(py, bytes);
        let py_buf = pa.call_method1("py_buffer", (py_bytes,))?;
        let float64_type = pa.call_method0("float64")?;
        let none_val = py.None();
        let buffers = (none_val, py_buf);
        let arr = pa
            .getattr("Array")?
            .call_method1("from_buffers", (float64_type, num_rows, buffers))?;
        Ok(arr.into())
    }

    /// Surfaces curve data based on the active output format.
    pub(crate) fn export_column(&self, py: Python<'_>, col_idx: usize) -> PyResult<PyObject> {
        match self.output_format {
            OutputFormat::NumPy => self.export_column_numpy(py, col_idx),
            OutputFormat::Arrow => self.export_column_arrow(py, col_idx),
            OutputFormat::Bytes => {
                let bytes = self.data.column_bytes(col_idx).ok_or_else(|| {
                    PyIndexError::new_err(format!("Column index {col_idx} out of range"))
                })?;
                Ok(PyBytes::new(py, bytes).into())
            }
            OutputFormat::List => {
                let col = self.data.column(col_idx).ok_or_else(|| {
                    PyIndexError::new_err(format!("Column index {col_idx} out of range"))
                })?;
                let py_list = PyList::new(py, col)?;
                Ok(py_list.into())
            }
        }
    }
}

#[pymethods]
impl LasWell {
    /// Creates a new `LasWell` instance.
    #[new]
    #[pyo3(signature = (
        data = None,
        curve_names = None,
        output_format = "numpy"
    ))]
    pub fn py_new(
        data: Option<&Bound<'_, PyAny>>,
        curve_names: Option<Vec<String>>,
        output_format: &str,
    ) -> PyResult<Self> {
        let fmt = parse_output_format_str(output_format)?;
        if let Some(d) = data {
            if let Ok(py_buf) = d.extract::<PyColumnarBuffer>() {
                return Ok(Self::new(py_buf.inner, None, None, None, None, None, fmt));
            }
            if let Ok(bytes) = d.extract::<&[u8]>() {
                return parse_las_bytes(bytes, None, None, fmt);
            }
            if let Ok(text) = d.extract::<&str>() {
                return parse_las_bytes(text.as_bytes(), None, None, fmt);
            }
            return Err(PyTypeError::new_err(
                "data must be a ColumnarBuffer, bytes, or string LAS content",
            ));
        }

        let names = curve_names.unwrap_or_default();
        let buf = ColumnarBuffer::with_names(names, 0);
        Ok(Self::new(buf, None, None, None, None, None, fmt))
    }

    /// Parses a complete CWLS LAS file from a local filesystem path.
    #[staticmethod]
    #[pyo3(signature = (
        path,
        validation_mode = None,
        null_policy = None,
        output_format = "numpy"
    ))]
    pub fn from_file(
        path: &str,
        validation_mode: Option<ValidationMode>,
        null_policy: Option<NullPolicy>,
        output_format: &str,
    ) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Failed to read LAS file '{path}': {e}"
            ))
        })?;
        let fmt = parse_output_format_str(output_format)?;
        parse_las_bytes(&bytes, validation_mode, null_policy, fmt)
    }

    /// Parses a complete CWLS LAS dataset from a raw byte slice (`&[u8]`).
    #[staticmethod]
    #[pyo3(signature = (
        bytes,
        validation_mode = None,
        null_policy = None,
        output_format = "numpy"
    ))]
    pub fn from_bytes(
        bytes: &[u8],
        validation_mode: Option<ValidationMode>,
        null_policy: Option<NullPolicy>,
        output_format: &str,
    ) -> PyResult<Self> {
        let fmt = parse_output_format_str(output_format)?;
        parse_las_bytes(bytes, validation_mode, null_policy, fmt)
    }

    /// Parses a complete CWLS LAS dataset from a string slice (`&str`).
    #[staticmethod]
    #[pyo3(signature = (
        text,
        validation_mode = None,
        null_policy = None,
        output_format = "numpy"
    ))]
    pub fn from_text(
        text: &str,
        validation_mode: Option<ValidationMode>,
        null_policy: Option<NullPolicy>,
        output_format: &str,
    ) -> PyResult<Self> {
        let fmt = parse_output_format_str(output_format)?;
        parse_las_bytes(text.as_bytes(), validation_mode, null_policy, fmt)
    }

    /// Direct curve lookup via Python mapping protocol: `well["GR"]` or `well[0]`.
    ///
    /// Validates the mnemonic, resolves the column index in $O(1)$ time, and surfaces
    /// the target curve data as a zero-copy PyArrow array or exported NumPy array.
    /// Raises a standard Python `KeyError` if the curve does not exist.
    pub fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let col_idx = self.resolve_column_index(key)?;
        self.export_column(py, col_idx)
    }

    /// Checks if a curve exists in the well: `"GR" in well` or `0 in well`.
    pub fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
        self.resolve_column_index(key).is_ok()
    }

    /// Number of curves in the well (matches standard Python mapping semantics).
    pub fn __len__(&self) -> usize {
        self.data.num_columns()
    }

    /// Iterates over curve mnemonics: `for mnem in well:`.
    pub fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let names = self.data.column_names().to_vec();
        let py_list = PyList::new(py, names)?;
        let iter = py_list.call_method0("__iter__")?;
        Ok(iter.into())
    }

    /// Returns a list of all curve mnemonics in the well.
    pub fn keys(&self) -> Vec<String> {
        self.data.column_names().to_vec()
    }

    /// Returns a list of exported curve data arrays.
    pub fn values(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let mut vals = Vec::with_capacity(self.data.num_columns());
        for i in 0..self.data.num_columns() {
            vals.push(self.export_column(py, i)?);
        }
        Ok(vals)
    }

    /// Returns a list of `(mnemonic, curve_data)` tuples.
    pub fn items(&self, py: Python<'_>) -> PyResult<Vec<(String, PyObject)>> {
        let names = self.data.column_names();
        let mut items = Vec::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            items.push((name.clone(), self.export_column(py, i)?));
        }
        Ok(items)
    }

    /// Safe curve lookup returning `default` (or `None`) if the curve is not found.
    #[pyo3(signature = (key, default = None))]
    pub fn get(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: Option<PyObject>,
    ) -> PyResult<PyObject> {
        match self.resolve_column_index(key) {
            Ok(idx) => self.export_column(py, idx),
            Err(_) => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    /// Retrieves an individual curve as a rich [`LasCurve`] object.
    pub fn curve(&self, key: &Bound<'_, PyAny>) -> PyResult<LasCurve> {
        let idx = self.resolve_column_index(key)?;
        let col = self
            .data
            .column(idx)
            .ok_or_else(|| PyIndexError::new_err(format!("Column index {idx} out of range")))?;
        let name = &self.data.column_names()[idx];

        let (unit, val, desc) = if let Some(item) = self.curve_items.get(idx) {
            (
                item.unit().to_string(),
                item.value().to_string(),
                item.description().to_string(),
            )
        } else {
            (String::new(), String::new(), String::new())
        };

        Ok(LasCurve::new(name, unit, val, desc, idx, col.to_vec())
            .with_output_format(self.output_format))
    }

    /// Returns a dictionary mapping curve mnemonics to rich [`LasCurve`] objects.
    #[getter]
    pub fn curves(&self) -> HashMap<String, LasCurve> {
        let mut map = HashMap::with_capacity(self.data.num_columns());
        for i in 0..self.data.num_columns() {
            let col = self.data.column(i).unwrap_or(&[]);
            let name = &self.data.column_names()[i];
            let (unit, val, desc) = if let Some(item) = self.curve_items.get(i) {
                (
                    item.unit().to_string(),
                    item.value().to_string(),
                    item.description().to_string(),
                )
            } else {
                (String::new(), String::new(), String::new())
            };
            let c = LasCurve::new(name, unit, val, desc, i, col.to_vec())
                .with_output_format(self.output_format);
            map.insert(name.clone(), c);
        }
        map
    }

    /// List of curve mnemonics in column order.
    #[getter]
    pub fn curve_names(&self) -> Vec<String> {
        self.data.column_names().to_vec()
    }

    /// Number of sample rows in the well data.
    #[getter]
    pub fn num_rows(&self) -> usize {
        self.data.num_rows()
    }

    /// Number of curves in the well.
    #[getter]
    pub fn num_curves(&self) -> usize {
        self.data.num_columns()
    }

    /// Number of columns in the well (alias for `num_curves`).
    #[getter]
    pub fn num_columns(&self) -> usize {
        self.data.num_columns()
    }

    /// Shape of the well data matrix as a tuple `(num_rows, num_curves)`.
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.data.shape()
    }

    /// Whether the well contains 0 sample rows.
    #[getter]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Current output format preference for curve lookups.
    #[getter]
    pub fn output_format(&self) -> String {
        match self.output_format {
            OutputFormat::NumPy => "numpy".to_string(),
            OutputFormat::Arrow => "arrow".to_string(),
            OutputFormat::Bytes => "bytes".to_string(),
            OutputFormat::List => "list".to_string(),
        }
    }

    /// Sets the output format preference for curve lookups.
    #[setter]
    pub fn set_output_format(&mut self, format: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(s) = format.extract::<&str>() {
            self.output_format = parse_output_format_str(s)?;
            Ok(())
        } else if let Ok(fmt) = format.extract::<OutputFormat>() {
            self.output_format = fmt;
            Ok(())
        } else {
            Err(PyTypeError::new_err(
                "output_format must be a string or OutputFormat enum",
            ))
        }
    }

    /// Explicitly surfaces a curve as an exported NumPy array view.
    pub fn curve_numpy(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let idx = self.resolve_column_index(key)?;
        self.export_column_numpy(py, idx)
    }

    /// Explicitly surfaces a curve as a zero-copy PyArrow `Array`.
    pub fn curve_arrow(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let idx = self.resolve_column_index(key)?;
        self.export_column_arrow(py, idx)
    }

    /// Explicitly surfaces a curve as raw contiguous IEEE 754 little-endian `bytes`.
    pub fn curve_bytes<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let idx = self.resolve_column_index(key)?;
        let bytes = self
            .data
            .column_bytes(idx)
            .ok_or_else(|| PyIndexError::new_err(format!("Column index {idx} out of range")))?;
        Ok(PyBytes::new(py, bytes))
    }

    /// Explicitly surfaces a curve as a Python `list[float]`.
    pub fn curve_list(&self, key: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let idx = self.resolve_column_index(key)?;
        let col = self
            .data
            .column(idx)
            .ok_or_else(|| PyIndexError::new_err(format!("Column index {idx} out of range")))?;
        Ok(col.to_vec())
    }

    /// Exports the entire well data matrix to an Apache Arrow `Table`.
    pub fn to_arrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        let py_buf = PyColumnarBuffer::from_inner(self.data.clone());
        py_buf.to_arrow(py)
    }

    /// Exports the entire well data matrix to a Polars `DataFrame`.
    pub fn to_polars(&self, py: Python<'_>) -> PyResult<PyObject> {
        let py_buf = PyColumnarBuffer::from_inner(self.data.clone());
        py_buf.to_polars(py)
    }

    /// Exports the entire well data matrix to a Pandas `DataFrame`.
    pub fn to_pandas(&self, py: Python<'_>) -> PyResult<PyObject> {
        let py_buf = PyColumnarBuffer::from_inner(self.data.clone());
        py_buf.to_pandas(py)
    }

    /// Exports the well data as a 2D NumPy array `(num_rows, num_columns)` or dict of 1D arrays.
    #[pyo3(signature = (as_dict = false))]
    pub fn to_numpy(&self, py: Python<'_>, as_dict: bool) -> PyResult<PyObject> {
        let py_buf = PyColumnarBuffer::from_inner(self.data.clone());
        py_buf.to_numpy(py, as_dict)
    }

    /// Exports all curves to a dictionary mapping curve names to lists of floats.
    pub fn to_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut map = HashMap::with_capacity(self.data.num_columns());
        for (idx, name) in self.data.column_names().iter().enumerate() {
            if let Some(col) = self.data.column(idx) {
                map.insert(name.clone(), col.to_vec());
            }
        }
        map
    }

    /// Accesses the underlying [`ColumnarBuffer`] wrapper.
    #[getter]
    pub fn data_buffer(&self) -> PyColumnarBuffer {
        PyColumnarBuffer::from_inner(self.data.clone())
    }

    // -----------------------------------------------------------------------
    // Well Metadata Accessors (from ~W Section)
    // -----------------------------------------------------------------------

    /// Unique Well Identifier (UWI or API number).
    #[getter]
    pub fn uwi(&self) -> Option<String> {
        self.get_header_value(SectionType::Well, "UWI")
            .or_else(|| self.get_header_value(SectionType::Well, "API"))
            .map(str::to_string)
    }

    /// API number of the well.
    #[getter]
    pub fn api(&self) -> Option<String> {
        self.get_header_value(SectionType::Well, "API")
            .map(str::to_string)
    }

    /// Well name.
    #[getter]
    pub fn well_name(&self) -> Option<String> {
        self.get_header_value(SectionType::Well, "WELL")
            .map(str::to_string)
    }

    /// Company / operator name.
    #[getter]
    pub fn company(&self) -> Option<String> {
        self.get_header_value(SectionType::Well, "COMP")
            .map(str::to_string)
    }

    /// Oil / gas field name.
    #[getter]
    pub fn field(&self) -> Option<String> {
        self.get_header_value(SectionType::Well, "FLD")
            .map(str::to_string)
    }

    /// Well location description.
    #[getter]
    pub fn location(&self) -> Option<String> {
        self.get_header_value(SectionType::Well, "LOC")
            .map(str::to_string)
    }

    /// Logging service company name.
    #[getter]
    pub fn service_company(&self) -> Option<String> {
        self.get_header_value(SectionType::Well, "SRVC")
            .map(str::to_string)
    }

    /// Logging date.
    #[getter]
    pub fn date(&self) -> Option<String> {
        self.get_header_value(SectionType::Well, "DATE")
            .map(str::to_string)
    }

    /// Start depth (STRT) value.
    #[getter]
    pub fn start_depth(&self) -> Option<f64> {
        self.get_header_value(SectionType::Well, "STRT")
            .and_then(|s| fast_float::parse::<f64, _>(s.trim().as_bytes()).ok())
    }

    /// Stop depth (STOP) value.
    #[getter]
    pub fn stop_depth(&self) -> Option<f64> {
        self.get_header_value(SectionType::Well, "STOP")
            .and_then(|s| fast_float::parse::<f64, _>(s.trim().as_bytes()).ok())
    }

    /// Step depth (STEP) value.
    #[getter]
    pub fn step(&self) -> Option<f64> {
        self.get_header_value(SectionType::Well, "STEP")
            .and_then(|s| fast_float::parse::<f64, _>(s.trim().as_bytes()).ok())
    }

    /// Declared NULL value sentinel.
    #[getter]
    pub fn null_value(&self) -> Option<f64> {
        self.get_header_value(SectionType::Well, "NULL")
            .and_then(|s| fast_float::parse::<f64, _>(s.trim().as_bytes()).ok())
    }

    // -----------------------------------------------------------------------
    // Version Section Accessors (from ~V Section)
    // -----------------------------------------------------------------------

    /// LAS format version (e.g. `"2.0"`).
    #[getter]
    pub fn version(&self) -> Option<String> {
        self.get_header_value(SectionType::Version, "VERS")
            .map(str::to_string)
    }

    /// Whether the file is in wrapped line mode (`WRAP. YES`).
    #[getter]
    pub fn is_wrapped(&self) -> bool {
        self.get_header_value(SectionType::Version, "WRAP")
            .map(|s| s.trim().eq_ignore_ascii_case("YES"))
            .unwrap_or(false)
    }

    // -----------------------------------------------------------------------
    // Header Items Collections
    // -----------------------------------------------------------------------

    /// Header records from the ~V (Version) section.
    #[getter]
    pub fn version_info(&self) -> Vec<HeaderItem> {
        self.version_items.clone()
    }

    /// Header records from the ~W (Well) section.
    #[getter]
    pub fn well_info(&self) -> Vec<HeaderItem> {
        self.well_items.clone()
    }

    /// Header records from the ~C (Curve) section.
    #[getter]
    pub fn curve_info(&self) -> Vec<HeaderItem> {
        self.curve_items.clone()
    }

    /// Header records from the ~P (Parameter) section.
    #[getter]
    pub fn parameters(&self) -> Vec<HeaderItem> {
        self.parameter_items.clone()
    }

    /// Text lines from the ~O (Other) section.
    #[getter]
    pub fn other(&self) -> Vec<String> {
        self.other_lines.clone()
    }

    /// Looks up a specific `HeaderItem` by section and mnemonic.
    pub fn header_item(&self, section: SectionType, mnemonic: &str) -> Option<HeaderItem> {
        self.get_header_item(section, mnemonic).cloned()
    }

    pub fn __repr__(&self) -> String {
        let uwi_part = match self.uwi() {
            Some(u) => format!(" uwi='{u}'"),
            None => match self.well_name() {
                Some(w) => format!(" well='{w}'"),
                None => String::new(),
            },
        };
        format!(
            "<LasWell{} curves={:?} rows={}>",
            uwi_part,
            self.data.column_names(),
            self.data.num_rows()
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Parses an output format string into the `OutputFormat` enum.
pub(crate) fn parse_output_format_str(format_str: &str) -> PyResult<OutputFormat> {
    match format_str.trim().to_ascii_lowercase().as_str() {
        "numpy" | "np" => Ok(OutputFormat::NumPy),
        "arrow" | "pa" | "pyarrow" => Ok(OutputFormat::Arrow),
        "bytes" | "raw" | "buffer" => Ok(OutputFormat::Bytes),
        "list" => Ok(OutputFormat::List),
        other => Err(PyValueError::new_err(format!(
            "Unsupported output format '{other}'; supported: 'numpy', 'arrow', 'bytes', 'list'"
        ))),
    }
}

/// Internal pipeline parsing a complete LAS byte buffer into a [`LasWell`].
pub fn parse_las_bytes(
    raw_bytes: &[u8],
    validation_mode: Option<ValidationMode>,
    null_policy: Option<NullPolicy>,
    output_format: OutputFormat,
) -> PyResult<LasWell> {
    // 1. Zero-copy encoding detection with CP1252 fallback
    let decoded = decode_buffer(raw_bytes);
    let text = decoded.as_str();

    let val_mode = validation_mode.unwrap_or(ValidationMode::Permissive);
    let mut sm = SectionStateMachine::new(val_mode);
    let policy = null_policy.unwrap_or(NullPolicy::Strict);

    let mut version_items = Vec::new();
    let mut well_items = Vec::new();
    let mut curve_items = Vec::new();
    let mut parameter_items = Vec::new();
    let mut other_lines = Vec::new();

    let mut in_ascii = false;
    let mut ascii_start_offset = 0;

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut line_start = 0;
    let mut line_no = 0;

    while line_start < len {
        line_no += 1;
        // Find end of current line
        let mut line_end = line_start;
        while line_end < len && bytes[line_end] != b'\n' && bytes[line_end] != b'\r' {
            line_end += 1;
        }

        let line_bytes = &bytes[line_start..line_end];
        // Advance to next line start
        let mut next_line_start = line_end;
        if next_line_start < len && bytes[next_line_start] == b'\r' {
            next_line_start += 1;
        }
        if next_line_start < len && bytes[next_line_start] == b'\n' {
            next_line_start += 1;
        }

        let Ok(line_str) = std::str::from_utf8(line_bytes) else {
            line_start = next_line_start;
            continue;
        };
        let trimmed = line_str.trim();

        // Check for section delimiter starting with ~
        if trimmed.starts_with('~') {
            sm.process_line(trimmed)
                .map_err(|e| LASSectionError::new_err(format!("Line {line_no}: {e}")))?;

            if let Some(sec) = sm.current_section() {
                if sec == SectionType::Ascii {
                    in_ascii = true;
                    ascii_start_offset = next_line_start;
                    break;
                }
            }
            line_start = next_line_start;
            continue;
        }

        // Comments and empty lines
        if trimmed.is_empty() || trimmed.starts_with('#') {
            if sm.current_section() == Some(SectionType::Other) {
                other_lines.push(line_str.to_string());
            }
            line_start = next_line_start;
            continue;
        }

        // Process header records according to current section
        if let Some(sec) = sm.current_section() {
            match sec {
                SectionType::Version => match HeaderItem::parse(line_str, line_no, val_mode) {
                    Ok(item) => version_items.push(item),
                    Err(e) => {
                        if val_mode.is_strict() {
                            return Err(LASHeaderError::new_err(format!("Line {line_no}: {e}")));
                        }
                    }
                },
                SectionType::Well => match HeaderItem::parse(line_str, line_no, val_mode) {
                    Ok(item) => well_items.push(item),
                    Err(e) => {
                        if val_mode.is_strict() {
                            return Err(LASHeaderError::new_err(format!("Line {line_no}: {e}")));
                        }
                    }
                },
                SectionType::Curve => match HeaderItem::parse(line_str, line_no, val_mode) {
                    Ok(item) => curve_items.push(item),
                    Err(e) => {
                        if val_mode.is_strict() {
                            return Err(LASHeaderError::new_err(format!("Line {line_no}: {e}")));
                        }
                    }
                },
                SectionType::Parameter => match HeaderItem::parse(line_str, line_no, val_mode) {
                    Ok(item) => parameter_items.push(item),
                    Err(e) => {
                        if val_mode.is_strict() {
                            return Err(LASHeaderError::new_err(format!("Line {line_no}: {e}")));
                        }
                    }
                },
                SectionType::Other => {
                    other_lines.push(line_str.to_string());
                }
                SectionType::Ascii => {
                    in_ascii = true;
                    ascii_start_offset = line_start;
                    break;
                }
            }
        }

        line_start = next_line_start;
    }

    // Extract curve names defined in ~C section
    let curve_names: Vec<String> = curve_items
        .iter()
        .map(|c| c.mnemonic().to_string())
        .collect();

    // Detect NULL value sentinel from ~W section
    let null_sentinel = well_items
        .iter()
        .find(|item| item.mnemonic().eq_ignore_ascii_case("NULL"))
        .and_then(|item| {
            let val_str = item.value().trim();
            fast_float::parse::<f64, _>(val_str.as_bytes()).ok()
        })
        .map(|val| NullSentinel::new(val).with_policy(policy))
        .unwrap_or_else(|| NullSentinel::default_cwls().with_policy(policy));

    // Detect WRAP flag from ~V section
    let is_wrapped = version_items
        .iter()
        .find(|item| item.mnemonic().eq_ignore_ascii_case("WRAP"))
        .map(|item| item.value().trim().eq_ignore_ascii_case("YES"))
        .unwrap_or(false);

    // Ingest ASCII numeric log data
    let columnar_buffer = if in_ascii && !curve_names.is_empty() {
        let ascii_bytes = &bytes[ascii_start_offset..];
        if is_wrapped {
            unwrapper::unwrap_and_parse(
                ascii_bytes,
                curve_names.len(),
                Some(curve_names.clone()),
                val_mode.is_strict(),
                false,
                Some(null_sentinel),
            )
            .map_err(|e| LASDataError::new_err(e.to_string()))?
        } else {
            let parser = ColumnarParser::new(curve_names.len())
                .with_column_names(curve_names.clone())
                .with_null_sentinel(null_sentinel)
                .with_strict(val_mode.is_strict());
            parser
                .parse_bytes(ascii_bytes)
                .map_err(|e| LASDataError::new_err(e.to_string()))?
        }
    } else {
        ColumnarBuffer::with_names(curve_names, 0)
    };

    Ok(LasWell::new(
        columnar_buffer,
        Some(curve_items),
        Some(well_items),
        Some(version_items),
        Some(parameter_items),
        Some(other_lines),
        output_format,
    ))
}

/// Standalone convenience function to read a LAS file from text or bytes.
#[pyfunction]
#[pyo3(name = "read_las", signature = (
    source,
    validation_mode = None,
    null_policy = None,
    output_format = "numpy"
))]
pub fn read_las(
    source: &Bound<'_, PyAny>,
    validation_mode: Option<ValidationMode>,
    null_policy: Option<NullPolicy>,
    output_format: &str,
) -> PyResult<LasWell> {
    let fmt = parse_output_format_str(output_format)?;

    // 1. Direct bytes
    if let Ok(bytes) = source.extract::<&[u8]>() {
        return parse_las_bytes(bytes, validation_mode, null_policy, fmt);
    }

    // 2. String: either file path or raw LAS text
    if let Ok(s) = source.extract::<&str>() {
        // If it starts with '~' or contains newlines, treat as content
        if s.starts_with('~') || s.contains('\n') {
            return parse_las_bytes(s.as_bytes(), validation_mode, null_policy, fmt);
        }
        // Otherwise attempt reading from path
        match std::fs::read(s) {
            Ok(bytes) => parse_las_bytes(&bytes, validation_mode, null_policy, fmt),
            Err(_) => {
                // If file doesn't exist, treat as content if not empty
                if !s.is_empty() {
                    parse_las_bytes(s.as_bytes(), validation_mode, null_policy, fmt)
                } else {
                    Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                        "File not found: '{s}'"
                    )))
                }
            }
        }
    } else {
        Err(PyTypeError::new_err(
            "source must be a file path string, raw LAS text string, or bytes",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_LAS: &str = "\
~Version Information
VERS. 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
WRAP. NO  : ONE LINE PER RECORD FLAG
~Well Information
STRT.M 1000.0 : START DEPTH
STOP.M 1001.0 : STOP DEPTH
STEP.M 0.5    : STEP
NULL. -999.25 : NULL VALUE
WELL. DRILL HOLE #1 : WELL NAME
COMP. ENERGY CORP : COMPANY
FLD . WILDCAT : FIELD
LOC . SEC 12 TWP 34 RGE 5 : LOCATION
SRVC. WIRELINE INC : SERVICE COMPANY
DATE. 2026-11-07 : DATE
UWI . 100/01-01-001-01W4/0 : UNIQUE WELL ID
~Curve Information
DEPT.M    : 1 DEPTH
GR  .GAPI : 2 GAMMA RAY
RHOB.K/M3 : 3 BULK DENSITY
~Ascii
 1000.0  45.2  2.45
 1000.5  48.7  2.46
 1001.0 -999.25 2.48
";

    #[test]
    fn test_parse_sample_las() {
        let well = parse_las_bytes(SAMPLE_LAS.as_bytes(), None, None, OutputFormat::List).unwrap();

        assert_eq!(well.num_curves(), 3);
        assert_eq!(well.num_rows(), 3);
        assert_eq!(well.curve_names(), vec!["DEPT", "GR", "RHOB"]);
        assert_eq!(well.uwi(), Some("100/01-01-001-01W4/0".to_string()));
        assert_eq!(well.well_name(), Some("DRILL HOLE #1".to_string()));
        assert_eq!(well.company(), Some("ENERGY CORP".to_string()));
        assert_eq!(well.start_depth(), Some(1000.0));
        assert_eq!(well.stop_depth(), Some(1001.0));
        assert_eq!(well.step(), Some(0.5));
        assert_eq!(well.null_value(), Some(-999.25));
        assert!(!well.is_wrapped());
        assert_eq!(well.version(), Some("2.0".to_string()));

        // Case-insensitive lookup
        assert!(well.has_curve("GR"));
        assert!(well.has_curve("gr"));
        assert!(well.has_curve("  dept  "));
        assert!(well.has_curve("rhob"));
        assert!(!well.has_curve("UNKNOWN"));

        assert_eq!(well.resolve_mnemonic("GR"), Some(1));
        assert_eq!(well.resolve_mnemonic("dept"), Some(0));
        assert_eq!(well.resolve_mnemonic("RHOB"), Some(2));

        // Curve data check
        let dept = well.curve_data_by_name("DEPT").unwrap();
        assert_eq!(dept, &[1000.0, 1000.5, 1001.0]);

        let gr = well.curve_data_by_name("GR").unwrap();
        assert_eq!(gr[0], 45.2);
        assert_eq!(gr[1], 48.7);
        assert!(gr[2].is_nan()); // -999.25 mapped to NaN!
    }

    #[test]
    fn test_las_curve_metadata_and_stats() {
        let well = parse_las_bytes(SAMPLE_LAS.as_bytes(), None, None, OutputFormat::List).unwrap();

        let curve_map = well.curves();
        let gr_curve = curve_map.get("GR").unwrap();
        assert_eq!(gr_curve.mnemonic(), "GR");
        assert_eq!(gr_curve.unit(), "GAPI");
        assert_eq!(gr_curve.description(), "2 GAMMA RAY");
        assert_eq!(gr_curve.num_rows(), 3);
        assert_eq!(gr_curve.null_count(), 1);
        assert_eq!(gr_curve.min_max(), Some((45.2, 48.7)));
        assert_eq!(gr_curve.mean(), Some((45.2 + 48.7) / 2.0));
    }

    #[test]
    fn test_wrapped_las_parsing() {
        let wrapped_las = "\
~Version Information
VERS. 2.0 : CWLS
WRAP. YES : WRAPPED
~Well Information
NULL. -999.25 : NULL
~Curve Information
DEPT.M : 1 DEPTH
GR.GAPI : 2 GAMMA RAY
RHOB.K/M3 : 3 BULK DENSITY
~Ascii
 1000.0 45.2
 2.45
 1000.5 48.7
 2.46
";
        let well = parse_las_bytes(wrapped_las.as_bytes(), None, None, OutputFormat::List).unwrap();

        assert_eq!(well.num_curves(), 3);
        assert_eq!(well.num_rows(), 2);
        assert!(well.is_wrapped());

        let gr = well.curve_data_by_name("GR").unwrap();
        assert_eq!(gr, &[45.2, 48.7]);
        let rhob = well.curve_data_by_name("RHOB").unwrap();
        assert_eq!(rhob, &[2.45, 2.46]);
    }
}
