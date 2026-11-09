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
use pyo3::types::{PyBytes, PyDict, PyList, PyMemoryView};
use std::collections::HashMap;
use std::sync::Arc;

use crate::columnar::{ColumnarBuffer, ColumnarParser, PyColumnarBuffer};
use crate::encoding::decode_buffer;
use crate::error::{LASDataError, LASHeaderError, LASSectionError};
use crate::header::HeaderItem;
use crate::null::{NullPolicy, NullSentinel};
use crate::section::SectionType;
use crate::state_machine::{LineAction, SectionStateMachine, ValidationMode};
use crate::unwrapper;

/// Output representation format for exported curve data.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Export curve data as an exported 1D NumPy array (`float64`).
    #[default]
    NumPy,
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
    data: Arc<Vec<f64>>,
    text_data: Option<Arc<Vec<String>>>,
    output_format: OutputFormat,
}

impl LasCurve {
    /// Creates a new numeric `LasCurve` instance.
    #[must_use]
    pub fn new(
        mnemonic: impl Into<String>,
        unit: impl Into<String>,
        value: impl Into<String>,
        description: impl Into<String>,
        column_index: usize,
        data: impl Into<Arc<Vec<f64>>>,
    ) -> Self {
        Self {
            mnemonic: mnemonic.into(),
            unit: unit.into(),
            value: value.into(),
            description: description.into(),
            column_index,
            data: data.into(),
            text_data: None,
            output_format: OutputFormat::NumPy,
        }
    }

    /// Creates a new `LasCurve` instance with optional text data.
    #[must_use]
    pub fn new_with_text(
        mnemonic: impl Into<String>,
        unit: impl Into<String>,
        value: impl Into<String>,
        description: impl Into<String>,
        column_index: usize,
        data: impl Into<Arc<Vec<f64>>>,
        text_data: Option<Arc<Vec<String>>>,
    ) -> Self {
        Self {
            mnemonic: mnemonic.into(),
            unit: unit.into(),
            value: value.into(),
            description: description.into(),
            column_index,
            data: data.into(),
            text_data,
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
    /// Creates a new `LasCurve` instance.
    #[new]
    #[pyo3(signature = (mnemonic, unit = "", value = "", description = "", column_index = 0, data = vec![]))]
    pub fn py_new(
        mnemonic: &str,
        unit: &str,
        value: &str,
        description: &str,
        column_index: usize,
        data: Vec<f64>,
    ) -> Self {
        Self::new(mnemonic, unit, value, description, column_index, data)
    }

    /// Whether this curve contains string / categorical text data.
    #[getter]
    pub fn is_string(&self) -> bool {
        self.text_data.is_some()
    }

    /// Returns a list of strings if this is a text curve.
    pub fn string_data(&self) -> Option<Vec<String>> {
        self.text_data.as_ref().map(|v| v.as_ref().clone())
    }
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

    /// Alias for description (compatible with `lasio.descr`).
    #[getter]
    pub fn descr(&self) -> &str {
        &self.description
    }

    /// Zero-based column offset in the raw columnar matrix.
    #[getter]
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Current output format preference for .data.
    #[getter]
    pub fn output_format(&self) -> String {
        match self.output_format {
            OutputFormat::NumPy => "numpy".to_string(),
            OutputFormat::Bytes => "bytes".to_string(),
            OutputFormat::List => "list".to_string(),
        }
    }

    /// Sets the output format preference for .data.
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

    /// Returns a copy of the curve with the specified output format.
    #[pyo3(name = "with_output_format")]
    pub fn py_with_output_format(&self, format: &Bound<'_, PyAny>) -> Result<LasCurve, PyErr> {
        let fmt = if let Ok(s) = format.extract::<&str>() {
            parse_output_format_str(s)?
        } else if let Ok(f) = format.extract::<OutputFormat>() {
            f
        } else {
            return Err(PyTypeError::new_err(
                "output_format must be a string or OutputFormat enum",
            ));
        };
        Ok(self.clone().with_output_format(fmt))
    }

    /// Number of sample rows in this curve.
    #[getter]
    pub fn num_rows(&self) -> usize {
        if let Some(ref t) = self.text_data {
            t.len()
        } else {
            self.data.len()
        }
    }

    /// Shape of the curve vector as a tuple `(num_rows,)`.
    #[getter]
    pub fn shape(&self) -> (usize,) {
        (self.num_rows(),)
    }

    /// Number of missing values (NaN for floats, empty strings for text) in this curve.
    pub fn null_count(&self) -> usize {
        if let Some(ref t) = self.text_data {
            t.iter().filter(|s| s.is_empty()).count()
        } else {
            self.data.iter().filter(|v| v.is_nan()).count()
        }
    }

    /// Computes `(min, max)` of non-NaN values in this curve.
    pub fn min_max(&self) -> Option<(f64, f64)> {
        if self.is_string() {
            return None;
        }
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut found = false;
        for &val in self.data.iter() {
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
        if self.is_string() {
            return None;
        }
        let mut sum = 0.0;
        let mut count = 0usize;
        for &val in self.data.iter() {
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
    pub fn as_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if self.is_string() {
            return Err(PyTypeError::new_err(
                "Cannot export string curve as float64 bytes",
            ));
        }
        // SAFETY: self.data is a valid contiguous [f64].
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().cast::<u8>(),
                self.data.len().saturating_mul(std::mem::size_of::<f64>()),
            )
        };
        Ok(PyBytes::new(py, byte_slice))
    }

    /// Exports curve data as a zero-copy 1D NumPy array view.
    pub fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(ref text_vec) = self.text_data {
            let py_list = PyList::new(py, text_vec.as_slice())?;
            return Ok(py_list.into());
        }
        let np = py.import("numpy")?;
        let dtype = np.getattr("float64")?;
        let py_bytes = self.as_bytes(py)?;
        let arr = np.call_method1("frombuffer", (py_bytes, &dtype))?;
        Ok(arr.into())
    }

    /// Exports curve data as a zero-copy Python `memoryview`.
    pub fn as_memoryview<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyMemoryView>> {
        let py_bytes = self.as_bytes(py)?;
        PyMemoryView::from(&py_bytes)
    }

    /// Exports curve data as a Python `list[float]`.
    pub fn to_list(&self) -> Vec<f64> {
        if let Some(ref t) = self.text_data {
            t.iter()
                .map(|s| fast_float::parse::<f64, _>(s.as_bytes()).unwrap_or(f64::NAN))
                .collect()
        } else {
            self.data.to_vec()
        }
    }

    /// Surfaces curve data based on the active output format.
    #[getter]
    pub fn data(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(ref text_vec) = self.text_data {
            let py_list = PyList::new(py, text_vec.as_slice())?;
            return Ok(py_list.into());
        }
        match self.output_format {
            OutputFormat::NumPy => match self.to_numpy(py) {
                Ok(arr) => Ok(arr),
                Err(_) => Ok(self.as_bytes(py)?.into()),
            },
            OutputFormat::Bytes => Ok(self.as_bytes(py)?.into()),
            OutputFormat::List => {
                let py_list = PyList::new(py, self.data.as_slice())?;
                Ok(py_list.into())
            }
        }
    }

    /// Provides NumPy `__array_interface__` dictionary for zero-copy wrapping.
    #[getter]
    pub fn __array_interface__(&self, py: Python<'_>) -> PyResult<PyObject> {
        if self.is_string() {
            return Err(PyTypeError::new_err(
                "Cannot construct float64 __array_interface__ for string curve",
            ));
        }
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
        self.num_rows()
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<PyObject> {
        let len = self.num_rows() as isize;
        let norm = if idx < 0 { idx + len } else { idx };
        if norm >= 0 && (norm as usize) < self.num_rows() {
            if let Some(ref text_vec) = self.text_data {
                let py_str = pyo3::types::PyString::new(py, &text_vec[norm as usize]);
                Ok(py_str.into_any().unbind())
            } else {
                let py_float = pyo3::types::PyFloat::new(py, self.data[norm as usize]);
                Ok(py_float.into_any().unbind())
            }
        } else {
            Err(PyIndexError::new_err(format!(
                "Curve sample index {idx} out of range (curve has {} samples)",
                self.num_rows()
            )))
        }
    }

    pub fn __repr__(&self) -> String {
        let unit_part = if self.is_string() {
            " [TEXT]".to_string()
        } else if self.unit.is_empty() {
            String::new()
        } else {
            format!(" [{}]", self.unit)
        };
        format!(
            "<LasCurve {}{}: {} samples>",
            self.mnemonic,
            unit_part,
            self.num_rows()
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
    /// Curve metadata items from ~C (Curve) or ~Log_Definition section, in column order.
    pub(crate) curve_items: Vec<HeaderItem>,
    /// Well information items from ~W (Well) section.
    pub(crate) well_items: Vec<HeaderItem>,
    /// Version information items from ~V (Version) section.
    pub(crate) version_items: Vec<HeaderItem>,
    /// Parameter information items from ~P (Parameter) or ~Log_Parameter section.
    pub(crate) parameter_items: Vec<HeaderItem>,
    /// Other non-structured comments / text lines from ~O (Other) section.
    pub(crate) other_lines: Vec<String>,
    /// Output format preference for curve lookups.
    pub(crate) output_format: OutputFormat,
    /// Additional data sections in LAS 3.0 (e.g. "CORE", "INCLINOMETRY", "TOPS").
    pub(crate) data_sections: HashMap<String, ColumnarBuffer>,
    /// Additional definition sections in LAS 3.0 (e.g. "CORE", "TOPS").
    pub(crate) definition_sections: HashMap<String, Vec<HeaderItem>>,
    /// Delimiter specified by DLM item in ~V (e.g. "COMMA", "SPACE", "TAB").
    pub(crate) delimiter: Option<String>,
}

impl LasWell {
    /// Creates a new `LasWell` with support for LAS 3.0 multi-section data.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn with_sections(
        data: ColumnarBuffer,
        curve_items: Option<Vec<HeaderItem>>,
        well_items: Option<Vec<HeaderItem>>,
        version_items: Option<Vec<HeaderItem>>,
        parameter_items: Option<Vec<HeaderItem>>,
        other_lines: Option<Vec<String>>,
        output_format: OutputFormat,
        data_sections: HashMap<String, ColumnarBuffer>,
        definition_sections: HashMap<String, Vec<HeaderItem>>,
        delimiter: Option<String>,
    ) -> Self {
        let mut lookup = HashMap::with_capacity(data.num_columns());
        for (idx, name) in data.column_names().iter().enumerate() {
            let norm = normalize_mnemonic(name);
            if lookup.contains_key(&norm) {
                lookup.insert(format!("{norm}_{idx}"), idx);
            } else {
                lookup.insert(norm, idx);
            }
        }

        let curves = curve_items.unwrap_or_else(|| {
            data.column_names()
                .iter()
                .map(|name| HeaderItem::new_item(name, "", "", "", 0, ""))
                .collect()
        });

        // Also index mnemonics from curve items if present, handling duplicate mnemonics in permissive mode
        for (idx, item) in curves.iter().enumerate() {
            let norm = normalize_mnemonic(item.mnemonic());
            if lookup.contains_key(&norm) {
                lookup.insert(format!("{norm}_{idx}"), idx);
            } else {
                lookup.insert(norm, idx);
            }
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
            data_sections,
            definition_sections,
            delimiter,
        }
    }

    /// Accesses the inner columnar buffer.
    #[must_use]
    pub fn inner_data(&self) -> &ColumnarBuffer {
        &self.data
    }

    /// Accesses parsed well items.
    #[must_use]
    pub fn well_items(&self) -> &[HeaderItem] {
        &self.well_items
    }

    /// Accesses parsed version items.
    #[must_use]
    pub fn version_items(&self) -> &[HeaderItem] {
        &self.version_items
    }

    /// Accesses parsed curve items.
    #[must_use]
    pub fn curve_items(&self) -> &[HeaderItem] {
        &self.curve_items
    }

    /// Accesses parsed parameter items.
    #[must_use]
    pub fn parameter_items(&self) -> &[HeaderItem] {
        &self.parameter_items
    }

    /// Accesses raw lines from ~Other section.
    #[must_use]
    pub fn other_lines(&self) -> &[String] {
        &self.other_lines
    }

    /// Returns a [`ColumnarBuffer`] for a specific section (e.g. "CORE", "LOG").
    #[must_use]
    pub fn section_data_buffer(&self, name: &str) -> Option<ColumnarBuffer> {
        let upper = name.trim().to_ascii_uppercase();
        if upper == "LOG" || upper == "CURVE" || upper == "ASCII" {
            return Some(self.data.clone());
        }
        self.data_sections.get(&upper).cloned()
    }

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
        Self::with_sections(
            data,
            curve_items,
            well_items,
            version_items,
            parameter_items,
            other_lines,
            output_format,
            HashMap::new(),
            HashMap::new(),
            None,
        )
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

        // If numpy is not installed, return PyBytes buffer
        let py_bytes = PyBytes::new(py, bytes);
        Ok(py_bytes.into())
    }

    /// Surfaces curve data based on the active output format.
    pub(crate) fn export_column(&self, py: Python<'_>, col_idx: usize) -> PyResult<PyObject> {
        if self.data.is_string_column(col_idx) {
            let col = self.data.column_text(col_idx).ok_or_else(|| {
                PyIndexError::new_err(format!("Column index {col_idx} out of range"))
            })?;
            let py_list = PyList::new(py, col)?;
            return Ok(py_list.into());
        }
        match self.output_format {
            OutputFormat::NumPy => self.export_column_numpy(py, col_idx),
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
        let name = &self.data.column_names()[idx];
        let is_str = self.data.is_string_column(idx);
        let text_data = if is_str {
            self.data.column_text_arc(idx)
        } else {
            None
        };
        let col = if is_str {
            Arc::new(Vec::new())
        } else {
            self.data
                .column_arc(idx)
                .ok_or_else(|| PyIndexError::new_err(format!("Column index {idx} out of range")))?
        };

        let (unit, val, desc) = if let Some(item) = self.curve_items.get(idx) {
            (
                item.unit().to_string(),
                item.value().to_string(),
                item.description().to_string(),
            )
        } else {
            (String::new(), String::new(), String::new())
        };

        Ok(
            LasCurve::new_with_text(name, unit, val, desc, idx, col, text_data)
                .with_output_format(self.output_format),
        )
    }

    /// Returns a dictionary mapping curve mnemonics to rich [`LasCurve`] objects.
    #[getter]
    pub fn curves(&self) -> HashMap<String, LasCurve> {
        let mut map = HashMap::with_capacity(self.data.num_columns());
        for i in 0..self.data.num_columns() {
            let name = &self.data.column_names()[i];
            let is_str = self.data.is_string_column(i);
            let text_data = if is_str {
                self.data.column_text_arc(i)
            } else {
                None
            };
            let col = if is_str {
                Arc::new(Vec::new())
            } else {
                self.data
                    .column_arc(i)
                    .unwrap_or_else(|| Arc::new(Vec::new()))
            };
            let (unit, val, desc) = if let Some(item) = self.curve_items.get(i) {
                (
                    item.unit().to_string(),
                    item.value().to_string(),
                    item.description().to_string(),
                )
            } else {
                (String::new(), String::new(), String::new())
            };
            let c = LasCurve::new_with_text(name, unit, val, desc, i, col, text_data)
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

    /// Explicitly surfaces a curve as a zero-copy Python `memoryview`.
    pub fn curve_memoryview<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyMemoryView>> {
        let idx = self.resolve_column_index(key)?;
        if self.data.is_string_column(idx) {
            return Err(PyTypeError::new_err(
                "Cannot construct float64 memoryview for string curve",
            ));
        }
        let py_bytes = self.curve_bytes(py, key)?;
        PyMemoryView::from(&py_bytes)
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

    /// Exports all curves to a list of float lists (one list per column).
    pub fn to_list(&self) -> Vec<Vec<f64>> {
        self.data.to_list()
    }

    /// Exports the well data as a 2D NumPy array `(num_rows, num_columns)` or dict of 1D arrays.
    #[pyo3(signature = (as_dict = false))]
    pub fn to_numpy(&self, py: Python<'_>, as_dict: bool) -> PyResult<PyObject> {
        let py_buf = PyColumnarBuffer::from_inner(self.data.clone());
        py_buf.to_numpy(py, as_dict)
    }

    /// Exports all curves to a dictionary mapping curve names to lists of values in column order.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.data.to_dict(py)
    }

    /// Serializes this well log to a standard CWLS LAS formatted file or string.
    ///
    /// If `destination` is provided as a string, writes directly to that filepath.
    /// If `destination` is None, returns the LAS file content as a Python `str`.
    #[pyo3(signature = (
        destination = None,
        version = 2.0,
        wrap = false,
        delimiter = None,
        float_precision = 4,
        null_value = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &self,
        py: Python<'_>,
        destination: Option<&str>,
        version: f64,
        wrap: bool,
        delimiter: Option<String>,
        float_precision: usize,
        null_value: Option<f64>,
    ) -> PyResult<PyObject> {
        let options = crate::writer::WriteOptions {
            version,
            wrap,
            delimiter,
            float_precision,
            null_value: null_value.unwrap_or_else(|| self.null_value().unwrap_or(-999.25)),
        };

        if let Some(path) = destination {
            crate::writer::write_las_file(self, path, &options)
                .map_err(|e| PyValueError::new_err(format!("Failed to write LAS file: {e}")))?;
            Ok(py.None())
        } else {
            let content = crate::writer::write_las_string(self, &options)
                .map_err(|e| PyValueError::new_err(format!("Failed to format LAS string: {e}")))?;
            let py_str = pyo3::types::PyString::new(py, &content);
            Ok(py_str.into_any().unbind())
        }
    }

    /// Whether this file is detected or confirmed as CWLS LAS 3.0.
    #[getter]
    pub fn is_las3(&self) -> bool {
        self.version()
            .map(|v| v.trim().starts_with("3."))
            .unwrap_or(false)
            || !self.data_sections.is_empty()
    }

    /// Delimiter specified by DLM in ~V (e.g. "COMMA", "SPACE", "TAB").
    #[getter]
    pub fn delimiter(&self) -> Option<String> {
        self.delimiter.clone()
    }

    /// Names of all data sections present in the file (e.g. `["LOG", "CORE"]`).
    #[getter]
    pub fn section_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if !self.data.is_empty() || !self.curve_items.is_empty() {
            names.push("LOG".to_string());
        }
        for k in self.data_sections.keys() {
            if !names.contains(k) {
                names.push(k.clone());
            }
        }
        names
    }

    /// Checks whether a specific data section exists (case-insensitive, e.g. "CORE").
    pub fn has_section(&self, name: &str) -> bool {
        let upper = name.trim().to_ascii_uppercase();
        if upper == "LOG" || upper == "CURVE" || upper == "ASCII" {
            return true;
        }
        self.data_sections.contains_key(&upper)
    }

    /// Returns a [`ColumnarBuffer`] for a specific section (e.g. "CORE", "LOG").
    pub fn section_data(&self, name: &str) -> Option<PyColumnarBuffer> {
        let upper = name.trim().to_ascii_uppercase();
        if upper == "LOG" || upper == "CURVE" || upper == "ASCII" {
            return Some(PyColumnarBuffer::from_inner(self.data.clone()));
        }
        self.data_sections
            .get(&upper)
            .cloned()
            .map(PyColumnarBuffer::from_inner)
    }

    /// Returns curve definition items for a specific section (e.g. "CORE", "LOG").
    pub fn section_curves(&self, name: &str) -> Option<Vec<HeaderItem>> {
        let upper = name.trim().to_ascii_uppercase();
        if upper == "LOG" || upper == "CURVE" || upper == "ASCII" {
            return Some(self.curve_items.clone());
        }
        self.definition_sections.get(&upper).cloned()
    }

    /// Returns a dictionary of all additional section data buffers.
    #[getter]
    pub fn sections(&self) -> HashMap<String, PyColumnarBuffer> {
        let mut map = HashMap::with_capacity(self.data_sections.len() + 1);
        map.insert(
            "LOG".to_string(),
            PyColumnarBuffer::from_inner(self.data.clone()),
        );
        for (k, v) in &self.data_sections {
            map.insert(k.clone(), PyColumnarBuffer::from_inner(v.clone()));
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
        "bytes" | "raw" | "buffer" => Ok(OutputFormat::Bytes),
        "list" => Ok(OutputFormat::List),
        other => Err(PyValueError::new_err(format!(
            "Unsupported output format '{other}'; supported: 'numpy', 'bytes', 'list'"
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
    let mut null_sentinel = NullSentinel::default_cwls().with_policy(policy);

    let mut version_items: Vec<HeaderItem> = Vec::new();
    let mut well_items: Vec<HeaderItem> = Vec::new();
    let mut curve_items: Vec<HeaderItem> = Vec::new();
    let mut parameter_items: Vec<HeaderItem> = Vec::new();
    let mut other_lines: Vec<String> = Vec::new();

    let mut definition_sections: HashMap<String, Vec<HeaderItem>> = HashMap::new();
    let mut data_sections: HashMap<String, ColumnarBuffer> = HashMap::new();
    let mut primary_columnar: Option<ColumnarBuffer> = None;

    let mut is_wrapped = false;
    let mut allow_commas = false;
    let mut delimiter_str: Option<String> = None;

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
            let action = sm
                .process_line(trimmed)
                .map_err(|e| LASSectionError::new_err(format!("Line {line_no}: {e}")))?;

            if let LineAction::SectionStart { section, delimiter } = action {
                if section == SectionType::Ascii {
                    let data_start = next_line_start;
                    let sec_name = delimiter.section_name().to_ascii_uppercase();

                    // Determine curve names for this data section
                    let is_primary = sec_name == "LOG"
                        || sec_name == "ASCII"
                        || sec_name == "CURVE"
                        || (primary_columnar.is_none() && data_sections.is_empty());

                    let mut section_curves: Vec<String> = if is_primary && !curve_items.is_empty() {
                        curve_items
                            .iter()
                            .map(|c| c.mnemonic().to_string())
                            .collect()
                    } else if let Some(defs) = definition_sections.get(&sec_name) {
                        defs.iter().map(|c| c.mnemonic().to_string()).collect()
                    } else if !curve_items.is_empty() {
                        curve_items
                            .iter()
                            .map(|c| c.mnemonic().to_string())
                            .collect()
                    } else {
                        Vec::new()
                    };

                    if section_curves.is_empty() && !delimiter.title().is_empty() {
                        let tokens: Vec<String> = delimiter
                            .title()
                            .split_whitespace()
                            .map(str::to_string)
                            .collect();
                        if !tokens.is_empty() {
                            for t in &tokens {
                                curve_items.push(HeaderItem::new_item(t, "", "", "", line_no, ""));
                            }
                            section_curves = tokens;
                        }
                    }

                    let data_bytes = &bytes[data_start..];
                    let (buf, consumed) = if !section_curves.is_empty() {
                        if is_wrapped {
                            let b = unwrapper::unwrap_and_parse(
                                data_bytes,
                                section_curves.len(),
                                Some(section_curves.clone()),
                                val_mode.is_strict(),
                                allow_commas,
                                Some(null_sentinel.clone()),
                            )
                            .map_err(|e| LASDataError::new_err(e.to_string()))?;
                            (b, data_bytes.len())
                        } else {
                            let parser = ColumnarParser::new(section_curves.len())
                                .with_column_names(section_curves.clone())
                                .with_null_sentinel(null_sentinel.clone())
                                .with_strict(val_mode.is_strict())
                                .with_allow_commas(allow_commas);
                            parser
                                .parse_bytes_with_consumed(data_bytes)
                                .map_err(|e| LASDataError::new_err(e.to_string()))?
                        }
                    } else {
                        (
                            ColumnarBuffer::with_names(section_curves, 0),
                            data_bytes.len(),
                        )
                    };

                    if is_primary && primary_columnar.is_none() {
                        primary_columnar = Some(buf);
                    } else {
                        data_sections.insert(sec_name, buf);
                    }

                    // In LAS 2.0 / 1.2, no sections follow Ascii: break immediately
                    if !sm.is_las3() {
                        break;
                    }

                    line_start = data_start + consumed;
                    continue;
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
            let current_sec_name = sm
                .current_delimiter()
                .map(|d| d.section_name().to_ascii_uppercase())
                .unwrap_or_else(|| sec.name().to_string());

            match sec {
                SectionType::Version => match HeaderItem::parse(line_str, line_no, val_mode) {
                    Ok(item) => {
                        let mnem_upper = item.mnemonic().to_ascii_uppercase();
                        if mnem_upper == "VERS" {
                            if item.value().trim().starts_with('3') {
                                sm.set_las3(true);
                            }
                        } else if mnem_upper == "WRAP" {
                            is_wrapped = item.value().trim().eq_ignore_ascii_case("YES");
                        } else if mnem_upper == "DLM" {
                            let dlm_val = item.value().trim().to_ascii_uppercase();
                            if dlm_val == "COMMA" || dlm_val == "," {
                                allow_commas = true;
                                delimiter_str = Some("COMMA".to_string());
                            } else if dlm_val == "TAB" || dlm_val == "\\T" {
                                delimiter_str = Some("TAB".to_string());
                            } else if dlm_val == "SPACE" {
                                delimiter_str = Some("SPACE".to_string());
                            } else {
                                delimiter_str = Some(item.value().trim().to_string());
                            }
                        }
                        version_items.push(item);
                    }
                    Err(e) => {
                        if val_mode.is_strict() {
                            return Err(LASHeaderError::new_err(format!("Line {line_no}: {e}")));
                        }
                    }
                },
                SectionType::Well => match HeaderItem::parse(line_str, line_no, val_mode) {
                    Ok(item) => {
                        if item.mnemonic().eq_ignore_ascii_case("NULL") {
                            let val_str = item.value().trim();
                            if let Ok(val) = fast_float::parse::<f64, _>(val_str.as_bytes()) {
                                null_sentinel = NullSentinel::new(val).with_policy(policy);
                            }
                        }
                        well_items.push(item);
                    }
                    Err(e) => {
                        if val_mode.is_strict() {
                            return Err(LASHeaderError::new_err(format!("Line {line_no}: {e}")));
                        }
                    }
                },
                SectionType::Curve => match HeaderItem::parse(line_str, line_no, val_mode) {
                    Ok(item) => {
                        if current_sec_name == "LOG" || current_sec_name == "CURVE" {
                            curve_items.push(item);
                        } else {
                            definition_sections
                                .entry(current_sec_name)
                                .or_default()
                                .push(item);
                        }
                    }
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
                SectionType::Ascii => {}
            }
        }

        line_start = next_line_start;
    }

    // Call finish on state machine
    sm.finish()
        .map_err(|e| LASSectionError::new_err(e.to_string()))?;

    // If curve_items is empty but definition_sections has entries, promote first definition section
    if curve_items.is_empty() {
        if let Some((first_name, items)) = definition_sections.iter().next() {
            let first_name = first_name.clone();
            curve_items.clone_from(items);
            if let Some(buf) = data_sections.remove(&first_name) {
                primary_columnar = Some(buf);
            }
        }
    }

    // Extract curve names defined in primary curve section
    let curve_names: Vec<String> = curve_items
        .iter()
        .map(|c| c.mnemonic().to_string())
        .collect();

    let final_primary_buffer =
        primary_columnar.unwrap_or_else(|| ColumnarBuffer::with_names(curve_names, 0));

    Ok(LasWell::with_sections(
        final_primary_buffer,
        Some(curve_items),
        Some(well_items),
        Some(version_items),
        Some(parameter_items),
        Some(other_lines),
        output_format,
        data_sections,
        definition_sections,
        delimiter_str,
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

/// Standalone convenience function to serialize a LAS well to a file or string.
#[pyfunction]
#[pyo3(name = "write_las", signature = (
    well,
    destination = None,
    version = 2.0,
    wrap = false,
    delimiter = None,
    float_precision = 4,
    null_value = None
))]
#[allow(clippy::too_many_arguments)]
pub fn write_las(
    py: Python<'_>,
    well: &LasWell,
    destination: Option<&str>,
    version: f64,
    wrap: bool,
    delimiter: Option<String>,
    float_precision: usize,
    null_value: Option<f64>,
) -> PyResult<PyObject> {
    well.write(
        py,
        destination,
        version,
        wrap,
        delimiter,
        float_precision,
        null_value,
    )
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
