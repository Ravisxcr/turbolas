//! In-memory columnar ASCII numeric parser for CWLS LAS files.
//!
//! # Architecture
//!
//! CWLS LAS (Log ASCII Standard) stores telemetry curves in the `~A` (ASCII Log Data) section
//! as whitespace- or delimiter-separated floating-point numbers. In high-density well log files,
//! this section often contains millions of float tokens spanning dozens of curves (e.g. `DEPT`,
//! `GR`, `RHOB`, `NPHI`, `DT`, `RES`).
//!
//! To achieve near-memory-bandwidth ingestion throughput:
//! 1. **Zero-Copy Byte Slice Tokenization**: The parser operates strictly over borrowed byte
//!    slices (`&[u8]`). Field delimiters (spaces, tabs, optional commas) and line endings (`\r\n`,
//!    `\n`, `\r`) are identified directly in the raw byte stream without constructing intermediate
//!    `String` objects, `&str` tokens, or row structs on the heap.
//! 2. **Fast-Float Exact Conversions**: Each identified numeric byte window is passed directly
//!    into `fast_float::parse::<f64, _>`, leveraging the Eisel-Lemire algorithm for exact IEEE 754
//!    floating-point conversions at multiple gigabytes per second.
//! 3. **Pre-allocated Column Vectors**: Instead of collecting row structs (`Vec<Row>`) that require
//!    transposition into columnar format, the parser directly pushes parsed primitives into
//!    pre-allocated, contiguous `Vec<f64>` column vectors. Target vector capacities are calculated
//!    upfront from byte stream length and column count to eliminate dynamic reallocations.
//! 4. **Hardware Prefetcher Friendly**: Contiguous linear column buffers preserve CPU cache
//!    locality and hardware prefetcher efficiency, and are directly compatible with Apache Arrow
//!    columnar arrays, Polars `DataFrames`, `NumPy` `frombuffer`, or low-level compute kernels.

#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;

use crate::encoding::PyDecodedBuffer;
use crate::error::{DataError, LASDataError};
use crate::null::NullSentinel;

/// Estimates row capacity for pre-allocating column vectors based on byte stream size.
///
/// Typical CWLS LAS numeric entries span 8 to 12 bytes including whitespace delimiters.
/// Pre-allocating column vectors with estimated capacity prevents expensive reallocations
/// during batch streaming.
#[must_use]
pub fn estimate_row_capacity(bytes_len: usize, num_columns: usize) -> usize {
    if num_columns == 0 || bytes_len == 0 {
        return 0;
    }
    // Typical float entry is 8-12 bytes. We assume ~10 bytes per float value.
    let bytes_per_row = num_columns.saturating_mul(10).max(16);
    let estimated_rows = bytes_len / bytes_per_row;
    // Provide a 10% headroom buffer and minimum baseline of 16 rows.
    (estimated_rows + (estimated_rows / 10)).max(16)
}

/// Checks whether a non-numeric token represents a standard or declared text NULL sentinel.
#[must_use]
pub fn is_text_null_sentinel(token: &[u8], sentinel: Option<&NullSentinel>) -> bool {
    if token.eq_ignore_ascii_case(b"nan")
        || token.eq_ignore_ascii_case(b"-nan")
        || token.eq_ignore_ascii_case(b"+nan")
        || token.eq_ignore_ascii_case(b"null")
        || token.eq_ignore_ascii_case(b"(null)")
        || token.eq_ignore_ascii_case(b"none")
        || token.eq_ignore_ascii_case(b"na")
        || token.eq_ignore_ascii_case(b"n/a")
        || token == b"-"
        || token == b"?"
    {
        return true;
    }

    if let Some(s) = sentinel {
        if let Some(raw) = s.raw_sentinel() {
            if token == raw.trim().as_bytes() {
                return true;
            }
        }
    }

    false
}

/// Contiguous columnar buffer storing parsed `f64` values per curve.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnarBuffer {
    columns: Vec<Vec<f64>>,
    column_names: Vec<String>,
    num_rows: usize,
}

impl ColumnarBuffer {
    /// Creates a new empty `ColumnarBuffer` with pre-allocated column vectors.
    #[must_use]
    pub fn new(num_columns: usize, capacity: usize) -> Self {
        let columns = (0..num_columns)
            .map(|_| Vec::with_capacity(capacity))
            .collect();
        let column_names = (0..num_columns).map(|i| format!("COL_{i}")).collect();
        Self {
            columns,
            column_names,
            num_rows: 0,
        }
    }

    /// Creates a new `ColumnarBuffer` with explicit column names and pre-allocated capacity.
    #[must_use]
    pub fn with_names(column_names: Vec<String>, capacity: usize) -> Self {
        let num_columns = column_names.len();
        let columns = (0..num_columns)
            .map(|_| Vec::with_capacity(capacity))
            .collect();
        Self {
            columns,
            column_names,
            num_rows: 0,
        }
    }

    /// Constructs a `ColumnarBuffer` directly from raw columns, names, and row count.
    #[must_use]
    pub fn from_raw_parts(
        columns: Vec<Vec<f64>>,
        column_names: Vec<String>,
        num_rows: usize,
    ) -> Self {
        Self {
            columns,
            column_names,
            num_rows,
        }
    }

    /// Returns the number of completed rows in the buffer.
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Returns the number of columns in the buffer.
    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns `true` if the buffer contains 0 rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Returns the shape of the buffer as `(num_rows, num_columns)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows, self.columns.len())
    }

    /// Returns the list of column names / mnemonics.
    #[must_use]
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Returns a slice of the column at the given index, if within bounds.
    #[must_use]
    pub fn column(&self, index: usize) -> Option<&[f64]> {
        self.columns.get(index).map(Vec::as_slice)
    }

    /// Returns a mutable reference to the column vector at the given index.
    pub fn column_mut(&mut self, index: usize) -> Option<&mut Vec<f64>> {
        self.columns.get_mut(index)
    }

    /// Finds the index of a column matching `name` (case-insensitive).
    #[must_use]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        let trimmed = name.trim();
        self.column_names
            .iter()
            .position(|c| c.eq_ignore_ascii_case(trimmed))
    }

    /// Returns a slice of the column matching `name` (case-insensitive).
    #[must_use]
    pub fn column_by_name(&self, name: &str) -> Option<&[f64]> {
        self.column_index(name).and_then(|idx| self.column(idx))
    }

    /// Returns all column vectors as a slice of vectors.
    #[must_use]
    pub fn columns(&self) -> &[Vec<f64>] {
        &self.columns
    }

    /// Consumes the buffer and returns the inner column vectors.
    #[must_use]
    pub fn into_columns(self) -> Vec<Vec<f64>> {
        self.columns
    }

    /// Returns a row across all columns as a new `Vec<f64>`.
    #[must_use]
    pub fn row(&self, row_idx: usize) -> Option<Vec<f64>> {
        if row_idx >= self.num_rows {
            return None;
        }
        Some(self.columns.iter().map(|col| col[row_idx]).collect())
    }

    /// Appends a complete row across all columns.
    pub fn push_row(&mut self, row: &[f64]) -> Result<(), DataError> {
        if row.len() != self.columns.len() {
            return Err(DataError::ColumnCountMismatch {
                line_number: self.num_rows + 1,
                expected: self.columns.len(),
                found: row.len(),
            });
        }
        for (col, &val) in self.columns.iter_mut().zip(row.iter()) {
            col.push(val);
        }
        self.num_rows += 1;
        Ok(())
    }

    /// Returns the column data as a contiguous slice of raw bytes (IEEE 754 64-bit little-endian floats).
    ///
    /// This provides instant zero-copy integration with Apache Arrow buffers and NumPy `frombuffer`.
    #[must_use]
    #[allow(unsafe_code)]
    pub fn column_bytes(&self, col_idx: usize) -> Option<&[u8]> {
        self.column(col_idx).map(|slice| {
            // SAFETY: `slice` points to a valid, initialized contiguous `[f64]`.
            // Casting an aligned `*const f64` to `*const u8` is always sound, and the total
            // length in bytes is exactly `slice.len() * std::mem::size_of::<f64>()`.
            unsafe {
                std::slice::from_raw_parts(
                    slice.as_ptr().cast::<u8>(),
                    slice.len().saturating_mul(std::mem::size_of::<f64>()),
                )
            }
        })
    }

    /// Returns the raw memory address pointer of the column's first element.
    #[must_use]
    pub fn column_ptr(&self, col_idx: usize) -> Option<*const f64> {
        self.column(col_idx).map(<[f64]>::as_ptr)
    }

    /// Concatenates all columns sequentially into a single flat column-major vector.
    #[must_use]
    pub fn to_flat_column_major(&self) -> Vec<f64> {
        let total_len = self.num_rows.saturating_mul(self.columns.len());
        let mut flat = Vec::with_capacity(total_len);
        for col in &self.columns {
            flat.extend_from_slice(col);
        }
        flat
    }

    /// Interleaves all columns into a single flat row-major vector `[r0c0, r0c1, ..., r1c0, r1c1, ...]`.
    #[must_use]
    pub fn to_flat_row_major(&self) -> Vec<f64> {
        let total_len = self.num_rows.saturating_mul(self.columns.len());
        let mut flat = Vec::with_capacity(total_len);
        for r in 0..self.num_rows {
            for col in &self.columns {
                flat.push(col[r]);
            }
        }
        flat
    }

    /// Counts the number of `NaN` entries in the specified column.
    #[must_use]
    pub fn null_count(&self, col_idx: usize) -> usize {
        self.column(col_idx)
            .map_or(0, |col| col.iter().filter(|val| val.is_nan()).count())
    }

    /// Computes the minimum and maximum of non-NaN values in the specified column.
    #[must_use]
    pub fn min_max(&self, col_idx: usize) -> Option<(f64, f64)> {
        let col = self.column(col_idx)?;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut found = false;

        for &val in col {
            if !val.is_nan() {
                found = true;
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
            }
        }

        if found {
            Some((min, max))
        } else {
            None
        }
    }

    /// Computes the arithmetic mean of non-NaN values in the specified column.
    #[must_use]
    pub fn mean(&self, col_idx: usize) -> Option<f64> {
        let col = self.column(col_idx)?;
        let mut sum = 0.0;
        let mut count = 0usize;

        for &val in col {
            if !val.is_nan() {
                sum += val;
                count += 1;
            }
        }

        if count > 0 {
            Some(sum / (count as f64))
        } else {
            None
        }
    }
}

/// Configuration options and execution engine for parsing ASCII columnar log data.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnarParser {
    num_columns: usize,
    column_names: Vec<String>,
    wrapped: bool,
    strict: bool,
    null_sentinel: Option<NullSentinel>,
    capacity_hint: Option<usize>,
    allow_commas: bool,
}

impl ColumnarParser {
    /// Creates a new `ColumnarParser` configured for `num_columns`.
    #[must_use]
    pub fn new(num_columns: usize) -> Self {
        Self {
            num_columns,
            column_names: Vec::new(),
            wrapped: false,
            strict: true,
            null_sentinel: None,
            capacity_hint: None,
            allow_commas: false,
        }
    }

    /// Sets the curve / column names.
    #[must_use]
    pub fn with_column_names(mut self, names: Vec<String>) -> Self {
        if self.num_columns == 0 {
            self.num_columns = names.len();
        }
        self.column_names = names;
        self
    }

    /// Sets whether wrapped data mode (`WRAP = YES`) is enabled.
    #[must_use]
    pub const fn with_wrapped(mut self, wrapped: bool) -> Self {
        self.wrapped = wrapped;
        self
    }

    /// Sets strict validation mode (checks column count per line and at EOF).
    #[must_use]
    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Sets the dynamic NULL sentinel for mapping missing values to `f64::NAN`.
    #[must_use]
    pub fn with_null_sentinel(mut self, sentinel: NullSentinel) -> Self {
        self.null_sentinel = Some(sentinel);
        self
    }

    /// Sets an explicit pre-allocation capacity hint (number of rows).
    #[must_use]
    pub const fn with_capacity_hint(mut self, capacity: usize) -> Self {
        self.capacity_hint = Some(capacity);
        self
    }

    /// Sets whether comma delimiters (`,` ) are permitted in addition to whitespace.
    #[must_use]
    pub const fn with_allow_commas(mut self, allow_commas: bool) -> Self {
        self.allow_commas = allow_commas;
        self
    }

    /// Returns the number of configured columns.
    #[must_use]
    pub const fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Returns the column names.
    #[must_use]
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Returns whether wrapped mode is enabled.
    #[must_use]
    pub const fn is_wrapped(&self) -> bool {
        self.wrapped
    }

    /// Returns whether strict mode is enabled.
    #[must_use]
    pub const fn is_strict(&self) -> bool {
        self.strict
    }

    /// Estimates row pre-allocation capacity for a given byte buffer length.
    #[must_use]
    pub fn estimate_capacity(&self, bytes_len: usize) -> usize {
        self.capacity_hint
            .unwrap_or_else(|| estimate_row_capacity(bytes_len, self.num_columns))
    }

    /// Parses ASCII log data directly from a borrowed byte slice (`&[u8]`).
    ///
    /// # Performance
    /// - Zero allocations for tokens, lines, or intermediate row structs.
    /// - Pre-allocated contiguous `Vec<f64>` column vectors.
    /// - Exact floating-point parsing via `fast_float`.
    ///
    /// # Errors
    /// Returns `DataError::NoColumnsDefined` if `num_columns == 0`.
    /// Returns `DataError::ColumnCountMismatch` if an unwrapped line does not match `num_columns`.
    /// Returns `DataError::InvalidFloat` if an unparseable numeric token is encountered.
    /// Returns `DataError::IncompleteRow` if EOF is reached with a partial row.
    /// Returns `DataError::EmptyData` if strict mode is enabled and no rows were parsed.
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<ColumnarBuffer, DataError> {
        if self.num_columns == 0 {
            return Err(DataError::NoColumnsDefined);
        }

        if self.wrapped {
            let mut unwrapper = crate::unwrapper::StreamingUnwrapper::new(bytes, self.num_columns)
                .with_column_names(self.column_names.clone())
                .with_strict(self.strict)
                .with_allow_commas(self.allow_commas);
            if let Some(ref sentinel) = self.null_sentinel {
                unwrapper = unwrapper.with_null_sentinel(sentinel.clone());
            }
            return unwrapper.parse_to_columnar();
        }

        let capacity = self.estimate_capacity(bytes.len());
        let mut columns: Vec<Vec<f64>> = (0..self.num_columns)
            .map(|_| Vec::with_capacity(capacity))
            .collect();

        let mut pos = 0usize;
        let len = bytes.len();
        let mut line_number = 1usize;
        let mut col_idx = 0usize;
        let mut values_on_line = 0usize;
        let mut at_line_start = true;

        while pos < len {
            let b = bytes[pos];

            // 1. Delimiters: spaces and tabs
            if b == b' ' || b == b'\t' {
                pos += 1;
                continue;
            }

            // 2. Optional comma delimiter
            if self.allow_commas && b == b',' {
                pos += 1;
                continue;
            }

            // 3. Newline: \n
            if b == b'\n' {
                pos += 1;
                if values_on_line > 0 {
                    if !self.wrapped {
                        if self.strict {
                            if values_on_line != self.num_columns {
                                return Err(DataError::ColumnCountMismatch {
                                    line_number,
                                    expected: self.num_columns,
                                    found: values_on_line,
                                });
                            }
                        } else if values_on_line < self.num_columns {
                            for col in columns
                                .iter_mut()
                                .take(self.num_columns)
                                .skip(values_on_line)
                            {
                                col.push(f64::NAN);
                            }
                            col_idx = 0;
                        }
                    }
                    values_on_line = 0;
                }
                line_number += 1;
                at_line_start = true;
                continue;
            }

            // 4. Newline: \r or \r\n
            if b == b'\r' {
                pos += 1;
                if pos < len && bytes[pos] == b'\n' {
                    pos += 1;
                }
                if values_on_line > 0 {
                    if !self.wrapped {
                        if self.strict {
                            if values_on_line != self.num_columns {
                                return Err(DataError::ColumnCountMismatch {
                                    line_number,
                                    expected: self.num_columns,
                                    found: values_on_line,
                                });
                            }
                        } else if values_on_line < self.num_columns {
                            for col in columns
                                .iter_mut()
                                .take(self.num_columns)
                                .skip(values_on_line)
                            {
                                col.push(f64::NAN);
                            }
                            col_idx = 0;
                        }
                    }
                    values_on_line = 0;
                }
                line_number += 1;
                at_line_start = true;
                continue;
            }

            // 5. Comment line: # skips to end of line
            if b == b'#' {
                pos += 1;
                while pos < len && bytes[pos] != b'\n' && bytes[pos] != b'\r' {
                    pos += 1;
                }
                continue;
            }

            // 6. Delimiter line starting with ~ terminates data section (e.g. ~Other after ~A)
            if b == b'~' && at_line_start {
                break;
            }

            // 7. Numeric token window
            let token_start = pos;
            while pos < len {
                let tb = bytes[pos];
                if tb == b' '
                    || tb == b'\t'
                    || tb == b'\r'
                    || tb == b'\n'
                    || tb == b'#'
                    || (self.allow_commas && tb == b',')
                {
                    break;
                }
                pos += 1;
            }

            let token_bytes = &bytes[token_start..pos];
            at_line_start = false;

            // Direct parse using fast_float Eisel-Lemire algorithm
            let mut val = match fast_float::parse::<f64, _>(token_bytes) {
                Ok(v) => v,
                Err(_) => {
                    // Check if token represents null/nan text
                    if is_text_null_sentinel(token_bytes, self.null_sentinel.as_ref()) {
                        f64::NAN
                    } else {
                        let col_name = self.column_names.get(col_idx).cloned();
                        let token_str = String::from_utf8_lossy(token_bytes).into_owned();
                        return Err(DataError::InvalidFloat {
                            line_number,
                            column_index: col_idx,
                            column_name: col_name,
                            token: token_str,
                            reason: "failed to parse floating-point value".to_string(),
                        });
                    }
                }
            };

            // Map null sentinel to IEEE 754 NaN
            if !val.is_nan() {
                if let Some(ref sentinel) = self.null_sentinel {
                    if sentinel.is_null(val) {
                        val = f64::NAN;
                    }
                }
            }

            // Write directly into contiguous column vector
            columns[col_idx].push(val);
            col_idx += 1;
            values_on_line += 1;

            if col_idx == self.num_columns {
                col_idx = 0;
            }
        }

        // Handle last line without trailing newline
        if values_on_line > 0 && !self.wrapped {
            if self.strict {
                if values_on_line != self.num_columns {
                    return Err(DataError::ColumnCountMismatch {
                        line_number,
                        expected: self.num_columns,
                        found: values_on_line,
                    });
                }
            } else if values_on_line < self.num_columns {
                for col in columns
                    .iter_mut()
                    .take(self.num_columns)
                    .skip(values_on_line)
                {
                    col.push(f64::NAN);
                }
                col_idx = 0;
            }
        }

        // Check for incomplete row at EOF
        if col_idx != 0 {
            if self.strict {
                return Err(DataError::IncompleteRow {
                    expected: self.num_columns,
                    found: col_idx,
                });
            }
            // Truncate partial row in lenient mode so all columns remain equal length
            for col in columns.iter_mut().take(col_idx) {
                col.pop();
            }
        }

        let num_rows = if self.num_columns > 0 && !columns.is_empty() {
            columns[0].len()
        } else {
            0
        };

        if self.strict && num_rows == 0 {
            return Err(DataError::EmptyData);
        }

        let names = if self.column_names.len() == self.num_columns {
            self.column_names.clone()
        } else {
            (0..self.num_columns).map(|i| format!("COL_{i}")).collect()
        };

        Ok(ColumnarBuffer {
            columns,
            column_names: names,
            num_rows,
        })
    }

    /// Parses ASCII log data from a string slice (`&str`).
    pub fn parse_str(&self, text: &str) -> Result<ColumnarBuffer, DataError> {
        self.parse_bytes(text.as_bytes())
    }
}

/// Standalone convenience function to parse columnar data with default options.
pub fn parse_columnar(bytes: &[u8], num_columns: usize) -> Result<ColumnarBuffer, DataError> {
    ColumnarParser::new(num_columns).parse_bytes(bytes)
}

/// Standalone convenience function to parse columnar data with a declared NULL sentinel.
pub fn parse_columnar_with_sentinel(
    bytes: &[u8],
    num_columns: usize,
    sentinel: Option<NullSentinel>,
) -> Result<ColumnarBuffer, DataError> {
    let mut parser = ColumnarParser::new(num_columns);
    if let Some(s) = sentinel {
        parser = parser.with_null_sentinel(s);
    }
    parser.parse_bytes(bytes)
}

// ---------------------------------------------------------------------------
// PyO3 Python Bindings
// ---------------------------------------------------------------------------

/// Python wrapper for `ColumnarBuffer`.
#[pyclass(name = "ColumnarBuffer")]
#[derive(Debug, Clone)]
pub struct PyColumnarBuffer {
    pub(crate) inner: ColumnarBuffer,
}

impl PyColumnarBuffer {
    #[must_use]
    pub fn from_inner(inner: ColumnarBuffer) -> Self {
        Self { inner }
    }

    fn resolve_index(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(idx) = key.extract::<isize>() {
            let num_cols = self.inner.num_columns() as isize;
            let normalized = if idx < 0 { idx + num_cols } else { idx };
            if normalized >= 0 && (normalized as usize) < self.inner.num_columns() {
                Ok(normalized as usize)
            } else {
                Err(LASDataError::new_err(format!(
                    "Column index {idx} out of range (buffer has {} columns)",
                    self.inner.num_columns()
                )))
            }
        } else if let Ok(name) = key.extract::<&str>() {
            self.inner.column_index(name).ok_or_else(|| {
                LASDataError::new_err(format!(
                    "Column mnemonic '{name}' not found; available columns: {:?}",
                    self.inner.column_names()
                ))
            })
        } else {
            Err(LASDataError::new_err(
                "Column identifier must be an integer index or string mnemonic",
            ))
        }
    }
}

#[pymethods]
impl PyColumnarBuffer {
    /// Number of completed rows in the columnar buffer.
    #[getter]
    pub fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    /// Number of curves / columns in the buffer.
    #[getter]
    pub fn num_columns(&self) -> usize {
        self.inner.num_columns()
    }

    /// List of curve mnemonics / column names.
    #[getter]
    pub fn column_names(&self) -> Vec<String> {
        self.inner.column_names().to_vec()
    }

    /// Shape of the buffer as a tuple `(num_rows, num_columns)`.
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Whether the buffer contains 0 rows.
    #[getter]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a column as a Python list of floats by integer index or string mnemonic.
    #[pyo3(signature = (key))]
    pub fn column(&self, key: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let idx = self.resolve_index(key)?;
        let col = self
            .inner
            .column(idx)
            .ok_or_else(|| LASDataError::new_err(format!("Column index {idx} out of range")))?;
        Ok(col.to_vec())
    }

    /// Returns a complete row across all columns by row index.
    #[pyo3(signature = (row_idx))]
    pub fn row(&self, row_idx: usize) -> PyResult<Vec<f64>> {
        self.inner.row(row_idx).ok_or_else(|| {
            LASDataError::new_err(format!(
                "Row index {row_idx} out of range (buffer has {} rows)",
                self.inner.num_rows()
            ))
        })
    }

    /// Returns a single cell float value at `(row_idx, col_key)`.
    #[pyo3(signature = (row_idx, col_key))]
    pub fn cell(&self, row_idx: usize, col_key: &Bound<'_, PyAny>) -> PyResult<f64> {
        let col_idx = self.resolve_index(col_key)?;
        let col = self
            .inner
            .column(col_idx)
            .ok_or_else(|| LASDataError::new_err(format!("Column index {col_idx} out of range")))?;
        col.get(row_idx).copied().ok_or_else(|| {
            LASDataError::new_err(format!(
                "Row index {row_idx} out of range (buffer has {} rows)",
                self.inner.num_rows()
            ))
        })
    }

    /// Exports all columns to a dictionary mapping column names to lists of floats.
    pub fn to_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut map = HashMap::with_capacity(self.inner.num_columns());
        for (idx, name) in self.inner.column_names().iter().enumerate() {
            if let Some(col) = self.inner.column(idx) {
                map.insert(name.clone(), col.to_vec());
            }
        }
        map
    }

    /// Counts the number of `NaN` / missing values in the specified column.
    #[pyo3(signature = (key))]
    pub fn null_count(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        let idx = self.resolve_index(key)?;
        Ok(self.inner.null_count(idx))
    }

    /// Computes `(min, max)` of non-NaN values in the specified column.
    #[pyo3(signature = (key))]
    pub fn min_max(&self, key: &Bound<'_, PyAny>) -> PyResult<Option<(f64, f64)>> {
        let idx = self.resolve_index(key)?;
        Ok(self.inner.min_max(idx))
    }

    /// Computes the arithmetic mean of non-NaN values in the specified column.
    #[pyo3(signature = (key))]
    pub fn mean(&self, key: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
        let idx = self.resolve_index(key)?;
        Ok(self.inner.mean(idx))
    }

    /// Returns a flat vector of all column values in column-major order.
    pub fn to_flat_column_major(&self) -> Vec<f64> {
        self.inner.to_flat_column_major()
    }

    /// Returns a flat vector of all row values in row-major order.
    pub fn to_flat_row_major(&self) -> Vec<f64> {
        self.inner.to_flat_row_major()
    }

    /// Returns the raw memory address pointer of the column vector.
    #[pyo3(signature = (key))]
    pub fn column_ptr(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        let col_idx = self.resolve_index(key)?;
        let ptr = self
            .inner
            .column_ptr(col_idx)
            .ok_or_else(|| LASDataError::new_err(format!("Column index {col_idx} out of range")))?;
        Ok(ptr as usize)
    }

    /// Returns the column data as a Python `bytes` object (IEEE 754 64-bit little-endian floats).
    ///
    /// This enables instantaneous zero-copy view creation with `numpy.frombuffer(buf, dtype=np.float64)`
    /// or `pyarrow.py_buffer(buf)`.
    #[pyo3(signature = (key))]
    pub fn column_as_bytes<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let col_idx = self.resolve_index(key)?;
        let slice = self
            .inner
            .column_bytes(col_idx)
            .ok_or_else(|| LASDataError::new_err(format!("Column index {col_idx} out of range")))?;
        Ok(PyBytes::new(py, slice))
    }

    /// Returns the full columnar matrix as a flat `bytes` buffer.
    #[pyo3(signature = (column_major = true))]
    #[allow(unsafe_code)]
    pub fn to_flat_bytes<'py>(&self, py: Python<'py>, column_major: bool) -> Bound<'py, PyBytes> {
        let flat = if column_major {
            self.inner.to_flat_column_major()
        } else {
            self.inner.to_flat_row_major()
        };
        // SAFETY: `flat` is a valid Vec<f64>, converting to &[u8] for PyBytes
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                flat.as_ptr().cast::<u8>(),
                flat.len().saturating_mul(std::mem::size_of::<f64>()),
            )
        };
        PyBytes::new(py, byte_slice)
    }

    /// Returns the `__array_interface__` dictionary for zero-copy NumPy array creation.
    #[pyo3(signature = (key))]
    pub fn column_array_interface(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let col_idx = self.resolve_index(key)?;
        let slice = self
            .inner
            .column(col_idx)
            .ok_or_else(|| LASDataError::new_err(format!("Column index {col_idx} out of range")))?;
        let ptr = slice.as_ptr() as usize;
        let dict = PyDict::new(py);
        dict.set_item("shape", (slice.len(),))?;
        dict.set_item("typestr", "<f8")?;
        dict.set_item("data", (ptr, true))?;
        dict.set_item("version", 3)?;
        Ok(dict.into())
    }

    /// Exports to a 2D NumPy array `(num_rows, num_columns)` or dictionary of 1D arrays.
    #[pyo3(signature = (as_dict = false))]
    pub fn to_numpy(&self, py: Python<'_>, as_dict: bool) -> PyResult<PyObject> {
        let Ok(np) = py.import("numpy") else {
            return Err(pyo3::exceptions::PyImportError::new_err(
                "numpy is required for to_numpy(); install with 'pip install numpy'",
            ));
        };

        let dtype = np.getattr("float64")?;

        if as_dict {
            let dict = PyDict::new(py);
            for (col_idx, name) in self.inner.column_names().iter().enumerate() {
                if let Some(bytes) = self.inner.column_bytes(col_idx) {
                    let py_bytes = PyBytes::new(py, bytes);
                    let arr = np.call_method1("frombuffer", (py_bytes, &dtype))?;
                    dict.set_item(name, arr)?;
                }
            }
            return Ok(dict.into());
        }

        let col_arrays = PyList::empty(py);
        for col_idx in 0..self.inner.num_columns() {
            if let Some(bytes) = self.inner.column_bytes(col_idx) {
                let py_bytes = PyBytes::new(py, bytes);
                let arr = np.call_method1("frombuffer", (py_bytes, &dtype))?;
                col_arrays.append(arr)?;
            }
        }
        let matrix = np.call_method1("column_stack", (col_arrays,))?;
        Ok(matrix.into())
    }

    /// Exports to an Apache Arrow `Table` via zero-copy buffer sharing.
    #[pyo3(signature = ())]
    pub fn to_arrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        let Ok(pa) = py.import("pyarrow") else {
            return Err(pyo3::exceptions::PyImportError::new_err(
                "pyarrow is required for to_arrow(); install with 'pip install pyarrow'",
            ));
        };

        let arrays = PyList::empty(py);
        let num_rows = self.inner.num_rows();

        for col_idx in 0..self.inner.num_columns() {
            if let Some(bytes) = self.inner.column_bytes(col_idx) {
                let py_bytes = PyBytes::new(py, bytes);
                let py_buf = pa.call_method1("py_buffer", (py_bytes,))?;
                let float64_type = pa.call_method0("float64")?;
                let none_val = py.None();
                let buffers = (none_val, py_buf);
                let arr = pa
                    .getattr("Array")?
                    .call_method1("from_buffers", (float64_type, num_rows, buffers))?;
                arrays.append(arr)?;
            }
        }

        let names = self.inner.column_names();
        let table = pa
            .getattr("Table")?
            .call_method1("from_arrays", (arrays, names))?;

        Ok(table.into())
    }

    /// Exports to a Polars `DataFrame`.
    #[pyo3(signature = ())]
    pub fn to_polars(&self, py: Python<'_>) -> PyResult<PyObject> {
        let Ok(pl) = py.import("polars") else {
            return Err(pyo3::exceptions::PyImportError::new_err(
                "polars is required for to_polars(); install with 'pip install polars'",
            ));
        };
        let arrow_table = self.to_arrow(py)?;
        let df = pl.call_method1("from_arrow", (arrow_table,))?;
        Ok(df.into())
    }

    /// Exports to a Pandas `DataFrame`.
    #[pyo3(signature = ())]
    pub fn to_pandas(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Ok(arrow_table) = self.to_arrow(py) {
            if let Ok(df) = arrow_table.call_method0(py, "to_pandas") {
                return Ok(df);
            }
        }
        let Ok(pd) = py.import("pandas") else {
            return Err(pyo3::exceptions::PyImportError::new_err(
                "pandas or pyarrow is required for to_pandas(); install with 'pip install pandas'",
            ));
        };
        let dict = self.to_dict();
        let df = pd.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    pub fn __len__(&self) -> usize {
        self.inner.num_rows()
    }

    pub fn __contains__(&self, name: &str) -> bool {
        self.inner.column_index(name).is_some()
    }

    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        self.column(key)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "<ColumnarBuffer rows={} cols={} columns={:?}>",
            self.inner.num_rows(),
            self.inner.num_columns(),
            self.inner.column_names()
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python wrapper for `ColumnarParser`.
#[pyclass(name = "ColumnarParser")]
#[derive(Debug, Clone)]
pub struct PyColumnarParser {
    inner: ColumnarParser,
}

#[pymethods]
impl PyColumnarParser {
    /// Creates a new `ColumnarParser` instance.
    #[new]
    #[pyo3(signature = (
        num_columns = 0,
        column_names = None,
        wrapped = false,
        strict = true,
        null_sentinel = None,
        capacity_hint = None,
        allow_commas = false
    ))]
    pub fn new(
        num_columns: usize,
        column_names: Option<Vec<String>>,
        wrapped: bool,
        strict: bool,
        null_sentinel: Option<NullSentinel>,
        capacity_hint: Option<usize>,
        allow_commas: bool,
    ) -> Self {
        let actual_cols = if let Some(ref names) = column_names {
            if num_columns == 0 {
                names.len()
            } else {
                num_columns
            }
        } else {
            num_columns
        };

        let mut parser = ColumnarParser::new(actual_cols)
            .with_wrapped(wrapped)
            .with_strict(strict)
            .with_allow_commas(allow_commas);

        if let Some(names) = column_names {
            parser = parser.with_column_names(names);
        }
        if let Some(s) = null_sentinel {
            parser = parser.with_null_sentinel(s);
        }
        if let Some(cap) = capacity_hint {
            parser = parser.with_capacity_hint(cap);
        }

        Self { inner: parser }
    }

    /// Parses ASCII log data from `bytes`, `str`, `memoryview`, or `DecodedBuffer`.
    pub fn parse(&self, data: &Bound<'_, PyAny>) -> PyResult<PyColumnarBuffer> {
        if let Ok(py_bytes) = data.extract::<&[u8]>() {
            let result = self.inner.parse_bytes(py_bytes)?;
            return Ok(PyColumnarBuffer { inner: result });
        }

        if let Ok(py_str) = data.extract::<&str>() {
            let result = self.inner.parse_str(py_str)?;
            return Ok(PyColumnarBuffer { inner: result });
        }

        if let Ok(decoded) = data.extract::<pyo3::PyRef<'_, PyDecodedBuffer>>() {
            let result = self.inner.parse_str(decoded.text())?;
            return Ok(PyColumnarBuffer { inner: result });
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected bytes, str, memoryview, or DecodedBuffer",
        ))
    }

    /// Estimates row capacity for pre-allocating column vectors.
    #[pyo3(signature = (bytes_len))]
    pub fn estimate_capacity(&self, bytes_len: usize) -> usize {
        self.inner.estimate_capacity(bytes_len)
    }

    /// Configured column count.
    #[getter]
    pub fn num_columns(&self) -> usize {
        self.inner.num_columns()
    }

    /// Configured column names.
    #[getter]
    pub fn column_names(&self) -> Vec<String> {
        self.inner.column_names().to_vec()
    }

    /// Whether wrapped mode is enabled.
    #[getter]
    pub fn is_wrapped(&self) -> bool {
        self.inner.is_wrapped()
    }

    /// Whether strict mode is enabled.
    #[getter]
    pub fn is_strict(&self) -> bool {
        self.inner.is_strict()
    }

    fn __repr__(&self) -> String {
        format!(
            "<ColumnarParser cols={} wrapped={} strict={}>",
            self.inner.num_columns(),
            self.inner.is_wrapped(),
            self.inner.is_strict()
        )
    }
}

/// Standalone Python function to parse delimited ASCII numeric log data into a `ColumnarBuffer`.
#[pyfunction(name = "parse_columnar")]
#[pyo3(signature = (
    data,
    num_columns = 0,
    column_names = None,
    wrapped = false,
    strict = true,
    null_sentinel = None,
    capacity_hint = None,
    allow_commas = false
))]
#[allow(clippy::too_many_arguments)]
pub fn py_parse_columnar(
    data: &Bound<'_, PyAny>,
    num_columns: usize,
    column_names: Option<Vec<String>>,
    wrapped: bool,
    strict: bool,
    null_sentinel: Option<NullSentinel>,
    capacity_hint: Option<usize>,
    allow_commas: bool,
) -> PyResult<PyColumnarBuffer> {
    let parser = PyColumnarParser::new(
        num_columns,
        column_names,
        wrapped,
        strict,
        null_sentinel,
        capacity_hint,
        allow_commas,
    );
    parser.parse(data)
}

/// Standalone Python function to estimate row capacity based on byte buffer length.
#[pyfunction(name = "estimate_row_capacity")]
#[pyo3(signature = (bytes_len, num_columns))]
#[must_use]
pub fn py_estimate_row_capacity(bytes_len: usize, num_columns: usize) -> usize {
    estimate_row_capacity(bytes_len, num_columns)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_row_capacity() {
        assert_eq!(estimate_row_capacity(0, 4), 0);
        assert_eq!(estimate_row_capacity(1000, 0), 0);
        let cap = estimate_row_capacity(100_000, 4);
        assert!(cap >= 2500);
    }

    #[test]
    fn test_parse_basic_unwrapped() {
        let data = b" 1000.0   25.5   120.3\n 1000.5   26.0   121.0\n 1001.0   27.2   122.5\n";
        let parser = ColumnarParser::new(3).with_column_names(vec![
            "DEPT".to_string(),
            "GR".to_string(),
            "RHOB".to_string(),
        ]);
        let buf = parser.parse_bytes(data).unwrap();

        assert_eq!(buf.num_rows(), 3);
        assert_eq!(buf.num_columns(), 3);
        assert_eq!(buf.shape(), (3, 3));
        assert_eq!(buf.column(0).unwrap(), &[1000.0, 1000.5, 1001.0]);
        assert_eq!(buf.column(1).unwrap(), &[25.5, 26.0, 27.2]);
        assert_eq!(buf.column(2).unwrap(), &[120.3, 121.0, 122.5]);
        assert_eq!(
            buf.column_by_name("dept").unwrap(),
            &[1000.0, 1000.5, 1001.0]
        );
        assert_eq!(buf.row(0).unwrap(), vec![1000.0, 25.5, 120.3]);
    }

    #[test]
    fn test_parse_scientific_notation_and_signs() {
        let data = b" 1.234e-03  -4.56E+02  +0.0\n -1.5e2  .75  -.25\n";
        let buf = ColumnarParser::new(3).parse_bytes(data).unwrap();

        assert_eq!(buf.num_rows(), 2);
        assert_eq!(buf.column(0).unwrap()[0], 1.234e-3);
        assert_eq!(buf.column(1).unwrap()[0], -456.0);
        assert_eq!(buf.column(2).unwrap()[0], 0.0);
        assert_eq!(buf.column(0).unwrap()[1], -150.0);
        assert_eq!(buf.column(1).unwrap()[1], 0.75);
        assert_eq!(buf.column(2).unwrap()[1], -0.25);
    }

    #[test]
    fn test_null_sentinel_mapping() {
        let data = b" 1000.0  -999.25  50.0\n 1000.5  NaN  null\n 1001.0  None  -nan\n";
        let sentinel = NullSentinel::default_cwls();
        let parser = ColumnarParser::new(3).with_null_sentinel(sentinel);
        let buf = parser.parse_bytes(data).unwrap();

        assert_eq!(buf.num_rows(), 3);
        assert_eq!(buf.column(0).unwrap(), &[1000.0, 1000.5, 1001.0]);
        assert!(buf.column(1).unwrap()[0].is_nan());
        assert!(buf.column(1).unwrap()[1].is_nan());
        assert!(buf.column(1).unwrap()[2].is_nan());
        assert_eq!(buf.column(2).unwrap()[0], 50.0);
        assert!(buf.column(2).unwrap()[1].is_nan());
        assert!(buf.column(2).unwrap()[2].is_nan());
        assert_eq!(buf.null_count(1), 3);
    }

    #[test]
    fn test_wrapped_mode() {
        // 5 columns, wrapped across 2 lines per row
        let data = b" 1000.0 25.5 120.3\n 0.45 2.65\n 1000.5 26.0 121.0\n 0.46 2.66\n";
        let parser = ColumnarParser::new(5).with_wrapped(true);
        let buf = parser.parse_bytes(data).unwrap();

        assert_eq!(buf.num_rows(), 2);
        assert_eq!(buf.num_columns(), 5);
        assert_eq!(buf.column(0).unwrap(), &[1000.0, 1000.5]);
        assert_eq!(buf.column(1).unwrap(), &[25.5, 26.0]);
        assert_eq!(buf.column(2).unwrap(), &[120.3, 121.0]);
        assert_eq!(buf.column(3).unwrap(), &[0.45, 0.46]);
        assert_eq!(buf.column(4).unwrap(), &[2.65, 2.66]);
    }

    #[test]
    fn test_ragged_row_detected_in_strict_unwrapped() {
        let data = b" 1000.0 25.5 120.3\n 1000.5 26.0\n 1001.0 27.0 122.0\n";
        let parser = ColumnarParser::new(3).with_strict(true);
        let err = parser.parse_bytes(data).unwrap_err();
        match err {
            DataError::ColumnCountMismatch {
                line_number,
                expected,
                found,
            } => {
                assert_eq!(line_number, 2);
                assert_eq!(expected, 3);
                assert_eq!(found, 2);
            }
            _ => panic!("Expected ColumnCountMismatch, got {err:?}"),
        }
    }

    #[test]
    fn test_lenient_mode_pads_missing_columns() {
        let data = b" 1000.0 25.5 120.3\n 1000.5 26.0\n 1001.0 27.0 122.0\n";
        let parser = ColumnarParser::new(3).with_strict(false);
        let buf = parser.parse_bytes(data).unwrap();

        assert_eq!(buf.num_rows(), 3);
        assert_eq!(buf.column(0).unwrap(), &[1000.0, 1000.5, 1001.0]);
        assert_eq!(buf.column(1).unwrap(), &[25.5, 26.0, 27.0]);
        assert_eq!(buf.column(2).unwrap()[0], 120.3);
        assert!(buf.column(2).unwrap()[1].is_nan());
        assert_eq!(buf.column(2).unwrap()[2], 122.0);
    }

    #[test]
    fn test_comments_and_blank_lines_skipped() {
        let data = b"# Comment header\n\n 1000.0 25.5\n# Mid comment\n\n 1000.5 26.0\r\n";
        let buf = ColumnarParser::new(2).parse_bytes(data).unwrap();
        assert_eq!(buf.num_rows(), 2);
        assert_eq!(buf.column(0).unwrap(), &[1000.0, 1000.5]);
        assert_eq!(buf.column(1).unwrap(), &[25.5, 26.0]);
    }

    #[test]
    fn test_trailing_section_terminates_ascii() {
        let data = b" 1000.0 25.0\n 1000.5 26.0\n~Other Information\nThis should be ignored";
        let buf = ColumnarParser::new(2).parse_bytes(data).unwrap();
        assert_eq!(buf.num_rows(), 2);
        assert_eq!(buf.column(0).unwrap(), &[1000.0, 1000.5]);
    }

    #[test]
    fn test_empty_data() {
        let data = b"# Empty file with comment only\n";
        let strict_err = ColumnarParser::new(2)
            .with_strict(true)
            .parse_bytes(data)
            .unwrap_err();
        assert_eq!(strict_err, DataError::EmptyData);

        let lenient_buf = ColumnarParser::new(2)
            .with_strict(false)
            .parse_bytes(data)
            .unwrap();
        assert_eq!(lenient_buf.num_rows(), 0);
        assert!(lenient_buf.is_empty());
    }

    #[test]
    fn test_flat_vector_exports() {
        let data = b" 1.0 2.0\n 3.0 4.0\n";
        let buf = ColumnarParser::new(2).parse_bytes(data).unwrap();

        assert_eq!(buf.to_flat_column_major(), vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(buf.to_flat_row_major(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_column_statistics() {
        let data = b" 10.0 1.0\n 20.0 NaN\n 30.0 3.0\n";
        let buf = ColumnarParser::new(2).parse_bytes(data).unwrap();

        assert_eq!(buf.min_max(0), Some((10.0, 30.0)));
        assert_eq!(buf.mean(0), Some(20.0));
        assert_eq!(buf.null_count(1), 1);
        assert_eq!(buf.min_max(1), Some((1.0, 3.0)));
        assert_eq!(buf.mean(1), Some(2.0));
    }

    #[test]
    fn test_allow_commas() {
        let data = b" 100.0, 20.0, 30.0\n 101.0, 21.0, 31.0\n";
        let parser = ColumnarParser::new(3).with_allow_commas(true);
        let buf = parser.parse_bytes(data).unwrap();
        assert_eq!(buf.num_rows(), 2);
        assert_eq!(buf.column(0).unwrap(), &[100.0, 101.0]);
        assert_eq!(buf.column(1).unwrap(), &[20.0, 21.0]);
        assert_eq!(buf.column(2).unwrap(), &[30.0, 31.0]);
    }

    #[test]
    fn test_zero_copy_byte_views() {
        let data = b" 100.0 200.0\n";
        let buf = ColumnarParser::new(2).parse_bytes(data).unwrap();
        let bytes0 = buf.column_bytes(0).unwrap();
        assert_eq!(bytes0.len(), 8);
        let val = f64::from_le_bytes(bytes0.try_into().unwrap());
        assert_eq!(val, 100.0);
        assert!(!buf.column_ptr(0).unwrap().is_null());
    }

    #[test]
    fn test_invalid_float_error() {
        let data = b" 100.0 BAD_VAL 300.0\n";
        let parser = ColumnarParser::new(3).with_column_names(vec![
            "DEPT".to_string(),
            "GR".to_string(),
            "RHOB".to_string(),
        ]);
        let err = parser.parse_bytes(data).unwrap_err();
        match err {
            DataError::InvalidFloat {
                line_number,
                column_index,
                column_name,
                token,
                ..
            } => {
                assert_eq!(line_number, 1);
                assert_eq!(column_index, 1);
                assert_eq!(column_name, Some("GR".to_string()));
                assert_eq!(token, "BAD_VAL");
            }
            _ => panic!("Expected InvalidFloat, got {err:?}"),
        }
    }
}
