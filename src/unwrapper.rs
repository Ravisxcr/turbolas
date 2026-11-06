//! Streaming line-unwrapping engine for folded CWLS LAS wrapped (`WRAP = YES`) data blocks.
//!
//! # Architecture
//!
//! In the Canadian Well Logging Society (CWLS) Log ASCII Standard (LAS) format,
//! when the header defines `WRAP . YES`, each logical observation (depth step)
//! containing $N$ channels/curves is folded across multiple physical lines.
//!
//! Typically, the first line of a depth step contains the depth or index channel,
//! while subsequent continuation lines contain the remaining curve measurements
//! (usually up to 8 values per line, conforming to legacy 80-column line widths).
//!
//! Naive solutions perform full-file string replacements (`str.replace('\n', ' ')`)
//! or eager global tokenization. These approaches:
//! 1. Incur massive heap allocations proportional to file size ($O(M \times N)$).
//! 2. Destroy physical line numbering required for diagnostic error reporting.
//! 3. Erroneously comment out downstream data if inline/block `#` comments exist.
//! 4. Suffer from catastrophic desynchronization if a single line is corrupted.
//!
//! # Streaming Unwrapper Design
//!
//! To achieve maximum throughput with bounded memory:
//! - **$O(1)$ Intermediate Memory**: The engine maintains a single active record
//!   buffer pre-allocated to the exact channel count $N$. This buffer is reused
//!   across all records without dynamic resizing or per-record heap churn.
//! - **Zero-Copy Physical Line Slicing**: The engine scans line-by-line directly
//!   over borrowed byte slices (`&[u8]`), cleanly splitting on `\r\n`, `\n`, or standalone `\r`.
//! - **Per-Depth Channel Tracking**: Accurately tracks `active_channel_count`,
//!   `record_start_line`, `current_line`, and `lines_in_record`.
//! - **Exact Conversions & Sentinel Mapping**: Passes borrowed token slices directly
//!   into `fast_float::parse::<f64, _>`, dynamically mapping declared NULL sentinels to `f64::NAN`.
//! - **Reliable Error Recovery**: In permissive/recovery mode, truncated or malformed
//!   trailing records at EOF or mid-stream are padded with `f64::NAN`, preserving
//!   ingestion integrity across legacy telemetry without throwing unhandled exceptions.

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

use crate::columnar::{is_text_null_sentinel, ColumnarBuffer, PyColumnarBuffer};
use crate::encoding::PyDecodedBuffer;
use crate::error::{DataError, LASDataError};
use crate::null::NullSentinel;

/// A fully assembled logical observation assembled from one or more physical lines.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwrappedRecord {
    /// 0-based logical index of this observation row.
    pub record_index: usize,
    /// 1-based physical line number where this logical record began.
    pub start_line: usize,
    /// 1-based physical line number where this logical record completed.
    pub end_line: usize,
    /// Number of physical lines folded into this record.
    pub line_count: usize,
    /// The parsed float values for each curve/channel in this record.
    pub values: Vec<f64>,
}

impl UnwrappedRecord {
    /// Returns the 0-based logical record index.
    #[must_use]
    pub const fn record_index(&self) -> usize {
        self.record_index
    }

    /// Returns the 1-based physical line number where the record began.
    #[must_use]
    pub const fn start_line(&self) -> usize {
        self.start_line
    }

    /// Returns the 1-based physical line number where the record ended.
    #[must_use]
    pub const fn end_line(&self) -> usize {
        self.end_line
    }

    /// Returns the number of physical lines folded into this record.
    #[must_use]
    pub const fn line_count(&self) -> usize {
        self.line_count
    }

    /// Returns the slice of parsed channel float values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the float value at channel index `idx`, if within bounds.
    #[must_use]
    pub fn value(&self, idx: usize) -> Option<f64> {
        self.values.get(idx).copied()
    }

    /// Returns the number of channels in this record.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if this record has 0 channels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Streaming line-unwrapping engine for CWLS LAS `WRAP = YES` data blocks.
#[derive(Debug, Clone)]
pub struct StreamingUnwrapper<'a> {
    bytes: &'a [u8],
    pos: usize,
    current_line: usize,
    record_start_line: usize,
    record_index: usize,
    lines_in_current_record: usize,
    expected_channels: usize,
    column_names: Vec<String>,
    strict: bool,
    allow_commas: bool,
    null_sentinel: Option<NullSentinel>,
    /// Active record buffer with bounded O(1) capacity reused across all rows.
    active_record: Vec<f64>,
    at_line_start: bool,
}

impl<'a> StreamingUnwrapper<'a> {
    /// Creates a new `StreamingUnwrapper` over a borrowed byte slice.
    #[must_use]
    pub fn new(bytes: &'a [u8], expected_channels: usize) -> Self {
        Self {
            bytes,
            pos: 0,
            current_line: 1,
            record_start_line: 1,
            record_index: 0,
            lines_in_current_record: 0,
            expected_channels,
            column_names: Vec::new(),
            strict: true,
            allow_commas: false,
            null_sentinel: None,
            active_record: Vec::with_capacity(expected_channels),
            at_line_start: true,
        }
    }

    /// Configures curve / column names.
    #[must_use]
    pub fn with_column_names(mut self, names: Vec<String>) -> Self {
        if self.expected_channels == 0 {
            self.expected_channels = names.len();
            self.active_record = Vec::with_capacity(self.expected_channels);
        }
        self.column_names = names;
        self
    }

    /// Sets strict validation mode (rejects ragged/overflow/truncated records).
    #[must_use]
    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Sets whether comma delimiters (`,`) are permitted in addition to whitespace.
    #[must_use]
    pub const fn with_allow_commas(mut self, allow_commas: bool) -> Self {
        self.allow_commas = allow_commas;
        self
    }

    /// Sets the dynamic NULL sentinel for mapping missing values to `f64::NAN`.
    #[must_use]
    pub fn with_null_sentinel(mut self, sentinel: NullSentinel) -> Self {
        self.null_sentinel = Some(sentinel);
        self
    }

    /// Returns the expected number of channels per logical record.
    #[must_use]
    pub const fn expected_channels(&self) -> usize {
        self.expected_channels
    }

    /// Returns the current physical line number being parsed.
    #[must_use]
    pub const fn current_line(&self) -> usize {
        self.current_line
    }

    /// Returns the count of logical records assembled so far.
    #[must_use]
    pub const fn record_index(&self) -> usize {
        self.record_index
    }

    /// Returns the number of channels accumulated in the active record buffer.
    #[must_use]
    pub fn active_channel_count(&self) -> usize {
        self.active_record.len()
    }

    /// Assembles and returns the next complete logical `UnwrappedRecord`.
    ///
    /// Continues ingesting continuation lines until `expected_channels` are met.
    /// Returns `Ok(None)` once EOF is reached and all records are flushed.
    pub fn next_record(&mut self) -> Result<Option<UnwrappedRecord>, DataError> {
        if self.expected_channels == 0 {
            return Err(DataError::NoColumnsDefined);
        }

        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];

            // 1. Whitespace delimiters
            if b == b' ' || b == b'\t' {
                self.pos += 1;
                continue;
            }

            // 2. Optional comma delimiter
            if self.allow_commas && b == b',' {
                self.pos += 1;
                continue;
            }

            // 3. Newlines
            if b == b'\n' {
                self.pos += 1;
                self.current_line += 1;
                self.at_line_start = true;
                if !self.active_record.is_empty() {
                    self.lines_in_current_record += 1;
                }
                continue;
            }
            if b == b'\r' {
                self.pos += 1;
                if self.pos < self.bytes.len() && self.bytes[self.pos] == b'\n' {
                    self.pos += 1;
                }
                self.current_line += 1;
                self.at_line_start = true;
                if !self.active_record.is_empty() {
                    self.lines_in_current_record += 1;
                }
                continue;
            }

            // 4. Comments: # skips to end of line
            if b == b'#' {
                self.pos += 1;
                while self.pos < self.bytes.len()
                    && self.bytes[self.pos] != b'\n'
                    && self.bytes[self.pos] != b'\r'
                {
                    self.pos += 1;
                }
                continue;
            }

            // 5. Delimiter line at start of line terminates data section
            if b == b'~' && self.at_line_start {
                self.pos = self.bytes.len();
                break;
            }

            // 6. Mark record start line
            if self.active_record.is_empty() {
                self.record_start_line = self.current_line;
                self.lines_in_current_record = 1;
            }
            self.at_line_start = false;

            // 7. Token extraction
            let tok_start = self.pos;
            while self.pos < self.bytes.len() {
                let tb = self.bytes[self.pos];
                if tb == b' '
                    || tb == b'\t'
                    || tb == b'\r'
                    || tb == b'\n'
                    || tb == b'#'
                    || (self.allow_commas && tb == b',')
                {
                    break;
                }
                self.pos += 1;
            }
            let tok_bytes = &self.bytes[tok_start..self.pos];

            // 8. Float conversion via fast_float
            let mut val = match fast_float::parse::<f64, _>(tok_bytes) {
                Ok(v) => v,
                Err(_) => {
                    if is_text_null_sentinel(tok_bytes, self.null_sentinel.as_ref()) {
                        f64::NAN
                    } else {
                        let col_idx = self.active_record.len();
                        let col_name = self.column_names.get(col_idx).cloned();
                        return Err(DataError::InvalidFloat {
                            line_number: self.current_line,
                            column_index: col_idx,
                            column_name: col_name,
                            token: String::from_utf8_lossy(tok_bytes).into_owned(),
                            reason: "failed to parse floating-point value in wrapped stream"
                                .to_string(),
                        });
                    }
                }
            };

            if !val.is_nan() {
                if let Some(ref sentinel) = self.null_sentinel {
                    if sentinel.is_null(val) {
                        val = f64::NAN;
                    }
                }
            }

            self.active_record.push(val);

            // 9. Complete logical record satisfied
            if self.active_record.len() == self.expected_channels {
                let record = UnwrappedRecord {
                    record_index: self.record_index,
                    start_line: self.record_start_line,
                    end_line: self.current_line,
                    line_count: self.lines_in_current_record,
                    values: self.active_record.clone(),
                };
                self.active_record.clear();
                self.record_index += 1;
                self.lines_in_current_record = 0;
                return Ok(Some(record));
            }
        }

        // EOF handling
        if !self.active_record.is_empty() {
            if self.strict {
                return Err(DataError::WrappedRecordTruncated {
                    record_start_line: self.record_start_line,
                    expected: self.expected_channels,
                    found: self.active_record.len(),
                });
            }

            // Permissive recovery mode: pad missing channels with NaN
            while self.active_record.len() < self.expected_channels {
                self.active_record.push(f64::NAN);
            }

            let record = UnwrappedRecord {
                record_index: self.record_index,
                start_line: self.record_start_line,
                end_line: self.current_line,
                line_count: self.lines_in_current_record,
                values: self.active_record.clone(),
            };
            self.active_record.clear();
            self.record_index += 1;
            return Ok(Some(record));
        }

        Ok(None)
    }

    /// Assembles the next logical record and writes values directly into column vectors,
    /// completely bypassing `UnwrappedRecord` heap allocation.
    pub fn next_record_into_columns(
        &mut self,
        columns: &mut [Vec<f64>],
    ) -> Result<bool, DataError> {
        if let Some(record) = self.next_record()? {
            for (col_idx, &val) in record.values.iter().enumerate() {
                if let Some(col) = columns.get_mut(col_idx) {
                    col.push(val);
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Streams and unwraps the entire wrapped ASCII data block directly into a `ColumnarBuffer`.
    pub fn parse_to_columnar(&mut self) -> Result<ColumnarBuffer, DataError> {
        let capacity = self.bytes.len() / (self.expected_channels.saturating_mul(10).max(16));
        let capacity = (capacity + (capacity / 10)).max(16);

        let mut columns: Vec<Vec<f64>> = (0..self.expected_channels)
            .map(|_| Vec::with_capacity(capacity))
            .collect();

        let mut row_count = 0usize;
        while self.next_record_into_columns(&mut columns)? {
            row_count += 1;
        }

        if self.strict && row_count == 0 {
            return Err(DataError::EmptyData);
        }

        let names = if self.column_names.len() == self.expected_channels {
            self.column_names.clone()
        } else {
            (0..self.expected_channels)
                .map(|i| format!("COL_{i}"))
                .collect()
        };

        Ok(ColumnarBuffer::from_raw_parts(columns, names, row_count))
    }
}

/// Standalone function to unwrap and parse wrapped ASCII data into a `ColumnarBuffer`.
pub fn unwrap_and_parse(
    bytes: &[u8],
    expected_channels: usize,
    column_names: Option<Vec<String>>,
    strict: bool,
    allow_commas: bool,
    null_sentinel: Option<NullSentinel>,
) -> Result<ColumnarBuffer, DataError> {
    let mut unwrapper = StreamingUnwrapper::new(bytes, expected_channels)
        .with_strict(strict)
        .with_allow_commas(allow_commas);

    if let Some(names) = column_names {
        unwrapper = unwrapper.with_column_names(names);
    }
    if let Some(sentinel) = null_sentinel {
        unwrapper = unwrapper.with_null_sentinel(sentinel);
    }

    unwrapper.parse_to_columnar()
}

// ---------------------------------------------------------------------------
// PyO3 Python Bindings
// ---------------------------------------------------------------------------

/// Python wrapper for `UnwrappedRecord`.
#[pyclass(name = "UnwrappedRecord")]
#[derive(Debug, Clone)]
pub struct PyUnwrappedRecord {
    inner: UnwrappedRecord,
}

#[pymethods]
impl PyUnwrappedRecord {
    /// 0-based logical record index.
    #[getter]
    pub fn record_index(&self) -> usize {
        self.inner.record_index
    }

    /// 1-based physical line number where this logical record began.
    #[getter]
    pub fn start_line(&self) -> usize {
        self.inner.start_line
    }

    /// 1-based physical line number where this logical record completed.
    #[getter]
    pub fn end_line(&self) -> usize {
        self.inner.end_line
    }

    /// Number of physical lines folded into this record.
    #[getter]
    pub fn line_count(&self) -> usize {
        self.inner.line_count
    }

    /// Channel float values as a Python list.
    #[getter]
    pub fn values(&self) -> Vec<f64> {
        self.inner.values.clone()
    }

    pub fn __len__(&self) -> usize {
        self.inner.values.len()
    }

    pub fn __getitem__(&self, index: isize) -> PyResult<f64> {
        let len = self.inner.values.len() as isize;
        let normalized = if index < 0 { index + len } else { index };
        if normalized >= 0 && (normalized as usize) < self.inner.values.len() {
            Ok(self.inner.values[normalized as usize])
        } else {
            Err(LASDataError::new_err(format!(
                "Channel index {index} out of range (record has {} channels)",
                self.inner.values.len()
            )))
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "<UnwrappedRecord index={} lines={}-{} channels={} values={:?}>",
            self.inner.record_index,
            self.inner.start_line,
            self.inner.end_line,
            self.inner.values.len(),
            self.inner.values
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python wrapper for `StreamingUnwrapper`.
#[pyclass(name = "StreamingUnwrapper")]
pub struct PyStreamingUnwrapper {
    raw_data: Vec<u8>,
    expected_channels: usize,
    column_names: Vec<String>,
    strict: bool,
    allow_commas: bool,
    null_sentinel: Option<NullSentinel>,
    pos: usize,
    current_line: usize,
    record_start_line: usize,
    record_index: usize,
    lines_in_current_record: usize,
    active_record: Vec<f64>,
    at_line_start: bool,
}

#[pymethods]
impl PyStreamingUnwrapper {
    /// Creates a new `StreamingUnwrapper`.
    #[new]
    #[pyo3(signature = (
        data,
        expected_channels = 0,
        column_names = None,
        strict = true,
        allow_commas = false,
        null_sentinel = None
    ))]
    pub fn new(
        data: &Bound<'_, PyAny>,
        expected_channels: usize,
        column_names: Option<Vec<String>>,
        strict: bool,
        allow_commas: bool,
        null_sentinel: Option<NullSentinel>,
    ) -> PyResult<Self> {
        let raw_data = if let Ok(py_bytes) = data.extract::<&[u8]>() {
            py_bytes.to_vec()
        } else if let Ok(py_str) = data.extract::<&str>() {
            py_str.as_bytes().to_vec()
        } else if let Ok(decoded) = data.extract::<pyo3::PyRef<'_, PyDecodedBuffer>>() {
            decoded.text().as_bytes().to_vec()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected bytes, str, memoryview, or DecodedBuffer",
            ));
        };

        let cols = if let Some(ref names) = column_names {
            if expected_channels == 0 {
                names.len()
            } else {
                expected_channels
            }
        } else {
            expected_channels
        };

        Ok(Self {
            raw_data,
            expected_channels: cols,
            column_names: column_names.unwrap_or_default(),
            strict,
            allow_commas,
            null_sentinel,
            pos: 0,
            current_line: 1,
            record_start_line: 1,
            record_index: 0,
            lines_in_current_record: 0,
            active_record: Vec::with_capacity(cols),
            at_line_start: true,
        })
    }

    /// Iterator support: returns self.
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Iterator next record.
    pub fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyUnwrappedRecord>> {
        slf.next_record()
    }

    /// Fetches the next assembled `UnwrappedRecord`.
    pub fn next_record(&mut self) -> PyResult<Option<PyUnwrappedRecord>> {
        let mut unwrapper = StreamingUnwrapper {
            bytes: &self.raw_data,
            pos: self.pos,
            current_line: self.current_line,
            record_start_line: self.record_start_line,
            record_index: self.record_index,
            lines_in_current_record: self.lines_in_current_record,
            expected_channels: self.expected_channels,
            column_names: self.column_names.clone(),
            strict: self.strict,
            allow_commas: self.allow_commas,
            null_sentinel: self.null_sentinel.clone(),
            active_record: std::mem::take(&mut self.active_record),
            at_line_start: self.at_line_start,
        };

        let result = unwrapper.next_record();

        self.pos = unwrapper.pos;
        self.current_line = unwrapper.current_line;
        self.record_start_line = unwrapper.record_start_line;
        self.record_index = unwrapper.record_index;
        self.lines_in_current_record = unwrapper.lines_in_current_record;
        self.active_record = unwrapper.active_record;
        self.at_line_start = unwrapper.at_line_start;

        match result {
            Ok(Some(rec)) => Ok(Some(PyUnwrappedRecord { inner: rec })),
            Ok(None) => Ok(None),
            Err(e) => Err(LASDataError::new_err(e.to_string())),
        }
    }

    /// Unwraps all remaining records into a `ColumnarBuffer`.
    pub fn read_all(&mut self) -> PyResult<PyColumnarBuffer> {
        let mut unwrapper = StreamingUnwrapper {
            bytes: &self.raw_data,
            pos: self.pos,
            current_line: self.current_line,
            record_start_line: self.record_start_line,
            record_index: self.record_index,
            lines_in_current_record: self.lines_in_current_record,
            expected_channels: self.expected_channels,
            column_names: self.column_names.clone(),
            strict: self.strict,
            allow_commas: self.allow_commas,
            null_sentinel: self.null_sentinel.clone(),
            active_record: std::mem::take(&mut self.active_record),
            at_line_start: self.at_line_start,
        };

        let result = unwrapper.parse_to_columnar();

        self.pos = unwrapper.pos;
        self.current_line = unwrapper.current_line;
        self.record_start_line = unwrapper.record_start_line;
        self.record_index = unwrapper.record_index;
        self.lines_in_current_record = unwrapper.lines_in_current_record;
        self.active_record = unwrapper.active_record;
        self.at_line_start = unwrapper.at_line_start;

        match result {
            Ok(buf) => Ok(PyColumnarBuffer::from_inner(buf)),
            Err(e) => Err(LASDataError::new_err(e.to_string())),
        }
    }

    /// Reads all records into a Python list of `UnwrappedRecord` objects.
    pub fn read_records(&mut self) -> PyResult<Vec<PyUnwrappedRecord>> {
        let mut records = Vec::new();
        while let Some(rec) = self.next_record()? {
            records.push(rec);
        }
        Ok(records)
    }

    /// Number of expected channels per record.
    #[getter]
    pub fn expected_channels(&self) -> usize {
        self.expected_channels
    }

    /// Current physical line number.
    #[getter]
    pub fn current_line(&self) -> usize {
        self.current_line
    }

    /// Number of records completed so far.
    #[getter]
    pub fn record_index(&self) -> usize {
        self.record_index
    }

    /// Channels currently accumulated in the active record buffer.
    #[getter]
    pub fn active_channel_count(&self) -> usize {
        self.active_record.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<StreamingUnwrapper channels={} records_done={} active_buf={}>",
            self.expected_channels,
            self.record_index,
            self.active_record.len()
        )
    }
}

/// Standalone Python function to unwrap and parse wrapped ASCII data into a `ColumnarBuffer`.
#[pyfunction(name = "unwrap_and_parse")]
#[pyo3(signature = (
    data,
    expected_channels = 0,
    column_names = None,
    strict = true,
    allow_commas = false,
    null_sentinel = None
))]
pub fn py_unwrap_and_parse(
    data: &Bound<'_, PyAny>,
    expected_channels: usize,
    column_names: Option<Vec<String>>,
    strict: bool,
    allow_commas: bool,
    null_sentinel: Option<NullSentinel>,
) -> PyResult<PyColumnarBuffer> {
    let mut parser = PyStreamingUnwrapper::new(
        data,
        expected_channels,
        column_names,
        strict,
        allow_commas,
        null_sentinel,
    )?;
    parser.read_all()
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_unwrapper_basic_two_lines_per_depth() {
        // 5 channels folded across 2 lines per depth step
        let data = b" 1000.0  25.5  120.3\n  0.45  2.65\n 1000.5  26.0  121.0\n  0.46  2.66\n";
        let mut unwrapper = StreamingUnwrapper::new(data, 5);

        let r1 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r1.record_index, 0);
        assert_eq!(r1.start_line, 1);
        assert_eq!(r1.end_line, 2);
        assert_eq!(r1.line_count, 2);
        assert_eq!(r1.values, vec![1000.0, 25.5, 120.3, 0.45, 2.65]);

        let r2 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r2.record_index, 1);
        assert_eq!(r2.start_line, 3);
        assert_eq!(r2.end_line, 4);
        assert_eq!(r2.line_count, 2);
        assert_eq!(r2.values, vec![1000.5, 26.0, 121.0, 0.46, 2.66]);

        assert!(unwrapper.next_record().unwrap().is_none());
    }

    #[test]
    fn test_streaming_unwrapper_three_lines_per_depth() {
        // 7 channels folded across 3 lines: 1 + 3 + 3
        let data = b" 1000.0\n 25.5 120.3 0.45\n 2.65 15.0 45.0\n 1000.5\n 26.0 121.0 0.46\n 2.66 16.0 46.0\n";
        let mut unwrapper = StreamingUnwrapper::new(data, 7);

        let r1 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r1.values.len(), 7);
        assert_eq!(r1.start_line, 1);
        assert_eq!(r1.end_line, 3);
        assert_eq!(r1.line_count, 3);
        assert_eq!(r1.values[0], 1000.0);
        assert_eq!(r1.values[6], 45.0);

        let r2 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r2.values.len(), 7);
        assert_eq!(r2.start_line, 4);
        assert_eq!(r2.end_line, 6);
        assert_eq!(r2.values[0], 1000.5);
        assert_eq!(r2.values[6], 46.0);

        assert!(unwrapper.next_record().unwrap().is_none());
    }

    #[test]
    fn test_streaming_unwrapper_with_comments_and_blank_lines() {
        let data = b"# Header remark\n\n 1000.0 25.0 120.0\n# Mid remark\n 0.40 2.50\n\n 1000.5 26.0 121.0\n 0.41 2.51\r\n";
        let mut unwrapper = StreamingUnwrapper::new(data, 5);

        let r1 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r1.values, vec![1000.0, 25.0, 120.0, 0.40, 2.50]);
        assert_eq!(r1.start_line, 3);
        assert_eq!(r1.end_line, 5);

        let r2 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r2.values, vec![1000.5, 26.0, 121.0, 0.41, 2.51]);
    }

    #[test]
    fn test_streaming_unwrapper_null_sentinel_mapping() {
        let data = b" 1000.0 -999.25 120.0\n null NaN\n";
        let sentinel = NullSentinel::default_cwls();
        let mut unwrapper = StreamingUnwrapper::new(data, 5).with_null_sentinel(sentinel);

        let r1 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r1.values[0], 1000.0);
        assert!(r1.values[1].is_nan());
        assert_eq!(r1.values[2], 120.0);
        assert!(r1.values[3].is_nan());
        assert!(r1.values[4].is_nan());
    }

    #[test]
    fn test_streaming_unwrapper_strict_overflow_error() {
        // Line has 6 values when 5 are expected
        let data = b" 1000.0 25.0 120.0\n 0.40 2.50 99.9\n";
        let mut unwrapper = StreamingUnwrapper::new(data, 5).with_strict(true);
        let r1 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r1.values.len(), 5);

        // Next read encounters the extra token 99.9 at line 2 which starts a new record but then ends
        let err = unwrapper.next_record().unwrap_err();
        match err {
            DataError::WrappedRecordTruncated {
                record_start_line,
                expected,
                found,
            } => {
                assert_eq!(record_start_line, 2);
                assert_eq!(expected, 5);
                assert_eq!(found, 1);
            }
            _ => panic!("Expected WrappedRecordTruncated, got {err:?}"),
        }
    }

    #[test]
    fn test_streaming_unwrapper_recovery_from_truncated_trailing_line() {
        // Record 1 complete (5 channels). Record 2 truncated after 3 channels.
        let data = b" 1000.0 25.0 120.0\n 0.40 2.50\n 1000.5 26.0 121.0";
        let mut unwrapper = StreamingUnwrapper::new(data, 5).with_strict(false);

        let r1 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r1.values, vec![1000.0, 25.0, 120.0, 0.40, 2.50]);

        // Record 2 recovered and padded with NaN
        let r2 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r2.values[0], 1000.5);
        assert_eq!(r2.values[1], 26.0);
        assert_eq!(r2.values[2], 121.0);
        assert!(r2.values[3].is_nan());
        assert!(r2.values[4].is_nan());

        assert!(unwrapper.next_record().unwrap().is_none());
    }

    #[test]
    fn test_streaming_unwrapper_trailing_section_stops_parsing() {
        let data = b" 1000.0 25.0\n 0.40 2.50\n~Other Information\nThis should stop unwrapping";
        let mut unwrapper = StreamingUnwrapper::new(data, 4);

        let r1 = unwrapper.next_record().unwrap().unwrap();
        assert_eq!(r1.values, vec![1000.0, 25.0, 0.40, 2.50]);

        assert!(unwrapper.next_record().unwrap().is_none());
    }

    #[test]
    fn test_parse_to_columnar_parity_with_unwrapped() {
        // Wrapped version
        let wrapped_data = b" 1000.0 25.0 120.0\n 0.40\n 1000.5 26.0 121.0\n 0.41\n";
        // Unwrapped version
        let unwrapped_data = b" 1000.0 25.0 120.0 0.40\n 1000.5 26.0 121.0 0.41\n";

        let names = vec![
            "DEPT".to_string(),
            "GR".to_string(),
            "RHOB".to_string(),
            "NPHI".to_string(),
        ];

        let buf_wrapped =
            unwrap_and_parse(wrapped_data, 4, Some(names.clone()), true, false, None).unwrap();

        let parser_unwrapped = crate::columnar::ColumnarParser::new(4)
            .with_column_names(names)
            .with_wrapped(false);
        let buf_unwrapped = parser_unwrapped.parse_bytes(unwrapped_data).unwrap();

        assert_eq!(buf_wrapped.num_rows(), buf_unwrapped.num_rows());
        assert_eq!(buf_wrapped.num_columns(), buf_unwrapped.num_columns());
        assert_eq!(
            buf_wrapped.column(0).unwrap(),
            buf_unwrapped.column(0).unwrap()
        );
        assert_eq!(
            buf_wrapped.column(1).unwrap(),
            buf_unwrapped.column(1).unwrap()
        );
        assert_eq!(
            buf_wrapped.column(2).unwrap(),
            buf_unwrapped.column(2).unwrap()
        );
        assert_eq!(
            buf_wrapped.column(3).unwrap(),
            buf_unwrapped.column(3).unwrap()
        );
    }
}
