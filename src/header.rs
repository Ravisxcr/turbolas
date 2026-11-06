//! Non-regex CWLS LAS header line parser (`MNEM.UNIT VALUE : DESC`).
//!
//! The Canadian Well Logging Society (CWLS) Log ASCII Standard (LAS) format
//! specifies that header metadata lines in sections `~V`, `~W`, `~C`, and `~P`
//! follow a structured layout:
//!
//! ```text
//! MNEM.UNIT               VALUE                 : DESCRIPTION
//! ```
//!
//! # Structural Specification
//! - **Mnemonic (`MNEM`)**: The property, curve, or parameter name up to the first dot (`.`).
//!   Leading whitespace is stripped. In strict CWLS LAS 2.0, mnemonics cannot contain
//!   internal whitespace or colons.
//! - **Dot (`.`)**: Mandatory delimiter separating the mnemonic from the unit.
//! - **Unit (`UNIT`)**: Measurement unit (e.g. `M`, `FT`, `K/M3`, `OHM-M`, `DEG_C`).
//!   Per CWLS LAS 2.0: "The units (if any) are located directly after the dot and continue
//!   up to the first space. If there are no units, the dot is followed by a space."
//! - **Value / Data (`VALUE`)**: The assigned setting or curve code, situated between the
//!   unit and the description delimiter colon (`:`).
//! - **Colon (`:`)**: Mandatory delimiter in strict mode dividing value and description.
//! - **Description (`DESC`)**: Human-readable text following the colon.
//!
//! # Performance
//! This module uses zero-allocation slice indexing and byte scanning rather than
//! regular expressions, avoiding regex engine overhead and enabling fast scanning
//! of thousands of header records.

use pyo3::prelude::*;
use std::fmt;

use crate::error::HeaderError;
use crate::state_machine::ValidationMode;

/// Borrowed, zero-allocation reference to a parsed CWLS LAS header line.
///
/// Holds slices directly borrowing from the input line buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderItemRef<'a> {
    /// Mnemonic identifier (trimmed).
    pub mnemonic: &'a str,
    /// Unit of measurement (empty if none).
    pub unit: &'a str,
    /// Data value (trimmed, empty if none).
    pub value: &'a str,
    /// Description text (trimmed, empty if none).
    pub description: &'a str,
    /// Clean raw line without trailing newline/carriage return.
    pub raw_line: &'a str,
    /// 1-based line number in the source file.
    pub line_number: usize,
}

impl<'a> HeaderItemRef<'a> {
    /// Parses a CWLS LAS header line without regular expressions or heap allocations.
    ///
    /// # Arguments
    /// * `line` - The raw text line to parse.
    /// * `line_number` - 1-based line number for error reporting.
    /// * `mode` - Validation mode (`Strict` or `Permissive`).
    ///
    /// # Errors
    /// Returns `HeaderError` if the line is empty, is a comment/delimiter, lacks a required
    /// dot, contains an empty/invalid mnemonic, or lacks a colon in strict mode.
    pub fn parse(
        line: &'a str,
        line_number: usize,
        mode: ValidationMode,
    ) -> Result<Self, HeaderError> {
        let clean_line = line.trim_end_matches(['\r', '\n']);
        let trimmed_start = clean_line.trim_start();

        // 1. Check for empty or whitespace-only line
        if trimmed_start.is_empty() {
            return Err(HeaderError::EmptyLine { line_number });
        }

        // 2. Check for comment line
        if trimmed_start.starts_with('#') {
            return Err(HeaderError::CommentLine {
                line_number,
                line: clean_line.to_string(),
            });
        }

        // 3. Check for section delimiter
        if trimmed_start.starts_with('~') {
            return Err(HeaderError::SectionDelimiterLine {
                line_number,
                line: clean_line.to_string(),
            });
        }

        // 4. Find dot '.' separating mnemonic and unit
        let Some(dot_idx) = find_mnemonic_dot(clean_line) else {
            return Err(HeaderError::MissingDot {
                line_number,
                line: clean_line.to_string(),
            });
        };

        // 5. Extract and validate mnemonic
        let raw_mnemonic = &clean_line[..dot_idx];
        let mnemonic = raw_mnemonic.trim();

        if mnemonic.is_empty() {
            return Err(HeaderError::EmptyMnemonic {
                line_number,
                line: clean_line.to_string(),
            });
        }

        if mnemonic.contains(':') {
            return Err(HeaderError::InvalidMnemonic {
                line_number,
                mnemonic: mnemonic.to_string(),
                reason: "mnemonic cannot contain colon ':'".to_string(),
            });
        }

        if mode.is_strict() && mnemonic.contains(char::is_whitespace) {
            return Err(HeaderError::InvalidMnemonic {
                line_number,
                mnemonic: mnemonic.to_string(),
                reason: "mnemonic contains whitespace, forbidden in strict CWLS LAS 2.0"
                    .to_string(),
            });
        }

        // 6. Extract unit
        let after_dot = &clean_line[dot_idx + 1..];
        let (unit, rest) = extract_unit(after_dot);

        // 7. Extract value and description
        let (value, description) = extract_value_and_desc(rest, line_number, clean_line, mode)?;

        Ok(Self {
            mnemonic,
            unit,
            value,
            description,
            raw_line: clean_line,
            line_number,
        })
    }

    /// Converts this borrowed reference into an owned [`HeaderItem`].
    #[must_use]
    pub fn to_owned(&self) -> HeaderItem {
        HeaderItem {
            mnemonic: self.mnemonic.to_string(),
            unit: self.unit.to_string(),
            value: self.value.to_string(),
            description: self.description.to_string(),
            raw_line: self.raw_line.to_string(),
            line_number: self.line_number,
        }
    }
}

/// Represents an owned, parsed CWLS LAS header item record.
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderItem {
    pub(crate) mnemonic: String,
    pub(crate) unit: String,
    pub(crate) value: String,
    pub(crate) description: String,
    pub(crate) raw_line: String,
    pub(crate) line_number: usize,
}

impl HeaderItem {
    /// Creates a new owned [`HeaderItem`].
    #[must_use]
    pub fn new_item(
        mnemonic: impl Into<String>,
        unit: impl Into<String>,
        value: impl Into<String>,
        description: impl Into<String>,
        line_number: usize,
        raw_line: impl Into<String>,
    ) -> Self {
        Self {
            mnemonic: mnemonic.into(),
            unit: unit.into(),
            value: value.into(),
            description: description.into(),
            line_number,
            raw_line: raw_line.into(),
        }
    }

    /// Parses a header line into an owned [`HeaderItem`].
    ///
    /// # Errors
    /// Returns `HeaderError` if parsing fails.
    pub fn parse(
        line: &str,
        line_number: usize,
        mode: ValidationMode,
    ) -> Result<Self, HeaderError> {
        HeaderItemRef::parse(line, line_number, mode).map(|r| r.to_owned())
    }

    /// Attempts to parse a header line, returning `None` on failure.
    #[must_use]
    pub fn try_parse(line: &str, line_number: usize, mode: ValidationMode) -> Option<Self> {
        Self::parse(line, line_number, mode).ok()
    }

    /// Returns the parsed mnemonic identifier.
    #[must_use]
    pub fn mnemonic(&self) -> &str {
        &self.mnemonic
    }

    /// Returns the unit of measurement, or empty string if none.
    #[must_use]
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Returns the value string, or empty string if none.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Returns the description string, or empty string if none.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the clean raw line as read from the source.
    #[must_use]
    pub fn raw_line(&self) -> &str {
        &self.raw_line
    }

    /// Returns the 1-based line number.
    #[must_use]
    pub const fn line_number(&self) -> usize {
        self.line_number
    }

    /// Returns whether this item has a non-empty unit.
    #[must_use]
    pub fn has_unit(&self) -> bool {
        !self.unit.is_empty()
    }

    /// Returns whether this item's value is empty.
    #[must_use]
    pub fn is_empty_value(&self) -> bool {
        self.value.is_empty()
    }

    /// Returns whether this item has a non-empty description.
    #[must_use]
    pub fn has_description(&self) -> bool {
        !self.description.is_empty()
    }

    /// Parses the value as an `f64` floating point number, if possible.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        self.value.trim().parse::<f64>().ok()
    }

    /// Parses the value as an `i64` integer, if possible.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        self.value.trim().parse::<i64>().ok()
    }

    /// Parses common boolean representations in LAS files.
    ///
    /// Recognizes `YES`/`Y`/`TRUE`/`T`/`1` as `true`,
    /// and `NO`/`N`/`FALSE`/`F`/`0` as `false` (case-insensitive).
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        let trimmed = self.value.trim();
        if trimmed.eq_ignore_ascii_case("yes")
            || trimmed.eq_ignore_ascii_case("y")
            || trimmed.eq_ignore_ascii_case("true")
            || trimmed.eq_ignore_ascii_case("t")
            || trimmed == "1"
        {
            Some(true)
        } else if trimmed.eq_ignore_ascii_case("no")
            || trimmed.eq_ignore_ascii_case("n")
            || trimmed.eq_ignore_ascii_case("false")
            || trimmed.eq_ignore_ascii_case("f")
            || trimmed == "0"
        {
            Some(false)
        } else {
            None
        }
    }

    /// Returns the value stripped of outer matching single or double quotes.
    #[must_use]
    pub fn unquoted_value(&self) -> &str {
        let trimmed = self.value.trim();
        if (trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2)
            || (trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2)
        {
            &trimmed[1..trimmed.len() - 1]
        } else {
            trimmed
        }
    }

    /// Returns whether this record defines the NULL sentinel (`mnemonic == "NULL"`).
    #[must_use]
    pub fn is_null_record(&self) -> bool {
        self.mnemonic.eq_ignore_ascii_case("NULL")
    }

    /// Checks whether this item's value is null according to the provided sentinel
    /// (or standard CWLS `-999.25` if none provided).
    #[must_use]
    pub fn is_null(&self, sentinel: Option<&crate::null::NullSentinel>) -> bool {
        if self.is_empty_value() {
            return true;
        }
        if let Some(f) = self.as_float() {
            if let Some(s) = sentinel {
                s.is_null(f)
            } else {
                crate::null::is_null(f, crate::null::DEFAULT_CWLS_NULL)
            }
        } else {
            false
        }
    }

    /// Parses the value as `f64`, mapping null sentinels to `f64::NAN`.
    ///
    /// Returns `Some(f64::NAN)` if the value is a null sentinel or empty,
    /// `Some(val)` if it parses to a valid non-null float, or `None` if parsing fails.
    #[must_use]
    pub fn as_float_mapped(&self, sentinel: Option<&crate::null::NullSentinel>) -> Option<f64> {
        if self.is_empty_value() {
            return Some(f64::NAN);
        }
        let f = self.as_float()?;
        let is_null = if let Some(s) = sentinel {
            s.is_null(f)
        } else {
            crate::null::is_null(f, crate::null::DEFAULT_CWLS_NULL)
        };
        if is_null {
            Some(f64::NAN)
        } else {
            Some(f)
        }
    }

    /// Formats this header item into a standard CWLS LAS line representation.
    #[must_use]
    pub fn format_line(
        &self,
        mnem_width: Option<usize>,
        unit_width: Option<usize>,
        val_width: Option<usize>,
    ) -> String {
        let mw = mnem_width.unwrap_or(4).max(self.mnemonic.len());
        let uw = unit_width.unwrap_or(4).max(self.unit.len());
        let vw = val_width.unwrap_or(12).max(self.value.len());

        format!(
            "{:<mw$}.{:<uw$} {:<vw$} : {}",
            self.mnemonic, self.unit, self.value, self.description
        )
    }

    /// Formats this header item to standard CWLS LAS line layout.
    #[must_use]
    pub fn to_line(&self) -> String {
        self.format_line(None, None, None)
    }
}

#[pymethods]
impl HeaderItem {
    /// Creates a new `HeaderItem`.
    #[new]
    #[pyo3(signature = (mnemonic, unit="", value="", description="", line_number=1, raw_line=""))]
    fn py_new(
        mnemonic: String,
        unit: &str,
        value: &str,
        description: &str,
        line_number: usize,
        raw_line: &str,
    ) -> Self {
        let clean_raw = if raw_line.is_empty() {
            format!("{mnemonic}.{unit} {value} : {description}")
        } else {
            raw_line.trim_end_matches(['\r', '\n']).to_string()
        };

        Self {
            mnemonic,
            unit: unit.to_string(),
            value: value.to_string(),
            description: description.to_string(),
            line_number,
            raw_line: clean_raw,
        }
    }

    /// Mnemonic identifier name.
    #[getter]
    fn get_mnemonic(&self) -> &str {
        self.mnemonic()
    }

    /// Unit of measurement.
    #[getter]
    fn get_unit(&self) -> &str {
        self.unit()
    }

    /// Value or data string.
    #[getter]
    fn get_value(&self) -> &str {
        self.value()
    }

    /// Description text.
    #[getter]
    fn get_description(&self) -> &str {
        self.description()
    }

    /// Alias for description (compatible with `lasio.descr`).
    #[getter]
    fn get_descr(&self) -> &str {
        self.description()
    }

    /// Raw unparsed source line text.
    #[getter]
    fn get_raw_line(&self) -> &str {
        self.raw_line()
    }

    /// 1-based source line number.
    #[getter]
    fn get_line_number(&self) -> usize {
        self.line_number()
    }

    /// True if unit string is non-empty.
    #[getter]
    fn get_has_unit(&self) -> bool {
        self.has_unit()
    }

    /// True if value string is empty.
    #[getter]
    fn get_is_empty_value(&self) -> bool {
        self.is_empty_value()
    }

    /// True if description string is non-empty.
    #[getter]
    fn get_has_description(&self) -> bool {
        self.has_description()
    }

    /// Parses the value as a float, returning `None` if invalid.
    #[pyo3(name = "as_float")]
    fn py_as_float(&self) -> Option<f64> {
        self.as_float()
    }

    /// Parses the value as an integer, returning `None` if invalid.
    #[pyo3(name = "as_int")]
    fn py_as_int(&self) -> Option<i64> {
        self.as_int()
    }

    /// Parses the value as a boolean (`YES`/`NO`, `TRUE`/`FALSE`, `1`/`0`).
    #[pyo3(name = "as_bool")]
    fn py_as_bool(&self) -> Option<bool> {
        self.as_bool()
    }

    /// Returns the value stripped of outer matching single or double quotes.
    #[pyo3(name = "unquoted_value")]
    fn py_unquoted_value(&self) -> &str {
        self.unquoted_value()
    }

    /// Returns whether this record defines the NULL sentinel.
    #[getter]
    fn get_is_null_record(&self) -> bool {
        self.is_null_record()
    }

    /// Checks whether this item's value is null.
    #[pyo3(name = "is_null", signature = (sentinel=None))]
    fn py_is_null(&self, sentinel: Option<&crate::null::NullSentinel>) -> bool {
        self.is_null(sentinel)
    }

    /// Parses value as float, mapping null sentinels to IEEE 754 NaN.
    #[pyo3(name = "as_float_mapped", signature = (sentinel=None))]
    fn py_as_float_mapped(&self, sentinel: Option<&crate::null::NullSentinel>) -> Option<f64> {
        self.as_float_mapped(sentinel)
    }

    /// Formats the header item back into a CWLS LAS formatted line string.
    #[pyo3(name = "format_line", signature = (mnem_width=None, unit_width=None, val_width=None))]
    fn py_format_line(
        &self,
        mnem_width: Option<usize>,
        unit_width: Option<usize>,
        val_width: Option<usize>,
    ) -> String {
        self.format_line(mnem_width, unit_width, val_width)
    }

    /// Formats to standard CWLS LAS line string.
    #[pyo3(name = "to_line")]
    fn py_to_line(&self) -> String {
        self.to_line()
    }

    /// Parses a single header line with the specified validation mode.
    #[staticmethod]
    #[pyo3(signature = (line, line_number=1, mode=None))]
    fn parse_line(line: &str, line_number: usize, mode: Option<ValidationMode>) -> PyResult<Self> {
        let validation_mode = mode.unwrap_or(ValidationMode::Strict);
        Self::parse(line, line_number, validation_mode).map_err(Into::into)
    }

    /// Attempts to parse a header line, returning `None` on error.
    #[staticmethod]
    #[pyo3(signature = (line, line_number=1, mode=None))]
    fn try_parse_line(
        line: &str,
        line_number: usize,
        mode: Option<ValidationMode>,
    ) -> Option<Self> {
        let validation_mode = mode.unwrap_or(ValidationMode::Strict);
        Self::try_parse(line, line_number, validation_mode)
    }

    /// Checks whether a line matches candidate header line structure.
    #[staticmethod]
    fn is_header_line(line: &str) -> bool {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('~') {
            return false;
        }
        trimmed.contains('.')
    }

    /// Checks whether a line is a LAS comment line (`#`).
    #[staticmethod]
    fn is_comment(line: &str) -> bool {
        line.trim_start().starts_with('#')
    }

    /// Checks whether a line is blank or whitespace-only.
    #[staticmethod]
    fn is_blank(line: &str) -> bool {
        line.trim().is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "HeaderItem(mnemonic='{}', unit='{}', value='{}', description='{}')",
            self.mnemonic, self.unit, self.value, self.description
        )
    }

    fn __str__(&self) -> String {
        self.to_line()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.mnemonic == other.mnemonic
            && self.unit == other.unit
            && self.value == other.value
            && self.description == other.description
    }
}

impl fmt::Display for HeaderItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_line())
    }
}

/// Configurable CWLS LAS header line parser supporting batch section parsing.
#[pyclass]
#[derive(Debug, Clone)]
pub struct HeaderParser {
    mode: ValidationMode,
}

impl HeaderParser {
    /// Creates a new `HeaderParser` with the specified validation mode.
    #[must_use]
    pub const fn new(mode: ValidationMode) -> Self {
        Self { mode }
    }

    /// Returns the current validation mode.
    #[must_use]
    pub const fn mode(&self) -> ValidationMode {
        self.mode
    }
}

#[pymethods]
impl HeaderParser {
    /// Creates a new `HeaderParser`.
    #[new]
    #[pyo3(signature = (mode=None, strict=None))]
    fn py_new(mode: Option<ValidationMode>, strict: Option<bool>) -> Self {
        let validation_mode = match (mode, strict) {
            (Some(m), _) => m,
            (None, Some(false)) => ValidationMode::Permissive,
            (None, _) => ValidationMode::Strict,
        };
        Self::new(validation_mode)
    }

    /// Returns the validation mode.
    #[getter]
    fn get_mode(&self) -> ValidationMode {
        self.mode
    }

    /// Sets the validation mode.
    #[setter]
    fn set_mode(&mut self, mode: ValidationMode) {
        self.mode = mode;
    }

    /// Returns whether this parser runs in strict mode.
    #[getter]
    fn get_is_strict(&self) -> bool {
        self.mode.is_strict()
    }

    /// Returns whether this parser runs in permissive mode.
    #[getter]
    fn get_is_permissive(&self) -> bool {
        self.mode.is_permissive()
    }

    /// Parses a single header line using this parser's configuration.
    #[pyo3(signature = (line, line_number=1))]
    fn parse_line(&self, line: &str, line_number: usize) -> PyResult<HeaderItem> {
        HeaderItem::parse(line, line_number, self.mode).map_err(Into::into)
    }

    /// Parses multiple lines in a header section, automatically skipping comments and blank lines.
    ///
    /// # Errors
    /// Returns `LASHeaderError` if any non-comment, non-blank line fails to parse.
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (lines, start_line_number=1))]
    fn parse_lines(
        &self,
        lines: Vec<String>,
        start_line_number: usize,
    ) -> PyResult<Vec<HeaderItem>> {
        let mut items = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let line_no = start_line_number + i;
            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let item = HeaderItem::parse(line, line_no, self.mode)?;
            items.push(item);
        }
        Ok(items)
    }
}

/// Standalone Python function to parse a single CWLS LAS header line.
///
/// # Errors
/// Returns `LASHeaderError` if the line fails to parse according to CWLS specifications.
#[pyfunction]
#[pyo3(signature = (line, line_number=1, mode=None))]
pub fn parse_header_line(
    line: &str,
    line_number: usize,
    mode: Option<ValidationMode>,
) -> PyResult<HeaderItem> {
    let validation_mode = mode.unwrap_or(ValidationMode::Strict);
    HeaderItem::parse(line, line_number, validation_mode).map_err(Into::into)
}

/// Finds the index of the dot `.` separating the mnemonic from the unit.
///
/// Ensures that a decimal point in a numeric data value (e.g. `1670.0000` in a malformed line
/// missing the mnemonic dot) is not mistakenly identified as the mnemonic separator dot.
fn find_mnemonic_dot(clean_line: &str) -> Option<usize> {
    for (idx, b) in clean_line.bytes().enumerate() {
        if b == b'.' {
            let before_dot = &clean_line[..idx];
            let words: Vec<&str> = before_dot.split_whitespace().collect();
            if words.is_empty() {
                // Dot at start of line: e.g. ".M 100 : DESC"
                return Some(idx);
            }
            if words.len() == 1 {
                return Some(idx);
            }
            if words.len() == 2 {
                let last_word = words[1];
                let is_all_digits = last_word.chars().all(|c| c.is_ascii_digit());
                if !is_all_digits {
                    return Some(idx);
                }
            }
            // If words.len() > 2 or words.len() == 2 with numeric second word,
            // this dot is likely a decimal point in a numeric value, not the mnemonic dot.
        }
    }
    None
}

/// Extracts unit and remainder from `after_dot`.
///
/// Follows CWLS LAS 2.0 standard:
/// "The units of measurement (if any) are located directly after the dot and continue
/// up to the first space. If there are no units, the dot is followed by a space."
fn extract_unit(after_dot: &str) -> (&str, &str) {
    if after_dot.is_empty() {
        return ("", "");
    }

    let bytes = after_dot.as_bytes();
    let first = bytes[0];

    // If directly after dot is a space, tab, or colon, unit is empty
    if first == b' ' || first == b'\t' || first == b':' {
        return ("", after_dot);
    }

    // Unit begins immediately after dot and continues up to whitespace, colon, or end of string
    let unit_len = bytes
        .iter()
        .position(|&b| b == b' ' || b == b'\t' || b == b':')
        .unwrap_or(bytes.len());

    let unit = &after_dot[..unit_len];
    let rest = &after_dot[unit_len..];

    (unit, rest)
}

/// Extracts value and description from the portion of the line after the unit.
fn extract_value_and_desc<'a>(
    rest: &'a str,
    line_number: usize,
    clean_line: &str,
    mode: ValidationMode,
) -> Result<(&'a str, &'a str), HeaderError> {
    if let Some(colon_idx) = find_description_colon(rest) {
        let val_part = &rest[..colon_idx];
        let desc_part = &rest[colon_idx + 1..];
        Ok((val_part.trim(), desc_part.trim()))
    } else if mode.is_strict() {
        Err(HeaderError::MissingColon {
            line_number,
            line: clean_line.to_string(),
        })
    } else {
        // Permissive mode allows missing colon: rest is value, description is empty
        Ok((rest.trim(), ""))
    }
}

/// Finds the index of the description delimiter colon `:` in `rest`.
///
/// Handles complex edge cases without regular expressions:
/// 1. Single colon: Fast-path, directly returns its index.
/// 2. Multiple colons:
///    - Skips colons inside matched quotes (`"..."` or `'...'`).
///    - Skips colons that are internal time separators (`HH:MM:SS`) or numeric ratios (`1:24000`).
///    - Returns the first colon that is an actual description delimiter.
#[allow(clippy::naive_bytecount)]
#[must_use]
pub fn find_description_colon(rest: &str) -> Option<usize> {
    let bytes = rest.as_bytes();
    let colon_count = bytes.iter().filter(|&&b| b == b':').count();

    if colon_count == 0 {
        return None;
    }

    if colon_count == 1 {
        return rest.find(':');
    }

    // Multiple colons: scan with quote and internal colon awareness
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'\'' if !in_double_quote => in_single_quote = !in_single_quote,
            b'"' if !in_single_quote => in_double_quote = !in_double_quote,
            b':' if !in_single_quote && !in_double_quote && !is_internal_colon(rest, i) => {
                return Some(i);
            }
            _ => {}
        }
    }

    None
}

/// Determines whether a colon at `idx` in `s` is an internal value separator
/// (such as time `12:30:00`, timestamp `2024-01-01 14:30:00`, or ratio `1:24000`)
/// rather than the description delimiter colon.
fn is_internal_colon(s: &str, idx: usize) -> bool {
    let bytes = s.as_bytes();
    if idx == 0 || idx + 1 >= bytes.len() {
        return false;
    }

    let prev = bytes[idx - 1];
    let next = bytes[idx + 1];

    let is_prev_time_or_digit = prev.is_ascii_digit()
        || prev == b'h'
        || prev == b'H'
        || prev == b'm'
        || prev == b'M'
        || prev == b's'
        || prev == b'S';

    let is_next_time_or_digit = next.is_ascii_digit()
        || next == b'h'
        || next == b'H'
        || next == b'm'
        || next == b'M'
        || next == b's'
        || next == b'S';

    if !is_prev_time_or_digit || !is_next_time_or_digit {
        return false;
    }

    // Check character following next: should be digit, colon, period, whitespace, or end of string
    if idx + 2 < bytes.len() {
        let after_next = bytes[idx + 2];
        if !after_next.is_ascii_digit()
            && after_next != b':'
            && after_next != b'.'
            && !after_next.is_ascii_whitespace()
            && after_next != b'm'
            && after_next != b'M'
            && after_next != b's'
            && after_next != b'S'
        {
            return false;
        }
    }

    true
}

#[cfg(test)]
#[allow(clippy::pedantic)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_version_records() {
        let line = "VERS. 2.0 : CWLS LOG ASCII STANDARD -VERSION 2.0";
        let item = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap();
        assert_eq!(item.mnemonic(), "VERS");
        assert_eq!(item.unit(), "");
        assert_eq!(item.value(), "2.0");
        assert_eq!(item.description(), "CWLS LOG ASCII STANDARD -VERSION 2.0");
        assert_eq!(item.as_float(), Some(2.0));

        let line2 = "WRAP. NO : ONE LINE PER DEPTH STEP";
        let item2 = HeaderItem::parse(line2, 2, ValidationMode::Strict).unwrap();
        assert_eq!(item2.mnemonic(), "WRAP");
        assert_eq!(item2.unit(), "");
        assert_eq!(item2.value(), "NO");
        assert_eq!(item2.as_bool(), Some(false));
    }

    #[test]
    fn test_standard_well_records() {
        let line = "STRT .M 1670.0000 : START DEPTH";
        let item = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap();
        assert_eq!(item.mnemonic(), "STRT");
        assert_eq!(item.unit(), "M");
        assert_eq!(item.value(), "1670.0000");
        assert_eq!(item.description(), "START DEPTH");
        assert_eq!(item.as_float(), Some(1670.0));

        let line_null = "NULL . -999.2500 : NULL VALUE";
        let item_null = HeaderItem::parse(line_null, 2, ValidationMode::Strict).unwrap();
        assert_eq!(item_null.mnemonic(), "NULL");
        assert_eq!(item_null.unit(), "");
        assert_eq!(item_null.value(), "-999.2500");
        assert_eq!(item_null.as_float(), Some(-999.25));

        let line_company = "COMP . ANY OIL COMPANY INC. : COMPANY";
        let item_comp = HeaderItem::parse(line_company, 3, ValidationMode::Strict).unwrap();
        assert_eq!(item_comp.mnemonic(), "COMP");
        assert_eq!(item_comp.unit(), "");
        assert_eq!(item_comp.value(), "ANY OIL COMPANY INC.");
        assert_eq!(item_comp.description(), "COMPANY");
    }

    #[test]
    fn test_standard_curve_records() {
        let line = "DEPT .M : 1 DEPTH";
        let item = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap();
        assert_eq!(item.mnemonic(), "DEPT");
        assert_eq!(item.unit(), "M");
        assert_eq!(item.value(), "");
        assert!(item.is_empty_value());
        assert_eq!(item.description(), "1 DEPTH");

        let line2 = "RHOB .K/M3 2550.0000 : 2 BULK DENSITY";
        let item2 = HeaderItem::parse(line2, 2, ValidationMode::Strict).unwrap();
        assert_eq!(item2.mnemonic(), "RHOB");
        assert_eq!(item2.unit(), "K/M3");
        assert_eq!(item2.value(), "2550.0000");
        assert_eq!(item2.as_float(), Some(2550.0));
        assert_eq!(item2.description(), "2 BULK DENSITY");
    }

    #[test]
    fn test_complex_units() {
        let cases = [
            ("RES.OHM-M 10.5 : RES", "RES", "OHM-M", "10.5"),
            ("TEMP.DEG_C 85.0 : TEMP", "TEMP", "DEG_C", "85.0"),
            ("POR.V/V 0.25 : POROSITY", "POR", "V/V", "0.25"),
            ("CALI.IN 8.5 : CALIPER", "CALI", "IN", "8.5"),
            ("PERM.MD 150 : PERM", "PERM", "MD", "150"),
            ("GAS.% 12.5 : GAS CONC", "GAS", "%", "12.5"),
        ];

        for (line, expected_mnem, expected_unit, expected_val) in cases {
            let item = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap();
            assert_eq!(item.mnemonic(), expected_mnem);
            assert_eq!(item.unit(), expected_unit);
            assert_eq!(item.value(), expected_val);
        }
    }

    #[test]
    fn test_empty_fields() {
        // No unit, no value, no desc
        let item = HeaderItem::parse("EMPTY. :", 1, ValidationMode::Strict).unwrap();
        assert_eq!(item.mnemonic(), "EMPTY");
        assert_eq!(item.unit(), "");
        assert_eq!(item.value(), "");
        assert_eq!(item.description(), "");

        // Unit with empty value and empty desc
        let item2 = HeaderItem::parse("DEPT.M :", 1, ValidationMode::Strict).unwrap();
        assert_eq!(item2.mnemonic(), "DEPT");
        assert_eq!(item2.unit(), "M");
        assert_eq!(item2.value(), "");
        assert_eq!(item2.description(), "");

        // No space between unit and colon
        let item3 = HeaderItem::parse("DEPT.M:DEPTH", 1, ValidationMode::Strict).unwrap();
        assert_eq!(item3.mnemonic(), "DEPT");
        assert_eq!(item3.unit(), "M");
        assert_eq!(item3.value(), "");
        assert_eq!(item3.description(), "DEPTH");
    }

    #[test]
    fn test_multiple_colons_timestamp_in_value() {
        let line = "TIME . 12:30:00 : LOGGING TIME";
        let item = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap();
        assert_eq!(item.mnemonic(), "TIME");
        assert_eq!(item.unit(), "");
        assert_eq!(item.value(), "12:30:00");
        assert_eq!(item.description(), "LOGGING TIME");

        let line_short = "TIME . 12:30 : LOGGING TIME";
        let item_short = HeaderItem::parse(line_short, 2, ValidationMode::Strict).unwrap();
        assert_eq!(item_short.value(), "12:30");
        assert_eq!(item_short.description(), "LOGGING TIME");

        let line_iso = "DATE . 2024-01-01 14:30:00 : LOG DATE AND TIME";
        let item_iso = HeaderItem::parse(line_iso, 3, ValidationMode::Strict).unwrap();
        assert_eq!(item_iso.value(), "2024-01-01 14:30:00");
        assert_eq!(item_iso.description(), "LOG DATE AND TIME");
    }

    #[test]
    fn test_multiple_colons_in_description() {
        let line = "WELL . A_1 : NOTE: DRILL HOLE #1";
        let item = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap();
        assert_eq!(item.mnemonic(), "WELL");
        assert_eq!(item.value(), "A_1");
        assert_eq!(item.description(), "NOTE: DRILL HOLE #1");

        let line2 = "NOTE . : PART 1: TEST; PART 2: FINAL";
        let item2 = HeaderItem::parse(line2, 2, ValidationMode::Strict).unwrap();
        assert_eq!(item2.mnemonic(), "NOTE");
        assert_eq!(item2.value(), "");
        assert_eq!(item2.description(), "PART 1: TEST; PART 2: FINAL");
    }

    #[test]
    fn test_multiple_colons_time_in_value_and_colon_in_description() {
        let line = "TIME . 12:30:00 : TIME: 12:30:00";
        let item = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap();
        assert_eq!(item.mnemonic(), "TIME");
        assert_eq!(item.value(), "12:30:00");
        assert_eq!(item.description(), "TIME: 12:30:00");
    }

    #[test]
    fn test_numeric_ratio_in_value() {
        let line = "SCALE . 1:24000 : MAP SCALE";
        let item = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap();
        assert_eq!(item.mnemonic(), "SCALE");
        assert_eq!(item.value(), "1:24000");
        assert_eq!(item.description(), "MAP SCALE");
    }

    #[test]
    fn test_quoted_value() {
        let line = "WELL . \"WELL: #1\" : WELL NAME";
        let item = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap();
        assert_eq!(item.mnemonic(), "WELL");
        assert_eq!(item.value(), "\"WELL: #1\"");
        assert_eq!(item.unquoted_value(), "WELL: #1");
        assert_eq!(item.description(), "WELL NAME");
    }

    #[test]
    fn test_strict_rejects_missing_dot() {
        let line = "STRT M 1670.0000 : START DEPTH";
        let err = HeaderItem::parse(line, 5, ValidationMode::Strict).unwrap_err();
        assert!(matches!(
            err,
            HeaderError::MissingDot { line_number: 5, .. }
        ));
    }

    #[test]
    fn test_strict_rejects_empty_mnemonic() {
        let line = ".M 1670.0000 : START DEPTH";
        let err = HeaderItem::parse(line, 10, ValidationMode::Strict).unwrap_err();
        assert!(matches!(
            err,
            HeaderError::EmptyMnemonic {
                line_number: 10,
                ..
            }
        ));

        let line_spaces = "   .M 1670.0000 : START DEPTH";
        let err2 = HeaderItem::parse(line_spaces, 11, ValidationMode::Strict).unwrap_err();
        assert!(matches!(
            err2,
            HeaderError::EmptyMnemonic {
                line_number: 11,
                ..
            }
        ));
    }

    #[test]
    fn test_strict_rejects_missing_colon() {
        let line = "STRT .M 1670.0000";
        let err = HeaderItem::parse(line, 15, ValidationMode::Strict).unwrap_err();
        assert!(matches!(
            err,
            HeaderError::MissingColon {
                line_number: 15,
                ..
            }
        ));
    }

    #[test]
    fn test_permissive_allows_missing_colon() {
        let line = "STRT .M 1670.0000";
        let item = HeaderItem::parse(line, 15, ValidationMode::Permissive).unwrap();
        assert_eq!(item.mnemonic(), "STRT");
        assert_eq!(item.unit(), "M");
        assert_eq!(item.value(), "1670.0000");
        assert_eq!(item.description(), "");
    }

    #[test]
    fn test_whitespace_in_mnemonic_strict_vs_permissive() {
        let line = "SET NAME . 123 : SET IDENTIFIER";
        // Strict mode rejects internal whitespace
        let err = HeaderItem::parse(line, 1, ValidationMode::Strict).unwrap_err();
        assert!(matches!(err, HeaderError::InvalidMnemonic { .. }));

        // Permissive mode accepts it
        let item = HeaderItem::parse(line, 1, ValidationMode::Permissive).unwrap();
        assert_eq!(item.mnemonic(), "SET NAME");
        assert_eq!(item.unit(), "");
        assert_eq!(item.value(), "123");
        assert_eq!(item.description(), "SET IDENTIFIER");
    }

    #[test]
    fn test_rejects_comments_and_delimiters() {
        let comment = "# This is a comment";
        let err1 = HeaderItem::parse(comment, 1, ValidationMode::Strict).unwrap_err();
        assert!(matches!(err1, HeaderError::CommentLine { .. }));

        let section = "~WELL INFORMATION";
        let err2 = HeaderItem::parse(section, 2, ValidationMode::Strict).unwrap_err();
        assert!(matches!(err2, HeaderError::SectionDelimiterLine { .. }));

        let blank = "   \r\n";
        let err3 = HeaderItem::parse(blank, 3, ValidationMode::Strict).unwrap_err();
        assert!(matches!(err3, HeaderError::EmptyLine { .. }));
    }

    #[test]
    fn test_type_conversions() {
        let item_f = HeaderItem::new_item("STRT", "M", "1670.5", "DEPTH", 1, "");
        assert_eq!(item_f.as_float(), Some(1670.5));
        assert_eq!(item_f.as_int(), None);

        let item_i = HeaderItem::new_item("COUNT", "", "42", "COUNT", 1, "");
        assert_eq!(item_i.as_int(), Some(42));
        assert_eq!(item_i.as_float(), Some(42.0));

        let item_b1 = HeaderItem::new_item("WRAP", "", "YES", "FLAG", 1, "");
        assert_eq!(item_b1.as_bool(), Some(true));

        let item_b2 = HeaderItem::new_item("WRAP", "", "NO", "FLAG", 1, "");
        assert_eq!(item_b2.as_bool(), Some(false));

        let item_b3 = HeaderItem::new_item("FLAG", "", "1", "FLAG", 1, "");
        assert_eq!(item_b3.as_bool(), Some(true));

        let item_b4 = HeaderItem::new_item("FLAG", "", "0", "FLAG", 1, "");
        assert_eq!(item_b4.as_bool(), Some(false));
    }

    #[test]
    fn test_batch_header_parser() {
        let parser = HeaderParser::new(ValidationMode::Strict);
        let lines = vec![
            "# Header block".to_string(),
            "".to_string(),
            "STRT .M 100.0 : START DEPTH".to_string(),
            "STOP .M 200.0 : STOP DEPTH".to_string(),
            "STEP .M 0.5 : STEP".to_string(),
        ];

        let items = parser.parse_lines(lines, 10).unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].mnemonic(), "STRT");
        assert_eq!(items[0].line_number(), 12);
        assert_eq!(items[1].mnemonic(), "STOP");
        assert_eq!(items[2].mnemonic(), "STEP");
    }

    #[test]
    fn test_formatting() {
        let item = HeaderItem::new_item("STRT", "M", "1670.0", "START DEPTH", 1, "");
        let line = item.to_line();
        assert!(line.contains("STRT"));
        assert!(line.contains(".M"));
        assert!(line.contains("1670.0"));
        assert!(line.contains(": START DEPTH"));

        // Roundtrip parsing of formatted line
        let roundtrip = HeaderItem::parse(&line, 1, ValidationMode::Strict).unwrap();
        assert_eq!(roundtrip.mnemonic(), item.mnemonic());
        assert_eq!(roundtrip.unit(), item.unit());
        assert_eq!(roundtrip.value(), item.value());
        assert_eq!(roundtrip.description(), item.description());
    }
}
