//! Dynamic NULL sentinel detection and IEEE 754 NaN mapping.
//!
//! In the Canadian Well Logging Society (CWLS) Log ASCII Standard (LAS) format,
//! missing, null, or invalid numerical measurements are designated by a sentinel value
//! defined in the `~Well Information` section:
//!
//! ```text
//! NULL . -999.2500 : NULL VALUE
//! ```
//!
//! The official CWLS LAS default sentinel is `-999.25`, but real-world files frequently
//! use `-9999`, `-999`, `-9999.25`, `-32767`, or vendor-specific variants.
//!
//! This module provides:
//! - **Dynamic NULL sentinel detection**: Automatically extracts the sentinel from
//!   the `~W` section's `NULL` record, with fallback to standard CWLS `-999.25`.
//! - **IEEE 754 `NaN` mapping**: Maps declared sentinel values to and from IEEE 754 `NaN`
//!   (`f64::NAN`), aligning with `NumPy` (`np.nan`), `Pandas`, and modern scientific workflows.
//! - **Configurable NULL policies**: `Strict`, `Common`, `Aggressive`, and `None` policies
//!   for handling clean, noisy, or legacy LAS files.
//! - **In-place array conversions and line tokenizers**: High-performance vectorized
//!   transformations between NULL sentinels and `NaN`.

use pyo3::prelude::*;
use std::fmt;

use crate::header::HeaderItem;

/// Common industry null sentinels used across standard logging software.
pub const COMMON_SENTINELS: [f64; 8] = [
    -999.25, 999.25, -9999.0, 9999.0, -999.0, 999.0, -9999.25, 9999.25,
];

/// Additional vendor/legacy sentinels recognized in aggressive substitution mode.
pub const AGGRESSIVE_SENTINELS: [f64; 9] = [
    -999.99,
    999.99,
    -9999.99,
    9999.99,
    -32767.0,
    32767.0,
    -2_147_483_647.0,
    2_147_483_647.0,
    -0.0,
];

/// Canonical CWLS LAS default null sentinel value.
pub const DEFAULT_CWLS_NULL: f64 = -999.25;

/// Default floating-point comparison tolerance for null detection.
pub const DEFAULT_TOLERANCE: f64 = 1e-5;

/// Policy controlling how missing/null values are identified and mapped to `NaN`.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NullPolicy {
    /// Strict policy: Maps only the explicitly declared NULL sentinel from the file
    /// (or `-999.25` default if undeclared) plus IEEE 754 `NaN`.
    #[default]
    Strict,

    /// Common policy: Maps the declared sentinel plus common LAS variants
    /// (`-999.25`, `999.25`, `-9999`, `9999`, `-999`, `999`).
    Common,

    /// Aggressive policy: Maps declared sentinel plus extended vendor and legacy sentinels
    /// (`-32767`, `32767`, `-999.99`, `-9999.99`, `-2147483647`, `-0.0`).
    Aggressive,

    /// None policy: Disables automatic sentinel conversion (only literal `NaN` is preserved).
    None,
}

impl NullPolicy {
    /// Returns `true` if this is strict mode.
    #[must_use]
    pub const fn is_strict(&self) -> bool {
        matches!(self, Self::Strict)
    }

    /// Returns `true` if this is common mode.
    #[must_use]
    pub const fn is_common(&self) -> bool {
        matches!(self, Self::Common)
    }

    /// Returns `true` if this is aggressive mode.
    #[must_use]
    pub const fn is_aggressive(&self) -> bool {
        matches!(self, Self::Aggressive)
    }

    /// Returns `true` if this is none mode.
    #[must_use]
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
#[pymethods]
impl NullPolicy {
    /// Returns `true` if strict policy.
    #[getter]
    fn get_is_strict(&self) -> bool {
        self.is_strict()
    }

    /// Returns `true` if common policy.
    #[getter]
    fn get_is_common(&self) -> bool {
        self.is_common()
    }

    /// Returns `true` if aggressive policy.
    #[getter]
    fn get_is_aggressive(&self) -> bool {
        self.is_aggressive()
    }

    /// Returns `true` if none policy.
    #[getter]
    fn get_is_none(&self) -> bool {
        self.is_none()
    }

    fn __repr__(&self) -> String {
        match self {
            Self::Strict => "NullPolicy.Strict".to_string(),
            Self::Common => "NullPolicy.Common".to_string(),
            Self::Aggressive => "NullPolicy.Aggressive".to_string(),
            Self::None => "NullPolicy.None".to_string(),
        }
    }
}

/// Dynamically manages the NULL sentinel value and IEEE 754 `NaN` transformations.
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct NullSentinel {
    value: f64,
    raw_sentinel: Option<String>,
    policy: NullPolicy,
    tolerance: f64,
    custom_substitutes: Vec<f64>,
}

impl NullSentinel {
    /// Creates a new `NullSentinel` with the given numeric value and strict policy.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self {
            value,
            raw_sentinel: None,
            policy: NullPolicy::Strict,
            tolerance: DEFAULT_TOLERANCE,
            custom_substitutes: Vec::new(),
        }
    }

    /// Creates a default CWLS LAS sentinel (`-999.25`).
    #[must_use]
    pub fn default_cwls() -> Self {
        Self {
            value: DEFAULT_CWLS_NULL,
            raw_sentinel: Some("-999.25".to_string()),
            policy: NullPolicy::Strict,
            tolerance: DEFAULT_TOLERANCE,
            custom_substitutes: Vec::new(),
        }
    }

    /// Sets the null policy.
    #[must_use]
    pub const fn with_policy(mut self, policy: NullPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sets the float comparison tolerance.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the raw sentinel string representation.
    #[must_use]
    pub fn with_raw_sentinel(mut self, raw: impl Into<String>) -> Self {
        self.raw_sentinel = Some(raw.into());
        self
    }

    /// Appends a custom substitute float value.
    #[must_use]
    pub fn with_custom_substitute(mut self, sub: f64) -> Self {
        self.custom_substitutes.push(sub);
        self
    }

    /// Returns the primary numerical sentinel value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Updates the primary numerical sentinel value.
    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }

    /// Returns the original raw sentinel string, if recorded.
    #[must_use]
    pub fn raw_sentinel(&self) -> Option<&str> {
        self.raw_sentinel.as_deref()
    }

    /// Updates the raw sentinel string.
    pub fn set_raw_sentinel(&mut self, raw: Option<String>) {
        self.raw_sentinel = raw;
    }

    /// Returns the active null policy.
    #[must_use]
    pub const fn policy(&self) -> NullPolicy {
        self.policy
    }

    /// Updates the active null policy.
    pub fn set_policy(&mut self, policy: NullPolicy) {
        self.policy = policy;
    }

    /// Returns the comparison tolerance.
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Updates the comparison tolerance.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    /// Returns the list of custom substitutes.
    #[must_use]
    pub fn custom_substitutes(&self) -> &[f64] {
        &self.custom_substitutes
    }

    /// Adds a custom substitute value.
    pub fn add_custom_substitute(&mut self, sub: f64) {
        self.custom_substitutes.push(sub);
    }

    /// Checks whether `val` matches the target within configured tolerance.
    #[must_use]
    pub fn matches_value(&self, val: f64, target: f64) -> bool {
        if (val - target).abs() == 0.0 {
            return true;
        }
        if self.tolerance > 0.0 {
            let scale = target.abs().max(1.0);
            (val - target).abs() <= self.tolerance * scale
        } else {
            false
        }
    }

    /// Checks whether `val` is considered a null / missing value.
    ///
    /// Returns `true` if `val` is already IEEE 754 `NaN`, or matches the sentinel
    /// under the current policy and tolerance rules.
    #[must_use]
    pub fn is_null(&self, val: f64) -> bool {
        if val.is_nan() {
            return true;
        }
        if self.policy == NullPolicy::None {
            return false;
        }

        if self.matches_value(val, self.value) {
            return true;
        }

        for &sub in &self.custom_substitutes {
            if self.matches_value(val, sub) {
                return true;
            }
        }

        match self.policy {
            NullPolicy::Strict | NullPolicy::None => false,
            NullPolicy::Common => {
                for &common in &COMMON_SENTINELS {
                    if self.matches_value(val, common) {
                        return true;
                    }
                }
                false
            }
            NullPolicy::Aggressive => {
                for &common in &COMMON_SENTINELS {
                    if self.matches_value(val, common) {
                        return true;
                    }
                }
                for &agg in &AGGRESSIVE_SENTINELS {
                    if self.matches_value(val, agg) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Checks whether text token `s` represents a null sentinel or `NaN`.
    #[must_use]
    pub fn is_null_str(&self, s: &str) -> bool {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return true;
        }

        if trimmed.eq_ignore_ascii_case("nan")
            || trimmed.eq_ignore_ascii_case("-nan")
            || trimmed.eq_ignore_ascii_case("+nan")
            || trimmed.eq_ignore_ascii_case("null")
            || trimmed.eq_ignore_ascii_case("(null)")
            || trimmed.eq_ignore_ascii_case("none")
        {
            return true;
        }

        if let Some(ref raw) = self.raw_sentinel {
            if trimmed == raw.trim() {
                return true;
            }
        }

        if let Ok(f) = trimmed.parse::<f64>() {
            self.is_null(f)
        } else {
            false
        }
    }

    /// Maps `val` to `f64::NAN` if it represents a null sentinel; otherwise returns `val`.
    #[must_use]
    pub fn map_to_nan(&self, val: f64) -> f64 {
        if self.is_null(val) {
            f64::NAN
        } else {
            val
        }
    }

    /// Replaces `f64::NAN` with the primary sentinel value; otherwise returns `val`.
    #[must_use]
    pub fn map_from_nan(&self, val: f64) -> f64 {
        if val.is_nan() {
            self.value
        } else {
            val
        }
    }

    /// Parses string token `s` to `f64`, mapping null sentinels to `f64::NAN`.
    ///
    /// Returns `None` if `s` is not a valid float representation.
    #[must_use]
    pub fn map_str_to_nan(&self, s: &str) -> Option<f64> {
        let trimmed = s.trim();
        if self.is_null_str(trimmed) {
            Some(f64::NAN)
        } else {
            trimmed.parse::<f64>().ok()
        }
    }

    /// In-place replaces all null sentinel values in `data` with `f64::NAN`.
    pub fn replace_nulls_slice(&self, data: &mut [f64]) {
        for x in data.iter_mut() {
            if self.is_null(*x) {
                *x = f64::NAN;
            }
        }
    }

    /// In-place replaces all `NaN` values in `data` with the sentinel value.
    pub fn replace_nans_slice(&self, data: &mut [f64]) {
        for x in data.iter_mut() {
            if x.is_nan() {
                *x = self.value;
            }
        }
    }

    /// Parses an ASCII log numerical data line, mapping null sentinels to `f64::NAN`.
    ///
    /// # Errors
    /// Returns an error message string if any non-null token cannot be parsed as a float.
    pub fn parse_ascii_line(&self, line: &str) -> Result<Vec<f64>, String> {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return Ok(Vec::new());
        }

        let tokens = trimmed.split_whitespace();
        let mut values = Vec::new();

        for token in tokens {
            if let Some(val) = self.map_str_to_nan(token) {
                values.push(val);
            } else {
                return Err(format!("Invalid numerical token in ASCII data: '{token}'"));
            }
        }

        Ok(values)
    }

    /// Formats the sentinel into a string for LAS file output.
    #[must_use]
    pub fn format_null(&self) -> String {
        self.raw_sentinel
            .clone()
            .unwrap_or_else(|| format!("{:.4}", self.value))
    }

    /// Attempts to construct a `NullSentinel` from a `HeaderItem`.
    ///
    /// Returns `Some(NullSentinel)` if `item.mnemonic` equals `"NULL"` (case-insensitive)
    /// and `item.value` parses to an `f64`.
    #[must_use]
    pub fn from_header_item(item: &HeaderItem) -> Option<Self> {
        if item.mnemonic().eq_ignore_ascii_case("NULL") {
            let val = item.as_float()?;
            Some(Self {
                value: val,
                raw_sentinel: Some(item.value().to_string()),
                policy: NullPolicy::Strict,
                tolerance: DEFAULT_TOLERANCE,
                custom_substitutes: Vec::new(),
            })
        } else {
            None
        }
    }

    /// Scans a collection of header lines to dynamically detect the NULL sentinel record.
    #[must_use]
    pub fn detect(lines: &[String]) -> Option<Self> {
        for line in lines {
            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('~') {
                continue;
            }
            if let Ok(item) =
                HeaderItem::parse(line, 1, crate::state_machine::ValidationMode::Permissive)
            {
                if let Some(sentinel) = Self::from_header_item(&item) {
                    return Some(sentinel);
                }
            }
        }
        None
    }
}

#[pymethods]
impl NullSentinel {
    /// Creates a new `NullSentinel`.
    #[new]
    #[pyo3(signature = (value=DEFAULT_CWLS_NULL, policy=None, tolerance=DEFAULT_TOLERANCE, raw_sentinel=None))]
    fn py_new(
        value: f64,
        policy: Option<NullPolicy>,
        tolerance: f64,
        raw_sentinel: Option<String>,
    ) -> Self {
        Self {
            value,
            raw_sentinel,
            policy: policy.unwrap_or(NullPolicy::Strict),
            tolerance,
            custom_substitutes: Vec::new(),
        }
    }

    /// The numerical sentinel value.
    #[getter(value)]
    fn get_value(&self) -> f64 {
        self.value
    }

    /// Sets the numerical sentinel value.
    #[setter(value)]
    fn set_value_py(&mut self, val: f64) {
        self.set_value(val);
    }

    /// The original raw sentinel string, if available.
    #[getter(raw_sentinel)]
    fn get_raw_sentinel(&self) -> Option<String> {
        self.raw_sentinel.clone()
    }

    /// Sets the raw sentinel string.
    #[setter(raw_sentinel)]
    fn set_raw_sentinel_py(&mut self, raw: Option<String>) {
        self.set_raw_sentinel(raw);
    }

    /// The active null policy.
    #[getter(policy)]
    fn get_policy(&self) -> NullPolicy {
        self.policy
    }

    /// Sets the null policy.
    #[setter(policy)]
    fn set_policy_py(&mut self, policy: NullPolicy) {
        self.set_policy(policy);
    }

    /// The comparison tolerance.
    #[getter(tolerance)]
    fn get_tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Sets the comparison tolerance.
    #[setter(tolerance)]
    fn set_tolerance_py(&mut self, tol: f64) {
        self.set_tolerance(tol);
    }

    /// Returns custom substitutes.
    #[getter(custom_substitutes)]
    fn get_custom_substitutes(&self) -> Vec<f64> {
        self.custom_substitutes.clone()
    }

    /// Adds a custom substitute value.
    fn add_substitute(&mut self, sub: f64) {
        self.add_custom_substitute(sub);
    }

    /// Returns `true` if `val` is considered null.
    #[pyo3(name = "is_null")]
    fn py_is_null(&self, val: f64) -> bool {
        self.is_null(val)
    }

    /// Returns `true` if string `s` represents a null value or `NaN`.
    #[pyo3(name = "is_null_str")]
    fn py_is_null_str(&self, s: &str) -> bool {
        self.is_null_str(s)
    }

    /// Maps `val` to `NaN` if null, else returns `val`.
    #[pyo3(name = "map_to_nan")]
    fn py_map_to_nan(&self, val: f64) -> f64 {
        self.map_to_nan(val)
    }

    /// Replaces `NaN` with sentinel value.
    #[pyo3(name = "map_from_nan")]
    fn py_map_from_nan(&self, val: f64) -> f64 {
        self.map_from_nan(val)
    }

    /// Parses string `s` to float, mapping nulls to `NaN`.
    #[pyo3(name = "map_str_to_nan")]
    fn py_map_str_to_nan(&self, s: &str) -> Option<f64> {
        self.map_str_to_nan(s)
    }

    /// Replaces all null sentinels in a list with `NaN`.
    fn replace_nulls(&self, mut data: Vec<f64>) -> Vec<f64> {
        self.replace_nulls_slice(&mut data);
        data
    }

    /// Replaces all `NaN` values in a list with the sentinel value.
    fn replace_nans(&self, mut data: Vec<f64>) -> Vec<f64> {
        self.replace_nans_slice(&mut data);
        data
    }

    /// Parses an ASCII log data line into a list of floats, mapping nulls to `NaN`.
    ///
    /// # Errors
    /// Returns `PyValueError` if any non-null token cannot be parsed as a float.
    fn parse_line(&self, line: &str) -> PyResult<Vec<f64>> {
        self.parse_ascii_line(line)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Formats the sentinel value into a string.
    #[pyo3(name = "format_null")]
    fn py_format_null(&self) -> String {
        self.format_null()
    }

    /// Creates a default CWLS sentinel (`-999.25`).
    #[staticmethod]
    #[pyo3(name = "default_cwls")]
    fn py_default_cwls() -> Self {
        Self::default_cwls()
    }

    /// Constructs a `NullSentinel` from a `HeaderItem` if its mnemonic is `NULL`.
    #[staticmethod]
    #[pyo3(name = "from_item")]
    fn py_from_item(item: &HeaderItem) -> Option<Self> {
        Self::from_header_item(item)
    }

    /// Scans lines to dynamically detect the NULL sentinel record.
    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    #[pyo3(name = "detect")]
    fn py_detect(lines: Vec<String>) -> Option<Self> {
        Self::detect(&lines)
    }

    fn __repr__(&self) -> String {
        format!(
            "NullSentinel(value={}, policy={:?}, tolerance={})",
            self.value, self.policy, self.tolerance
        )
    }

    fn __str__(&self) -> String {
        self.format_null()
    }
}

impl Default for NullSentinel {
    fn default() -> Self {
        Self::default_cwls()
    }
}

impl fmt::Display for NullSentinel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_null())
    }
}

/// Standalone helper: Maps `val` to `f64::NAN` if matching `sentinel` (default `-999.25`).
#[must_use]
#[pyfunction]
#[pyo3(signature = (val, sentinel=DEFAULT_CWLS_NULL))]
pub fn map_to_nan(val: f64, sentinel: f64) -> f64 {
    if val.is_nan() || (val - sentinel).abs() <= DEFAULT_TOLERANCE * sentinel.abs().max(1.0) {
        f64::NAN
    } else {
        val
    }
}

/// Standalone helper: Replaces `NaN` with `sentinel` (default `-999.25`).
#[must_use]
#[pyfunction]
#[pyo3(signature = (val, sentinel=DEFAULT_CWLS_NULL))]
pub fn map_from_nan(val: f64, sentinel: f64) -> f64 {
    if val.is_nan() {
        sentinel
    } else {
        val
    }
}

/// Standalone helper: Returns `true` if `val` is `NaN` or matches `sentinel` (default `-999.25`).
#[must_use]
#[pyfunction]
#[pyo3(signature = (val, sentinel=DEFAULT_CWLS_NULL))]
pub fn is_null(val: f64, sentinel: f64) -> bool {
    val.is_nan() || (val - sentinel).abs() <= DEFAULT_TOLERANCE * sentinel.abs().max(1.0)
}

#[cfg(test)]
#[allow(clippy::pedantic)]
mod tests {
    use super::*;

    #[test]
    fn test_default_cwls_sentinel() {
        let ns = NullSentinel::default_cwls();
        assert_eq!(ns.value(), -999.25);
        assert!(ns.is_null(-999.25));
        assert!(ns.is_null(f64::NAN));
        assert!(!ns.is_null(1670.0));
        assert!(!ns.is_null(0.0));

        let mapped = ns.map_to_nan(-999.25);
        assert!(mapped.is_nan());

        let kept = ns.map_to_nan(1670.0);
        assert_eq!(kept, 1670.0);
    }

    #[test]
    fn test_custom_sentinel() {
        let ns = NullSentinel::new(-9999.0).with_raw_sentinel("-9999");
        assert_eq!(ns.value(), -9999.0);
        assert!(ns.is_null(-9999.0));
        assert!(ns.is_null_str("-9999"));
        assert!(!ns.is_null(-999.25)); // Strict mode does not match -999.25 if sentinel is -9999
    }

    #[test]
    fn test_policies() {
        let strict = NullSentinel::new(-9999.0).with_policy(NullPolicy::Strict);
        assert!(strict.is_null(-9999.0));
        assert!(!strict.is_null(-999.25));

        let common = NullSentinel::new(-9999.0).with_policy(NullPolicy::Common);
        assert!(common.is_null(-9999.0));
        assert!(common.is_null(-999.25)); // Included in common
        assert!(common.is_null(999.25));

        let agg = NullSentinel::new(-9999.0).with_policy(NullPolicy::Aggressive);
        assert!(agg.is_null(-32767.0)); // Included in aggressive
        assert!(agg.is_null(-999.99));

        let none = NullSentinel::new(-999.25).with_policy(NullPolicy::None);
        assert!(!none.is_null(-999.25)); // None policy ignores sentinel
        assert!(none.is_null(f64::NAN)); // Still respects literal NaN
    }

    #[test]
    fn test_string_detection() {
        let ns = NullSentinel::default_cwls();
        assert!(ns.is_null_str("-999.25"));
        assert!(ns.is_null_str("-999.2500"));
        assert!(ns.is_null_str("NaN"));
        assert!(ns.is_null_str("nan"));
        assert!(ns.is_null_str("NULL"));
        assert!(ns.is_null_str(""));
        assert!(!ns.is_null_str("1670.0"));
    }

    #[test]
    fn test_slice_replacements() {
        let ns = NullSentinel::default_cwls();
        let mut data: Vec<f64> = vec![100.0, -999.25, 200.0, -999.25, 300.0];
        ns.replace_nulls_slice(&mut data);
        assert_eq!(data[0], 100.0);
        assert!(data[1].is_nan());
        assert_eq!(data[2], 200.0);
        assert!(data[3].is_nan());
        assert_eq!(data[4], 300.0);

        ns.replace_nans_slice(&mut data);
        assert_eq!(data[1], -999.25);
        assert_eq!(data[3], -999.25);
    }

    #[test]
    fn test_ascii_line_parsing() {
        let ns = NullSentinel::default_cwls();
        let line = "  1670.000   2550.000  -999.2500   0.250  ";
        let vals = ns.parse_ascii_line(line).unwrap();
        assert_eq!(vals.len(), 4);
        assert_eq!(vals[0], 1670.0);
        assert_eq!(vals[1], 2550.0);
        assert!(vals[2].is_nan());
        assert_eq!(vals[3], 0.25);
    }

    #[test]
    fn test_dynamic_detection_from_header_item() {
        let line = "NULL . -9999.0000 : NULL VALUE";
        let item =
            HeaderItem::parse(line, 1, crate::state_machine::ValidationMode::Strict).unwrap();
        let ns = NullSentinel::from_header_item(&item).unwrap();
        assert_eq!(ns.value(), -9999.0);
        assert_eq!(ns.raw_sentinel(), Some("-9999.0000"));
        assert!(ns.is_null(-9999.0));
    }

    #[test]
    fn test_dynamic_detection_from_lines() {
        let lines = vec![
            "STRT .M 100.0 : START".to_string(),
            "STOP .M 200.0 : STOP".to_string(),
            "NULL . -999.2500 : NULL VALUE".to_string(),
            "WELL . TEST : WELL".to_string(),
        ];
        let ns = NullSentinel::detect(&lines).unwrap();
        assert_eq!(ns.value(), -999.25);
        assert_eq!(ns.raw_sentinel(), Some("-999.2500"));
    }

    #[test]
    fn test_standalone_helpers() {
        assert!(is_null(-999.25, -999.25));
        assert!(is_null(f64::NAN, -999.25));
        assert!(!is_null(100.0, -999.25));

        assert!(map_to_nan(-999.25, -999.25).is_nan());
        assert_eq!(map_to_nan(100.0, -999.25), 100.0);

        assert_eq!(map_from_nan(f64::NAN, -999.25), -999.25);
        assert_eq!(map_from_nan(100.0, -999.25), 100.0);
    }
}
