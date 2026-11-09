//! CWLS LAS section types and delimiter parsing.
//!
//! The Canadian Well Logging Society (CWLS) Log ASCII Standard (LAS) format
//! defines six standard section delimiters:
//! - `~V` : Version and format information (Required, must be first)
//! - `~W` : Well identification and metadata (Required, second)
//! - `~C` : Curve definitions and mnemonics (Required, before `~A`)
//! - `~P` : Logging parameters and constants (Optional)
//! - `~O` : Other information and remarks (Optional)
//! - `~A` : ASCII log numerical data (Required, must be last)

use pyo3::prelude::*;
use std::fmt;

use crate::error::StateMachineError;
use crate::state_machine::ValidationMode;

/// Enumeration of standard CWLS LAS section types.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionType {
    /// `~V` - Version information section.
    Version,
    /// `~W` - Well information section.
    Well,
    /// `~C` - Curve information section.
    Curve,
    /// `~P` - Parameter information section.
    Parameter,
    /// `~O` - Other remarks / comments section.
    Other,
    /// `~A` - ASCII log data section.
    Ascii,
}

impl SectionType {
    /// Array containing all six standard CWLS sections in their conventional order.
    pub const ALL_CWLS: [Self; 6] = [
        Self::Version,
        Self::Well,
        Self::Curve,
        Self::Parameter,
        Self::Other,
        Self::Ascii,
    ];

    /// Returns the single-letter section character code (e.g. `'V'`, `'W'`).
    #[must_use]
    pub const fn char_code(&self) -> char {
        match self {
            Self::Version => 'V',
            Self::Well => 'W',
            Self::Curve => 'C',
            Self::Parameter => 'P',
            Self::Other => 'O',
            Self::Ascii => 'A',
        }
    }

    /// Returns the canonical delimiter prefix string (e.g. `"~V"`, `"~W"`).
    #[must_use]
    pub const fn delimiter(&self) -> &'static str {
        match self {
            Self::Version => "~V",
            Self::Well => "~W",
            Self::Curve => "~C",
            Self::Parameter => "~P",
            Self::Other => "~O",
            Self::Ascii => "~A",
        }
    }

    /// Returns the canonical uppercase name of the section.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Version => "VERSION",
            Self::Well => "WELL",
            Self::Curve => "CURVE",
            Self::Parameter => "PARAMETER",
            Self::Other => "OTHER",
            Self::Ascii => "ASCII",
        }
    }

    /// Returns a human-readable description of the section.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Version => "Version and format information",
            Self::Well => "Well identification information",
            Self::Curve => "Curve definitions and mnemonics",
            Self::Parameter => "Logging parameters and constants",
            Self::Other => "Other information and remarks",
            Self::Ascii => "ASCII tabular log data",
        }
    }

    /// Returns `true` if this section is required by CWLS LAS 2.0 specifications.
    #[must_use]
    pub const fn is_mandatory(&self) -> bool {
        matches!(self, Self::Version | Self::Well | Self::Curve | Self::Ascii)
    }

    /// Returns `true` if this section is optional in CWLS LAS specifications.
    #[must_use]
    pub const fn is_optional(&self) -> bool {
        matches!(self, Self::Parameter | Self::Other)
    }

    /// Returns `true` if this section contains structured header records (`MNEM.UNIT DATA: DESC`).
    #[must_use]
    pub const fn is_header_section(&self) -> bool {
        matches!(
            self,
            Self::Version | Self::Well | Self::Curve | Self::Parameter
        )
    }

    /// Returns `true` if this section contains numerical tabular log measurements.
    #[must_use]
    pub const fn is_data_section(&self) -> bool {
        matches!(self, Self::Ascii)
    }

    /// Returns `true` if this section contains free-form text remarks.
    #[must_use]
    pub const fn is_remarks_section(&self) -> bool {
        matches!(self, Self::Other)
    }

    /// Attempts to parse a section type from a single character identifier.
    ///
    /// # Errors
    ///
    /// Returns `StateMachineError::UnknownSection` if `c` is not one of `V`, `W`, `C`, `P`, `O`, `A` (case-insensitive).
    pub fn from_char(c: char, line_number: usize, raw: &str) -> Result<Self, StateMachineError> {
        match c.to_ascii_uppercase() {
            'V' => Ok(Self::Version),
            'W' => Ok(Self::Well),
            'C' => Ok(Self::Curve),
            'P' => Ok(Self::Parameter),
            'O' => Ok(Self::Other),
            'A' => Ok(Self::Ascii),
            code => Err(StateMachineError::UnknownSection {
                code,
                line_number,
                raw: raw.to_string(),
            }),
        }
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
#[pymethods]
impl SectionType {
    /// Returns the character code representation of the section.
    #[getter]
    fn get_char_code(&self) -> String {
        self.char_code().to_string()
    }

    /// Returns the canonical delimiter prefix (e.g. `"~V"`).
    #[getter]
    fn get_delimiter(&self) -> &'static str {
        self.delimiter()
    }

    /// Returns the section name.
    #[getter]
    fn get_name(&self) -> &'static str {
        self.name()
    }

    /// Returns the section description.
    #[getter]
    fn get_description(&self) -> &'static str {
        self.description()
    }

    /// Returns whether this section is mandatory in LAS 2.0.
    #[getter]
    fn get_is_mandatory(&self) -> bool {
        self.is_mandatory()
    }

    /// Returns whether this section is optional in LAS 2.0.
    #[getter]
    fn get_is_optional(&self) -> bool {
        self.is_optional()
    }

    /// Returns whether this is a structured header metadata section.
    #[getter]
    fn get_is_header_section(&self) -> bool {
        self.is_header_section()
    }

    /// Returns whether this is the ASCII tabular data section.
    #[getter]
    fn get_is_data_section(&self) -> bool {
        self.is_data_section()
    }

    /// Returns whether this is the free-text remarks section.
    #[getter]
    fn get_is_remarks_section(&self) -> bool {
        self.is_remarks_section()
    }

    fn __repr__(&self) -> String {
        format!("SectionType.{}", self.name())
    }

    fn __str__(&self) -> &'static str {
        self.delimiter()
    }
}

impl fmt::Display for SectionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.delimiter())
    }
}

/// Parses a section header following the leading `~` character.
///
/// Supports CWLS LAS 1.2, 2.0 (`~V`, `~W`, `~C`, `~P`, `~O`, `~A`)
/// and LAS 3.0 compound section naming conventions (`~Log_Definition`, `~Log_Data`,
/// `~Core_Definition`, `~Core_Data`, `~Inclinometry_Parameter`, etc.).
///
/// Returns a tuple of `(SectionType, section_name, title)`.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn parse_section_header(after_tilde: &str) -> Option<(SectionType, String, String)> {
    let trimmed = after_tilde.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Split on first whitespace or '|' to separate keyword from title/remarks
    let (kw_str, title_str) = match trimmed.split_once(|c: char| c.is_whitespace() || c == '|') {
        Some((kw, rest)) => (kw.trim(), rest.trim().trim_start_matches('|').trim()),
        None => (trimmed, ""),
    };

    let kw_upper = kw_str.to_ascii_uppercase();

    // 1. Check for LAS 3.0 compound suffixes (_Definition, -Definition, .Definition, _Data, etc.)
    let def_pos = kw_upper
        .find("_DEFINITION")
        .or_else(|| kw_upper.find("-DEFINITION"))
        .or_else(|| kw_upper.find(".DEFINITION"))
        .or_else(|| kw_upper.find("_DEF"))
        .or_else(|| kw_upper.find("-DEF"))
        .or_else(|| kw_upper.find(".DEF"));
    if let Some(pos) = def_pos {
        let sec_name = &kw_str[..pos];
        return Some((
            SectionType::Curve,
            sec_name.to_ascii_uppercase(),
            title_str.to_string(),
        ));
    }

    let data_pos = kw_upper
        .find("_DATA")
        .or_else(|| kw_upper.find("-DATA"))
        .or_else(|| kw_upper.find(".DATA"));
    if let Some(pos) = data_pos {
        let sec_name = &kw_str[..pos];
        return Some((
            SectionType::Ascii,
            sec_name.to_ascii_uppercase(),
            title_str.to_string(),
        ));
    }

    let param_pos = kw_upper
        .find("_PARAMETER")
        .or_else(|| kw_upper.find("-PARAMETER"))
        .or_else(|| kw_upper.find(".PARAMETER"))
        .or_else(|| kw_upper.find("_PARAM"))
        .or_else(|| kw_upper.find("-PARAM"))
        .or_else(|| kw_upper.find(".PARAM"));
    if let Some(pos) = param_pos {
        let sec_name = &kw_str[..pos];
        return Some((
            SectionType::Parameter,
            sec_name.to_ascii_uppercase(),
            title_str.to_string(),
        ));
    }

    // 2. Standard CWLS 1.2 / 2.0 section prefixes
    if kw_upper.starts_with('V') {
        let title = if kw_upper.starts_with("VERSION") {
            let rem = kw_str["VERSION".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else if kw_upper.starts_with("VERS") {
            let rem = kw_str["VERS".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else {
            let rem = kw_str[1..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        };
        return Some((SectionType::Version, "VERSION".to_string(), title));
    }
    if kw_upper.starts_with('W') {
        let title = if kw_upper.starts_with("WELL") {
            let rem = kw_str["WELL".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else {
            let rem = kw_str[1..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        };
        return Some((SectionType::Well, "WELL".to_string(), title));
    }
    if kw_upper.starts_with('C') && !kw_upper.contains('_') && !kw_upper.contains('-') {
        let title = if kw_upper.starts_with("CURVES") {
            let rem = kw_str["CURVES".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else if kw_upper.starts_with("CURVE") {
            let rem = kw_str["CURVE".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else {
            let rem = kw_str[1..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        };
        return Some((SectionType::Curve, "CURVE".to_string(), title));
    }
    if kw_upper.starts_with('P') && !kw_upper.contains('_') && !kw_upper.contains('-') {
        let title = if kw_upper.starts_with("PARAMETERS") {
            let rem = kw_str["PARAMETERS".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else if kw_upper.starts_with("PARAMETER") {
            let rem = kw_str["PARAMETER".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else if kw_upper.starts_with("PARAMS") {
            let rem = kw_str["PARAMS".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else {
            let rem = kw_str[1..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        };
        return Some((SectionType::Parameter, "PARAMETER".to_string(), title));
    }
    if kw_upper.starts_with('O') && !kw_upper.contains('_') && !kw_upper.contains('-') {
        let title = if kw_upper.starts_with("OTHER") {
            let rem = kw_str["OTHER".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else {
            let rem = kw_str[1..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        };
        return Some((SectionType::Other, "OTHER".to_string(), title));
    }
    if kw_upper.starts_with('A') && !kw_upper.contains('_') && !kw_upper.contains('-') {
        let title = if kw_upper.starts_with("ASCII") {
            let rem = kw_str["ASCII".len()..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        } else {
            let rem = kw_str[1..].trim();
            if rem.is_empty() {
                title_str.to_string()
            } else {
                format!("{rem} {title_str}").trim().to_string()
            }
        };
        return Some((SectionType::Ascii, "ASCII".to_string(), title));
    }

    // Unrecognized section delimiter
    None
}

/// Represents a parsed CWLS LAS section delimiter line.
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionDelimiter {
    /// The identified section type.
    section_type: SectionType,
    /// Logical section name (e.g. "LOG", "CORE", "CURVE", "ASCII", "VERSION").
    section_name: String,
    /// 1-based line number in the source file.
    line_number: usize,
    /// The raw line text as read from the source.
    raw_line: String,
    /// Title or extra parameters following the delimiter identifier.
    title: String,
}

impl SectionDelimiter {
    /// Creates a new `SectionDelimiter`.
    #[must_use]
    pub fn new(
        section_type: SectionType,
        line_number: usize,
        raw_line: String,
        title: String,
    ) -> Self {
        let section_name = section_type.name().to_string();
        Self {
            section_type,
            section_name,
            line_number,
            raw_line,
            title,
        }
    }

    /// Creates a new `SectionDelimiter` with an explicit section name.
    #[must_use]
    pub fn with_section_name(
        section_type: SectionType,
        section_name: impl Into<String>,
        line_number: usize,
        raw_line: String,
        title: String,
    ) -> Self {
        Self {
            section_type,
            section_name: section_name.into(),
            line_number,
            raw_line,
            title,
        }
    }

    /// Returns the parsed `SectionType`.
    #[must_use]
    pub const fn section_type(&self) -> SectionType {
        self.section_type
    }

    /// Returns the logical section name (e.g. `"LOG"`, `"CORE"`, `"VERSION"`).
    #[must_use]
    pub fn section_name(&self) -> &str {
        &self.section_name
    }

    /// Returns the 1-based line number where the delimiter appeared.
    #[must_use]
    pub const fn line_number(&self) -> usize {
        self.line_number
    }

    /// Returns the raw delimiter line text.
    #[must_use]
    pub fn raw_line(&self) -> &str {
        &self.raw_line
    }

    /// Returns any title or remarks that followed the section keyword on the delimiter line.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Returns `true` if `line` begins with a tilde (`~`) after optional leading whitespace.
    #[must_use]
    pub fn is_delimiter_line(line: &str) -> bool {
        line.trim_start().starts_with('~')
    }

    /// Returns `true` if `line` begins with `#` after optional leading whitespace.
    #[must_use]
    pub fn is_comment_line(line: &str) -> bool {
        line.trim_start().starts_with('#')
    }

    /// Returns `true` if `line` contains only whitespace or is empty.
    #[must_use]
    pub fn is_blank_line(line: &str) -> bool {
        line.trim().is_empty()
    }

    /// Parses a line known or suspected to be a section delimiter line.
    ///
    /// # Errors
    ///
    /// Returns `StateMachineError::InvalidDelimiter` if the line does not start with `~`
    /// or has no character following `~`.
    /// Parses a line known or suspected to be a section delimiter line with a specific validation mode.
    ///
    /// In permissive mode, unrecognized section characters (e.g. `~Z`) or custom headers (e.g. `~P_CALCULATED`)
    /// are classified as [`SectionType::Other`] with the specified section name rather than returning an error.
    pub fn parse_with_mode(
        line: &str,
        line_number: usize,
        mode: ValidationMode,
    ) -> Result<Self, StateMachineError> {
        let trimmed_start = line.trim_start();
        if !trimmed_start.starts_with('~') {
            return Err(StateMachineError::InvalidDelimiter {
                line_number,
                line: line.to_string(),
                reason: "line does not begin with '~'".to_string(),
            });
        }

        let after_tilde = &trimmed_start[1..];
        let content = after_tilde.trim_start();
        let clean_raw = line.trim_end_matches(['\r', '\n']).to_string();

        let (section_type, section_name, title) = if let Some(res) = parse_section_header(content) {
            res
        } else {
            if content.is_empty() {
                return Err(StateMachineError::InvalidDelimiter {
                    line_number,
                    line: line.to_string(),
                    reason: "missing section character after '~'".to_string(),
                });
            }
            if mode.is_permissive() {
                let sec_name = content
                    .split_whitespace()
                    .next()
                    .unwrap_or("OTHER")
                    .trim_start_matches('~')
                    .to_ascii_uppercase();
                (SectionType::Other, sec_name, String::new())
            } else {
                let first_char = content.chars().next().unwrap_or('?');
                return Err(StateMachineError::UnknownSection {
                    code: first_char.to_ascii_uppercase(),
                    line_number,
                    raw: line.to_string(),
                });
            }
        };

        Ok(Self {
            section_type,
            section_name,
            line_number,
            raw_line: clean_raw,
            title,
        })
    }

    /// Parses a line known or suspected to be a section delimiter line in strict mode.
    ///
    /// # Errors
    /// Returns `StateMachineError` if the line cannot be parsed as a valid delimiter.
    pub fn parse(line: &str, line_number: usize) -> Result<Self, StateMachineError> {
        Self::parse_with_mode(line, line_number, ValidationMode::Strict)
    }

    /// Attempts to parse `line` as a section delimiter.
    ///
    /// If the line does not start with `~`, returns `Ok(None)`.
    /// If it starts with `~`, attempts to parse it into `SectionDelimiter`.
    ///
    /// # Errors
    ///
    /// Returns `StateMachineError` if the line starts with `~` but is malformed or invalid.
    pub fn try_parse(line: &str, line_number: usize) -> Result<Option<Self>, StateMachineError> {
        if !Self::is_delimiter_line(line) {
            return Ok(None);
        }
        Self::parse(line, line_number).map(Some)
    }
}

#[pymethods]
impl SectionDelimiter {
    /// Returns the parsed section type.
    #[getter]
    fn get_section_type(&self) -> SectionType {
        self.section_type
    }

    /// Returns the logical section name (e.g. "LOG", "CORE", "VERSION").
    #[getter]
    fn get_section_name(&self) -> &str {
        &self.section_name
    }

    /// Returns the character code of the section.
    #[getter]
    fn get_char_code(&self) -> String {
        self.section_type.char_code().to_string()
    }

    /// Returns the line number (1-based) where this delimiter was found.
    #[getter]
    fn get_line_number(&self) -> usize {
        self.line_number
    }

    /// Returns the raw delimiter line text.
    #[getter]
    fn get_raw_line(&self) -> &str {
        &self.raw_line
    }

    /// Returns any title or extra remarks following the section identifier on the delimiter line.
    #[getter]
    fn get_title(&self) -> &str {
        &self.title
    }

    /// Static method to check if a line starts with `~`.
    #[staticmethod]
    fn is_delimiter(line: &str) -> bool {
        Self::is_delimiter_line(line)
    }

    /// Static method to check if a line is a comment.
    #[staticmethod]
    fn is_comment(line: &str) -> bool {
        Self::is_comment_line(line)
    }

    /// Static method to check if a line is blank.
    #[staticmethod]
    fn is_blank(line: &str) -> bool {
        Self::is_blank_line(line)
    }

    /// Parse a delimiter line with optional validation mode.
    #[staticmethod]
    #[pyo3(signature = (line, line_number = 1, mode = None))]
    fn parse_line(line: &str, line_number: usize, mode: Option<ValidationMode>) -> PyResult<Self> {
        let val_mode = mode.unwrap_or(ValidationMode::Strict);
        Self::parse_with_mode(line, line_number, val_mode).map_err(Into::into)
    }

    fn __repr__(&self) -> String {
        format!(
            "SectionDelimiter(type=SectionType.{}, line={}, title='{}')",
            self.section_type.name(),
            self.line_number,
            self.title
        )
    }

    fn __str__(&self) -> &str {
        &self.raw_line
    }
}

impl fmt::Display for SectionDelimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "~{} (line {}): {}",
            self.section_type.char_code(),
            self.line_number,
            self.title
        )
    }
}

#[cfg(test)]
#[allow(clippy::pedantic)]
mod tests {
    use super::*;

    #[test]
    fn test_all_cwls_sections() {
        assert_eq!(SectionType::ALL_CWLS.len(), 6);
        assert_eq!(SectionType::ALL_CWLS[0], SectionType::Version);
        assert_eq!(SectionType::ALL_CWLS[1], SectionType::Well);
        assert_eq!(SectionType::ALL_CWLS[2], SectionType::Curve);
        assert_eq!(SectionType::ALL_CWLS[3], SectionType::Parameter);
        assert_eq!(SectionType::ALL_CWLS[4], SectionType::Other);
        assert_eq!(SectionType::ALL_CWLS[5], SectionType::Ascii);
    }

    #[test]
    fn test_section_type_properties() {
        // Version
        assert_eq!(SectionType::Version.char_code(), 'V');
        assert_eq!(SectionType::Version.delimiter(), "~V");
        assert_eq!(SectionType::Version.name(), "VERSION");
        assert!(SectionType::Version.is_mandatory());
        assert!(!SectionType::Version.is_optional());
        assert!(SectionType::Version.is_header_section());
        assert!(!SectionType::Version.is_data_section());
        assert!(!SectionType::Version.is_remarks_section());

        // Well
        assert_eq!(SectionType::Well.char_code(), 'W');
        assert_eq!(SectionType::Well.delimiter(), "~W");
        assert!(SectionType::Well.is_mandatory());
        assert!(SectionType::Well.is_header_section());

        // Curve
        assert_eq!(SectionType::Curve.char_code(), 'C');
        assert_eq!(SectionType::Curve.delimiter(), "~C");
        assert!(SectionType::Curve.is_mandatory());
        assert!(SectionType::Curve.is_header_section());

        // Parameter
        assert_eq!(SectionType::Parameter.char_code(), 'P');
        assert_eq!(SectionType::Parameter.delimiter(), "~P");
        assert!(!SectionType::Parameter.is_mandatory());
        assert!(SectionType::Parameter.is_optional());
        assert!(SectionType::Parameter.is_header_section());

        // Other
        assert_eq!(SectionType::Other.char_code(), 'O');
        assert_eq!(SectionType::Other.delimiter(), "~O");
        assert!(!SectionType::Other.is_mandatory());
        assert!(SectionType::Other.is_optional());
        assert!(!SectionType::Other.is_header_section());
        assert!(SectionType::Other.is_remarks_section());

        // Ascii
        assert_eq!(SectionType::Ascii.char_code(), 'A');
        assert_eq!(SectionType::Ascii.delimiter(), "~A");
        assert!(SectionType::Ascii.is_mandatory());
        assert!(!SectionType::Ascii.is_header_section());
        assert!(SectionType::Ascii.is_data_section());
    }

    #[test]
    fn test_from_char() {
        assert_eq!(
            SectionType::from_char('V', 1, "~V").unwrap(),
            SectionType::Version
        );
        assert_eq!(
            SectionType::from_char('v', 1, "~v").unwrap(),
            SectionType::Version
        );
        assert_eq!(
            SectionType::from_char('W', 1, "~W").unwrap(),
            SectionType::Well
        );
        assert_eq!(
            SectionType::from_char('w', 1, "~w").unwrap(),
            SectionType::Well
        );
        assert_eq!(
            SectionType::from_char('C', 1, "~C").unwrap(),
            SectionType::Curve
        );
        assert_eq!(
            SectionType::from_char('c', 1, "~c").unwrap(),
            SectionType::Curve
        );
        assert_eq!(
            SectionType::from_char('P', 1, "~P").unwrap(),
            SectionType::Parameter
        );
        assert_eq!(
            SectionType::from_char('p', 1, "~p").unwrap(),
            SectionType::Parameter
        );
        assert_eq!(
            SectionType::from_char('O', 1, "~O").unwrap(),
            SectionType::Other
        );
        assert_eq!(
            SectionType::from_char('o', 1, "~o").unwrap(),
            SectionType::Other
        );
        assert_eq!(
            SectionType::from_char('A', 1, "~A").unwrap(),
            SectionType::Ascii
        );
        assert_eq!(
            SectionType::from_char('a', 1, "~a").unwrap(),
            SectionType::Ascii
        );

        // Invalid chars
        assert!(matches!(
            SectionType::from_char('Z', 1, "~Z"),
            Err(StateMachineError::UnknownSection { code: 'Z', .. })
        ));
        assert!(matches!(
            SectionType::from_char('1', 1, "~1"),
            Err(StateMachineError::UnknownSection { code: '1', .. })
        ));
    }

    #[test]
    fn test_delimiter_parsing_standard() {
        let v = SectionDelimiter::parse("~VERSION INFORMATION", 1).unwrap();
        assert_eq!(v.section_type(), SectionType::Version);
        assert_eq!(v.title(), "INFORMATION");
        assert_eq!(v.line_number(), 1);

        let w = SectionDelimiter::parse("~Well Information Section", 2).unwrap();
        assert_eq!(w.section_type(), SectionType::Well);
        assert_eq!(w.title(), "Information Section");

        let c = SectionDelimiter::parse("~CURVE INFORMATION", 3).unwrap();
        assert_eq!(c.section_type(), SectionType::Curve);
        assert_eq!(c.title(), "INFORMATION");

        let p = SectionDelimiter::parse("~PARAMETER INFORMATION", 4).unwrap();
        assert_eq!(p.section_type(), SectionType::Parameter);
        assert_eq!(p.title(), "INFORMATION");

        let o = SectionDelimiter::parse("~OTHER", 5).unwrap();
        assert_eq!(o.section_type(), SectionType::Other);
        assert_eq!(o.title(), "");

        let a = SectionDelimiter::parse("~A DEPTH RHOB NPHI", 6).unwrap();
        assert_eq!(a.section_type(), SectionType::Ascii);
        assert_eq!(a.title(), "DEPTH RHOB NPHI");
    }

    #[test]
    fn test_delimiter_parsing_short_and_casing() {
        let v = SectionDelimiter::parse("~v", 1).unwrap();
        assert_eq!(v.section_type(), SectionType::Version);
        assert_eq!(v.title(), "");

        let w = SectionDelimiter::parse("~w", 2).unwrap();
        assert_eq!(w.section_type(), SectionType::Well);

        let c = SectionDelimiter::parse("~c", 3).unwrap();
        assert_eq!(c.section_type(), SectionType::Curve);

        let p = SectionDelimiter::parse("~p", 4).unwrap();
        assert_eq!(p.section_type(), SectionType::Parameter);

        let o = SectionDelimiter::parse("~o", 5).unwrap();
        assert_eq!(o.section_type(), SectionType::Other);

        let a = SectionDelimiter::parse("~a", 6).unwrap();
        assert_eq!(a.section_type(), SectionType::Ascii);
    }

    #[test]
    fn test_delimiter_whitespace_and_crlf() {
        let delim = SectionDelimiter::parse("   ~VERSION INFORMATION   \r\n", 10).unwrap();
        assert_eq!(delim.section_type(), SectionType::Version);
        assert_eq!(delim.title(), "INFORMATION");
        assert_eq!(delim.raw_line(), "   ~VERSION INFORMATION   ");
    }

    #[test]
    fn test_delimiter_invalid() {
        // Not starting with ~
        assert!(matches!(
            SectionDelimiter::parse("VERSION", 1),
            Err(StateMachineError::InvalidDelimiter { .. })
        ));

        // Bare ~
        assert!(matches!(
            SectionDelimiter::parse("~", 1),
            Err(StateMachineError::InvalidDelimiter { .. })
        ));
        assert!(matches!(
            SectionDelimiter::parse("~   ", 1),
            Err(StateMachineError::InvalidDelimiter { .. })
        ));

        // Unknown code
        assert!(matches!(
            SectionDelimiter::parse("~Z UNKNOWN", 1),
            Err(StateMachineError::UnknownSection { code: 'Z', .. })
        ));
    }

    #[test]
    fn test_delimiter_classification_helpers() {
        assert!(SectionDelimiter::is_delimiter_line("~V"));
        assert!(SectionDelimiter::is_delimiter_line("  ~A "));
        assert!(!SectionDelimiter::is_delimiter_line("# comment"));
        assert!(!SectionDelimiter::is_delimiter_line("STRT.M 100.0:"));

        assert!(SectionDelimiter::is_comment_line("# Comment"));
        assert!(SectionDelimiter::is_comment_line("   # Indented"));
        assert!(!SectionDelimiter::is_comment_line("~V"));

        assert!(SectionDelimiter::is_blank_line(""));
        assert!(SectionDelimiter::is_blank_line("   \t  \r\n"));
        assert!(!SectionDelimiter::is_blank_line("~V"));
    }

    #[test]
    fn test_try_parse() {
        assert!(SectionDelimiter::try_parse("not delimiter", 1)
            .unwrap()
            .is_none());
        assert!(SectionDelimiter::try_parse("# comment", 1)
            .unwrap()
            .is_none());
        assert!(SectionDelimiter::try_parse("~V", 1).unwrap().is_some());
    }
}
