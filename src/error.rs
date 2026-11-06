//! Error types for CWLS LAS state machine and parsing.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;

use crate::section::SectionType;

/// Errors that can occur during LAS section delimiter state machine operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateMachineError {
    /// Non-comment, non-whitespace content was encountered before the first section.
    ContentBeforeFirstSection {
        /// Line number (1-based) where the content was found.
        line_number: usize,
        /// The content that was encountered.
        content: String,
    },
    /// The first section in the LAS file was not `~V` (`Version`).
    ExpectedVersionFirst {
        /// The section type that was encountered instead.
        found: SectionType,
        /// Line number (1-based) where the section was found.
        line_number: usize,
    },
    /// A section delimiter appeared more than once in strict mode.
    DuplicateSection {
        /// The section type that was duplicated.
        section: SectionType,
        /// Line number (1-based) of the duplicate delimiter.
        line_number: usize,
    },
    /// `~A` (`Ascii`) data was encountered before `~C` (`Curve`) definition section.
    AsciiBeforeCurve {
        /// Line number (1-based) where `~A` was encountered.
        line_number: usize,
    },
    /// A section delimiter was encountered after `~A` (`Ascii`) in strict mode.
    SectionAfterAscii {
        /// The section type that was encountered after `~A`.
        section: SectionType,
        /// Line number (1-based) where the delimiter was found.
        line_number: usize,
    },
    /// One or more required sections (`~V`, `~W`, `~C`, `~A`) were missing when parsing finished.
    MissingRequiredSections {
        /// List of required sections that were not found.
        missing: Vec<SectionType>,
    },
    /// An unknown or non-standard section delimiter character was encountered.
    UnknownSection {
        /// The unknown character code.
        code: char,
        /// Line number (1-based) where it was found.
        line_number: usize,
        /// The raw delimiter line.
        raw: String,
    },
    /// A malformed delimiter line was encountered (e.g. empty `~`).
    InvalidDelimiter {
        /// Line number (1-based) where the invalid delimiter occurred.
        line_number: usize,
        /// The raw line text.
        line: String,
        /// Explanation of why the delimiter is invalid.
        reason: String,
    },
    /// Operation attempted after parsing has already completed.
    AlreadyCompleted,
}

impl fmt::Display for StateMachineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContentBeforeFirstSection {
                line_number,
                content,
            } => {
                write!(
                    f,
                    "Line {line_number}: unexpected content before first section: '{content}'"
                )
            }
            Self::ExpectedVersionFirst { found, line_number } => {
                write!(
                    f,
                    "Line {line_number}: CWLS LAS file must begin with ~V (Version) section, but found ~{} ({})",
                    found.char_code(),
                    found.name()
                )
            }
            Self::DuplicateSection {
                section,
                line_number,
            } => {
                write!(
                    f,
                    "Line {line_number}: duplicate section ~{} ({}) is not allowed in strict mode",
                    section.char_code(),
                    section.name()
                )
            }
            Self::AsciiBeforeCurve { line_number } => {
                write!(
                    f,
                    "Line {line_number}: ASCII log data section (~A) cannot appear before Curve section (~C)"
                )
            }
            Self::SectionAfterAscii {
                section,
                line_number,
            } => {
                write!(
                    f,
                    "Line {line_number}: section ~{} ({}) cannot appear after ASCII log data section (~A)",
                    section.char_code(),
                    section.name()
                )
            }
            Self::MissingRequiredSections { missing } => {
                let names: Vec<String> = missing
                    .iter()
                    .map(|s| format!("~{} ({})", s.char_code(), s.name()))
                    .collect();
                write!(
                    f,
                    "Missing required CWLS LAS section(s): {}",
                    names.join(", ")
                )
            }
            Self::UnknownSection {
                code,
                line_number,
                raw,
            } => {
                write!(
                    f,
                    "Line {line_number}: unknown CWLS section identifier '~{code}' in line '{raw}'"
                )
            }
            Self::InvalidDelimiter {
                line_number,
                line,
                reason,
            } => {
                write!(
                    f,
                    "Line {line_number}: invalid delimiter line '{line}': {reason}"
                )
            }
            Self::AlreadyCompleted => {
                write!(f, "State machine has already completed parsing")
            }
        }
    }
}

impl std::error::Error for StateMachineError {}

// Python exception mapping
pyo3::create_exception!(
    turbolas,
    LASSectionError,
    PyValueError,
    "Exception raised for LAS section delimiter and state machine errors."
);

impl From<StateMachineError> for PyErr {
    fn from(err: StateMachineError) -> Self {
        LASSectionError::new_err(err.to_string())
    }
}

/// Errors that can occur during CWLS LAS header line parsing (`MNEM.UNIT VALUE : DESC`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderError {
    /// Line is empty or contains only whitespace.
    EmptyLine {
        /// Line number (1-based) where the empty line occurred.
        line_number: usize,
    },
    /// Line is a comment (starts with `#`), not a header record.
    CommentLine {
        /// Line number (1-based) where the comment occurred.
        line_number: usize,
        /// The raw comment text.
        line: String,
    },
    /// Line is a section delimiter (starts with `~`), not a header record.
    SectionDelimiterLine {
        /// Line number (1-based) where the delimiter occurred.
        line_number: usize,
        /// The raw section line text.
        line: String,
    },
    /// Line is missing the dot separator `.` between mnemonic and unit.
    MissingDot {
        /// Line number (1-based) where the error occurred.
        line_number: usize,
        /// The raw line text.
        line: String,
    },
    /// Mnemonic is empty (e.g. line starts with `.`).
    EmptyMnemonic {
        /// Line number (1-based) where the error occurred.
        line_number: usize,
        /// The raw line text.
        line: String,
    },
    /// Line is missing the colon separator `:` between value and description in strict mode.
    MissingColon {
        /// Line number (1-based) where the error occurred.
        line_number: usize,
        /// The raw line text.
        line: String,
    },
    /// Invalid mnemonic (e.g. contains forbidden colon `:` or whitespace in strict mode).
    InvalidMnemonic {
        /// Line number (1-based) where the error occurred.
        line_number: usize,
        /// The extracted mnemonic string.
        mnemonic: String,
        /// Explanation of why the mnemonic is invalid.
        reason: String,
    },
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLine { line_number } => {
                write!(
                    f,
                    "Line {line_number}: header line is empty or whitespace only"
                )
            }
            Self::CommentLine { line_number, line } => {
                write!(
                    f,
                    "Line {line_number}: expected header line, found comment: '{line}'"
                )
            }
            Self::SectionDelimiterLine { line_number, line } => {
                write!(
                    f,
                    "Line {line_number}: expected header line, found section delimiter: '{line}'"
                )
            }
            Self::MissingDot { line_number, line } => {
                write!(
                    f,
                    "Line {line_number}: header line missing required dot '.' delimiter: '{line}'"
                )
            }
            Self::EmptyMnemonic { line_number, line } => {
                write!(
                    f,
                    "Line {line_number}: header line has empty mnemonic before dot '.': '{line}'"
                )
            }
            Self::MissingColon { line_number, line } => {
                write!(f, "Line {line_number}: header line missing required colon ':' delimiter in strict mode: '{line}'")
            }
            Self::InvalidMnemonic {
                line_number,
                mnemonic,
                reason,
            } => {
                write!(
                    f,
                    "Line {line_number}: invalid mnemonic '{mnemonic}': {reason}"
                )
            }
        }
    }
}

impl std::error::Error for HeaderError {}

pyo3::create_exception!(
    turbolas,
    LASHeaderError,
    PyValueError,
    "Exception raised for LAS header line parsing errors."
);

impl From<HeaderError> for PyErr {
    fn from(err: HeaderError) -> Self {
        LASHeaderError::new_err(err.to_string())
    }
}

/// Errors that can occur during CWLS LAS ASCII log data parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataError {
    /// Zero columns defined or requested.
    NoColumnsDefined,
    /// Invalid float token encountered.
    InvalidFloat {
        /// Line number (1-based) where the invalid float occurred.
        line_number: usize,
        /// Column index (0-based) where the invalid float occurred.
        column_index: usize,
        /// Column name/mnemonic if available.
        column_name: Option<String>,
        /// The invalid token text.
        token: String,
        /// Detailed explanation of why the float failed to parse.
        reason: String,
    },
    /// Row has fewer or more columns than expected in strict unwrapped mode.
    ColumnCountMismatch {
        /// Line number (1-based) where the mismatch occurred.
        line_number: usize,
        /// Expected column count.
        expected: usize,
        /// Actual columns found on the line.
        found: usize,
    },
    /// Stream ended with an incomplete row (fewer columns than expected).
    IncompleteRow {
        /// Expected column count per row.
        expected: usize,
        /// Number of columns parsed before EOF.
        found: usize,
    },
    /// Data section is empty or contains no numeric values in strict mode.
    EmptyData,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoColumnsDefined => {
                write!(f, "Cannot parse ASCII log data with 0 columns defined")
            }
            Self::InvalidFloat {
                line_number,
                column_index,
                column_name,
                token,
                reason,
            } => {
                let col_desc = match column_name {
                    Some(name) => format!("column {column_index} ('{name}')"),
                    None => format!("column {column_index}"),
                };
                write!(
                    f,
                    "Line {line_number}, {col_desc}: failed to parse float token '{token}': {reason}"
                )
            }
            Self::ColumnCountMismatch {
                line_number,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Line {line_number}: column count mismatch (expected {expected} columns, found {found})"
                )
            }
            Self::IncompleteRow { expected, found } => {
                write!(
                    f,
                    "End of data reached with incomplete row (expected {expected} columns, found {found})"
                )
            }
            Self::EmptyData => {
                write!(f, "ASCII log data section contains no numeric values")
            }
        }
    }
}

impl std::error::Error for DataError {}

pyo3::create_exception!(
    turbolas,
    LASDataError,
    PyValueError,
    "Exception raised for LAS ASCII log data parsing errors."
);

impl From<DataError> for PyErr {
    fn from(err: DataError) -> Self {
        LASDataError::new_err(err.to_string())
    }
}
