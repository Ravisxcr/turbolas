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
