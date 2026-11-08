//! Finite state machine for tracking CWLS LAS section transitions.
//!
//! Enforces CWLS LAS standard rules (Version first, ASCII last, Curve before ASCII)
//! in strict mode, while offering a permissive mode for legacy or non-compliant files.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;

use crate::error::StateMachineError;
use crate::header::HeaderItem;
use crate::null::NullSentinel;
use crate::section::{SectionDelimiter, SectionType};

/// FSM state representing the current section context within a LAS file.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum State {
    /// Initial state before any section delimiter has been encountered.
    #[default]
    Initial,
    /// Currently inside the `~V` (Version) section.
    Version,
    /// Currently inside the `~W` (Well) section.
    Well,
    /// Currently inside the `~C` (Curve) section.
    Curve,
    /// Currently inside the `~P` (Parameter) section.
    Parameter,
    /// Currently inside the `~O` (Other remarks) section.
    Other,
    /// Currently inside the `~A` (ASCII log data) section.
    Ascii,
    /// Parsing has finished and mandatory section validation passed.
    Complete,
}

impl State {
    /// Returns the active `SectionType`, or `None` if `Initial` or `Complete`.
    #[must_use]
    pub const fn section_type(&self) -> Option<SectionType> {
        match self {
            Self::Initial | Self::Complete => None,
            Self::Version => Some(SectionType::Version),
            Self::Well => Some(SectionType::Well),
            Self::Curve => Some(SectionType::Curve),
            Self::Parameter => Some(SectionType::Parameter),
            Self::Other => Some(SectionType::Other),
            Self::Ascii => Some(SectionType::Ascii),
        }
    }

    /// Converts a `SectionType` into its corresponding active `State`.
    #[must_use]
    pub const fn from_section_type(section: SectionType) -> Self {
        match section {
            SectionType::Version => Self::Version,
            SectionType::Well => Self::Well,
            SectionType::Curve => Self::Curve,
            SectionType::Parameter => Self::Parameter,
            SectionType::Other => Self::Other,
            SectionType::Ascii => Self::Ascii,
        }
    }

    /// Returns `true` if this is the initial state prior to any section.
    #[must_use]
    pub const fn is_initial(&self) -> bool {
        matches!(self, Self::Initial)
    }

    /// Returns `true` if the state machine is inside an active section.
    #[must_use]
    pub const fn is_in_section(&self) -> bool {
        self.section_type().is_some()
    }

    /// Returns `true` if the state machine is inside the ASCII data section (`~A`).
    #[must_use]
    pub const fn is_in_data(&self) -> bool {
        matches!(self, Self::Ascii)
    }

    /// Returns `true` if the state machine has reached the terminal `Complete` state.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }

    /// Returns the uppercase name of the state.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Initial => "INITIAL",
            Self::Version => "VERSION",
            Self::Well => "WELL",
            Self::Curve => "CURVE",
            Self::Parameter => "PARAMETER",
            Self::Other => "OTHER",
            Self::Ascii => "ASCII",
            Self::Complete => "COMPLETE",
        }
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
#[pymethods]
impl State {
    /// Returns the active `SectionType`, or `None` if in `INITIAL` or `COMPLETE`.
    #[getter]
    fn get_section_type(&self) -> Option<SectionType> {
        self.section_type()
    }

    /// Returns the state name.
    #[getter]
    fn get_name(&self) -> &'static str {
        self.name()
    }

    /// Returns whether this is the initial state.
    #[getter]
    fn get_is_initial(&self) -> bool {
        self.is_initial()
    }

    /// Returns whether this is within an active section.
    #[getter]
    fn get_is_in_section(&self) -> bool {
        self.is_in_section()
    }

    /// Returns whether this is in the ASCII log data section.
    #[getter]
    fn get_is_in_data(&self) -> bool {
        self.is_in_data()
    }

    /// Returns whether this is the completed terminal state.
    #[getter]
    fn get_is_complete(&self) -> bool {
        self.is_complete()
    }

    fn __repr__(&self) -> String {
        format!("State.{}", self.name())
    }

    fn __str__(&self) -> &'static str {
        self.name()
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Validation mode for the CWLS LAS state machine.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationMode {
    /// Enforces CWLS LAS 2.0 specifications strictly:
    /// - `~V` must be the first section.
    /// - `~W` must appear before `~A`.
    /// - `~C` must appear before `~A`.
    /// - `~A` must be the final section.
    /// - Sections cannot be duplicated.
    /// - No content lines before the first section.
    /// - All mandatory sections (`~V`, `~W`, `~C`, `~A`) required on finish.
    #[default]
    Strict,

    /// Permissive mode for real-world or non-compliant LAS files:
    /// - Sections may appear in any order.
    /// - Allows sections after `~A`.
    /// - Duplicate sections are permitted.
    /// - Content before the first section is skipped.
    /// - Missing mandatory sections do not trigger hard errors on finish.
    Permissive,
}

impl ValidationMode {
    /// Returns `true` if this is strict validation mode.
    #[must_use]
    pub const fn is_strict(&self) -> bool {
        matches!(self, Self::Strict)
    }

    /// Returns `true` if this is permissive validation mode.
    #[must_use]
    pub const fn is_permissive(&self) -> bool {
        matches!(self, Self::Permissive)
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
#[pymethods]
impl ValidationMode {
    /// Returns `true` if this is strict mode.
    #[getter]
    fn get_is_strict(&self) -> bool {
        self.is_strict()
    }

    /// Returns `true` if this is permissive mode.
    #[getter]
    fn get_is_permissive(&self) -> bool {
        self.is_permissive()
    }

    fn __repr__(&self) -> String {
        match self {
            Self::Strict => "ValidationMode.Strict".to_string(),
            Self::Permissive => "ValidationMode.Permissive".to_string(),
        }
    }
}

/// Classification of a processed line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineAction {
    /// Line triggered a new section delimiter transition.
    SectionStart {
        /// The new section type.
        section: SectionType,
        /// The parsed delimiter metadata.
        delimiter: SectionDelimiter,
    },
    /// A comment line (starting with `#`).
    Comment(String),
    /// An empty or whitespace-only line.
    Blank,
    /// A content / record line belonging to the active section.
    Content {
        /// The active section type.
        section: SectionType,
        /// Content text.
        text: String,
    },
}

/// Python wrapper for `LineAction`.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyLineAction {
    /// Type of action: `"section_start"`, `"comment"`, `"blank"`, or `"content"`.
    action_type: String,
    /// The section type if applicable.
    section: Option<SectionType>,
    /// Delimiter metadata if `section_start`.
    delimiter: Option<SectionDelimiter>,
    /// Text content for comment or content line.
    text: Option<String>,
}

#[pymethods]
impl PyLineAction {
    /// The action type string.
    #[getter]
    fn get_action_type(&self) -> &str {
        &self.action_type
    }

    /// Associated section type if any.
    #[getter]
    fn get_section(&self) -> Option<SectionType> {
        self.section
    }

    /// Associated delimiter if this line started a section.
    #[getter]
    fn get_delimiter(&self) -> Option<SectionDelimiter> {
        self.delimiter.clone()
    }

    /// Line text if this is a comment or content line.
    #[getter]
    fn get_text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// Whether this line triggered a section start.
    #[getter]
    fn get_is_section_start(&self) -> bool {
        self.action_type == "section_start"
    }

    /// Whether this line is a comment.
    #[getter]
    fn get_is_comment(&self) -> bool {
        self.action_type == "comment"
    }

    /// Whether this line is blank.
    #[getter]
    fn get_is_blank(&self) -> bool {
        self.action_type == "blank"
    }

    /// Whether this line is content in an active section.
    #[getter]
    fn get_is_content(&self) -> bool {
        self.action_type == "content"
    }

    fn __repr__(&self) -> String {
        format!(
            "LineAction(type='{}', section={:?})",
            self.action_type,
            self.section.map(|s| s.name())
        )
    }
}

impl From<LineAction> for PyLineAction {
    fn from(action: LineAction) -> Self {
        match action {
            LineAction::SectionStart { section, delimiter } => Self {
                action_type: "section_start".to_string(),
                section: Some(section),
                delimiter: Some(delimiter),
                text: None,
            },
            LineAction::Comment(text) => Self {
                action_type: "comment".to_string(),
                section: None,
                delimiter: None,
                text: Some(text),
            },
            LineAction::Blank => Self {
                action_type: "blank".to_string(),
                section: None,
                delimiter: None,
                text: None,
            },
            LineAction::Content { section, text } => Self {
                action_type: "content".to_string(),
                section: Some(section),
                delimiter: None,
                text: Some(text),
            },
        }
    }
}

/// CWLS LAS section delimiter state machine.
///
/// Tracks section transitions (`~V`, `~W`, `~C`, `~P`, `~O`, `~A`), validates
/// ordering and uniqueness rules, and tallies line counts per section.
#[pyclass]
#[derive(Debug, Clone)]
pub struct SectionStateMachine {
    /// Active FSM state.
    state: State,
    /// Operating validation mode.
    mode: ValidationMode,
    /// Current 1-based line number in stream.
    line_number: usize,
    /// Ordered list of sections seen so far.
    seen_sections: Vec<SectionType>,
    /// Line count for each section.
    section_line_counts: HashMap<SectionType, usize>,
    /// Total lines processed.
    total_lines_processed: usize,
    /// Line count inside the current active section.
    current_section_line_count: usize,
    /// Delimiter metadata for current active section.
    current_delimiter: Option<SectionDelimiter>,
    /// Dynamically detected or configured NULL sentinel.
    null_sentinel: NullSentinel,
    /// Whether the NULL sentinel was dynamically detected from header records.
    is_null_detected: bool,
}

impl Default for SectionStateMachine {
    fn default() -> Self {
        Self::new(ValidationMode::Strict)
    }
}

impl SectionStateMachine {
    /// Creates a new state machine with the specified validation mode.
    #[must_use]
    pub fn new(mode: ValidationMode) -> Self {
        Self {
            state: State::Initial,
            mode,
            line_number: 0,
            seen_sections: Vec::with_capacity(6),
            section_line_counts: HashMap::with_capacity(6),
            total_lines_processed: 0,
            current_section_line_count: 0,
            current_delimiter: None,
            null_sentinel: NullSentinel::default_cwls(),
            is_null_detected: false,
        }
    }

    /// Creates a strict CWLS-compliant state machine.
    #[must_use]
    pub fn strict() -> Self {
        Self::new(ValidationMode::Strict)
    }

    /// Creates a permissive state machine.
    #[must_use]
    pub fn permissive() -> Self {
        Self::new(ValidationMode::Permissive)
    }

    /// Returns the active state.
    #[must_use]
    pub const fn current_state(&self) -> State {
        self.state
    }

    /// Returns the active `SectionType`, if inside a section.
    #[must_use]
    pub const fn current_section(&self) -> Option<SectionType> {
        self.state.section_type()
    }

    /// Returns the validation mode.
    #[must_use]
    pub const fn mode(&self) -> ValidationMode {
        self.mode
    }

    /// Returns the current line number (1-based, or 0 if nothing processed).
    #[must_use]
    pub const fn line_number(&self) -> usize {
        self.line_number
    }

    /// Returns total lines processed so far.
    #[must_use]
    pub const fn total_lines_processed(&self) -> usize {
        self.total_lines_processed
    }

    /// Returns the number of distinct section headers encountered.
    #[must_use]
    pub fn section_count(&self) -> usize {
        self.seen_sections.len()
    }

    /// Returns an ordered slice of sections encountered.
    #[must_use]
    pub fn seen_sections(&self) -> &[SectionType] {
        &self.seen_sections
    }

    /// Returns whether the given section has already been encountered.
    #[must_use]
    pub fn has_seen(&self, section: SectionType) -> bool {
        self.seen_sections.contains(&section)
    }

    /// Returns `true` if the state machine is currently reading `~A` data.
    #[must_use]
    pub const fn is_in_data_section(&self) -> bool {
        self.state.is_in_data()
    }

    /// Returns the line count processed in the current active section.
    #[must_use]
    pub const fn current_section_line_count(&self) -> usize {
        self.current_section_line_count
    }

    /// Returns the line count recorded for a specific section.
    #[must_use]
    pub fn line_count_for_section(&self, section: SectionType) -> usize {
        self.section_line_counts.get(&section).copied().unwrap_or(0)
    }

    /// Returns a reference to the active `NullSentinel`.
    #[must_use]
    pub fn null_sentinel(&self) -> &NullSentinel {
        &self.null_sentinel
    }

    /// Returns the current numerical NULL sentinel value.
    #[must_use]
    pub fn null_value(&self) -> f64 {
        self.null_sentinel.value()
    }

    /// Returns whether the NULL sentinel was dynamically detected.
    #[must_use]
    pub const fn is_null_detected(&self) -> bool {
        self.is_null_detected
    }

    /// Returns the current section's delimiter metadata.
    #[must_use]
    pub fn current_delimiter(&self) -> Option<&SectionDelimiter> {
        self.current_delimiter.as_ref()
    }

    /// Transitions the state machine into the section specified by `delimiter`.
    ///
    /// Validates transition rules based on `mode`.
    ///
    /// # Errors
    ///
    /// Returns `StateMachineError` if the transition violates CWLS rules in strict mode:
    /// - Section other than `~V` first
    /// - Duplicate section
    /// - `~A` before `~C` or `~W`
    /// - Section delimiter after `~A`
    pub fn transition(&mut self, delimiter: SectionDelimiter) -> Result<State, StateMachineError> {
        let target = delimiter.section_type();
        let target_line = delimiter.line_number();

        if self.mode == ValidationMode::Strict {
            if self.state.is_complete() {
                return Err(StateMachineError::AlreadyCompleted);
            }

            // Rule 1: ~V must be the first section
            if self.state.is_initial() && target != SectionType::Version {
                return Err(StateMachineError::ExpectedVersionFirst {
                    found: target,
                    line_number: target_line,
                });
            }

            // Rule 2: No section delimiter may follow ~A (Ascii)
            if self.state == State::Ascii {
                return Err(StateMachineError::SectionAfterAscii {
                    section: target,
                    line_number: target_line,
                });
            }

            // Rule 3: Sections cannot be duplicated
            if self.has_seen(target) {
                return Err(StateMachineError::DuplicateSection {
                    section: target,
                    line_number: target_line,
                });
            }

            // Rule 4: ~A requires ~C (Curve) to have been defined
            if target == SectionType::Ascii && !self.has_seen(SectionType::Curve) {
                return Err(StateMachineError::AsciiBeforeCurve {
                    line_number: target_line,
                });
            }

            // Rule 5: ~A requires ~W (Well) to have been seen
            if target == SectionType::Ascii && !self.has_seen(SectionType::Well) {
                return Err(StateMachineError::MissingRequiredSections {
                    missing: vec![SectionType::Well],
                });
            }
        }

        // Apply transition
        let next_state = State::from_section_type(target);
        self.state = next_state;
        self.seen_sections.push(target);
        self.current_section_line_count = 0;
        self.current_delimiter = Some(delimiter);

        Ok(next_state)
    }

    /// Convenience method to transition to a `SectionType` without a delimiter instance.
    ///
    /// # Errors
    ///
    /// Returns `StateMachineError` if the transition violates state rules in strict mode.
    pub fn transition_to(&mut self, section: SectionType) -> Result<State, StateMachineError> {
        let line_num = if self.line_number == 0 {
            1
        } else {
            self.line_number
        };
        let delim = SectionDelimiter::new(
            section,
            line_num,
            format!("~{}", section.char_code()),
            String::new(),
        );
        self.transition(delim)
    }

    /// Ingests and processes a single line of text from a LAS stream.
    ///
    /// Classifies the line as blank, comment, section delimiter, or section content.
    /// Updates line numbers, section counters, and state transitions.
    ///
    /// # Errors
    ///
    /// Returns `StateMachineError` on invalid delimiter, illegal state transition,
    /// or content prior to the first section in strict mode.
    pub fn process_line(&mut self, line: &str) -> Result<LineAction, StateMachineError> {
        self.line_number += 1;
        self.total_lines_processed += 1;

        // Check if blank line
        if SectionDelimiter::is_blank_line(line) {
            return Ok(LineAction::Blank);
        }

        // Check if comment line
        if SectionDelimiter::is_comment_line(line) {
            let trimmed = line.trim_start();
            let comment_text = trimmed[1..].trim();
            return Ok(LineAction::Comment(comment_text.to_string()));
        }

        // Check if section delimiter line
        if SectionDelimiter::is_delimiter_line(line) {
            let delimiter = SectionDelimiter::parse(line, self.line_number)?;
            let section = delimiter.section_type();
            self.transition(delimiter.clone())?;
            return Ok(LineAction::SectionStart { section, delimiter });
        }

        // Non-delimiter, non-comment, non-blank line: Content line
        if self.state.is_initial() {
            if self.mode == ValidationMode::Strict {
                return Err(StateMachineError::ContentBeforeFirstSection {
                    line_number: self.line_number,
                    content: line.trim().to_string(),
                });
            }
            // In permissive mode, ignore content before first section
            return Ok(LineAction::Blank);
        }

        if let Some(current_sec) = self.current_section() {
            self.current_section_line_count += 1;
            let counter = self.section_line_counts.entry(current_sec).or_insert(0);
            *counter += 1;

            let clean_text = line.trim_end_matches(['\r', '\n']).to_string();

            // Dynamic NULL sentinel detection:
            if current_sec.is_header_section() {
                let trimmed = clean_text.trim_start();
                if trimmed.len() >= 4 && trimmed[..4].eq_ignore_ascii_case("NULL") {
                    if let Ok(item) =
                        HeaderItem::parse(&clean_text, self.line_number, ValidationMode::Permissive)
                    {
                        if let Some(sentinel) = NullSentinel::from_header_item(&item) {
                            self.null_sentinel = sentinel;
                            self.is_null_detected = true;
                        }
                    }
                }
            }

            Ok(LineAction::Content {
                section: current_sec,
                text: clean_text,
            })
        } else {
            Ok(LineAction::Blank)
        }
    }

    /// Consumes a line and returns the new `SectionType` if this line opened a section.
    ///
    /// # Errors
    ///
    /// Returns `StateMachineError` if line processing fails.
    pub fn feed_line(&mut self, line: &str) -> Result<Option<SectionType>, StateMachineError> {
        match self.process_line(line)? {
            LineAction::SectionStart { section, .. } => Ok(Some(section)),
            _ => Ok(None),
        }
    }

    /// Signals the end of the LAS stream and validates required sections.
    ///
    /// In strict mode, verifies that mandatory sections (`~V`, `~W`, `~C`, `~A`) were present.
    /// Transitions state machine to `State::Complete`.
    ///
    /// # Errors
    ///
    /// Returns `StateMachineError::MissingRequiredSections` if any mandatory section was absent in strict mode.
    pub fn finish(&mut self) -> Result<(), StateMachineError> {
        if self.state.is_complete() {
            return Ok(());
        }

        if self.mode == ValidationMode::Strict {
            let mandatory = [
                SectionType::Version,
                SectionType::Well,
                SectionType::Curve,
                SectionType::Ascii,
            ];
            let mut missing = Vec::new();
            for sec in mandatory {
                if !self.has_seen(sec) {
                    missing.push(sec);
                }
            }
            if !missing.is_empty() {
                return Err(StateMachineError::MissingRequiredSections { missing });
            }
        }

        self.state = State::Complete;
        Ok(())
    }

    /// Resets the state machine to its initial state, clearing all history.
    pub fn reset(&mut self) {
        self.state = State::Initial;
        self.line_number = 0;
        self.total_lines_processed = 0;
        self.current_section_line_count = 0;
        self.seen_sections.clear();
        self.section_line_counts.clear();
        self.current_delimiter = None;
        self.null_sentinel = NullSentinel::default_cwls();
        self.is_null_detected = false;
    }
}

#[pymethods]
impl SectionStateMachine {
    /// Creates a new `SectionStateMachine`.
    ///
    /// Parameters:
    /// - `strict`: `bool` = `True` (enforce CWLS LAS standards)
    /// - `mode`: `Optional[ValidationMode]` = `None` (explicit `ValidationMode` enum)
    #[new]
    #[pyo3(signature = (strict = true, mode = None))]
    fn py_new(strict: bool, mode: Option<ValidationMode>) -> Self {
        let val_mode = mode.unwrap_or(if strict {
            ValidationMode::Strict
        } else {
            ValidationMode::Permissive
        });
        Self::new(val_mode)
    }

    /// Returns the active state.
    #[getter]
    fn get_current_state(&self) -> State {
        self.state
    }

    /// Returns the active state name as a string.
    #[getter]
    fn get_current_state_name(&self) -> &'static str {
        self.state.name()
    }

    /// Returns the active `SectionType`, or `None`.
    #[getter]
    fn get_current_section(&self) -> Option<SectionType> {
        self.current_section()
    }

    /// Returns the validation mode.
    #[getter]
    fn get_mode(&self) -> ValidationMode {
        self.mode
    }

    /// Current line number (1-based).
    #[getter]
    fn get_line_number(&self) -> usize {
        self.line_number
    }

    /// Total lines processed.
    #[getter]
    fn get_total_lines_processed(&self) -> usize {
        self.total_lines_processed
    }

    /// Number of sections encountered so far.
    #[getter]
    fn get_section_count(&self) -> usize {
        self.section_count()
    }

    /// List of sections encountered in order.
    #[getter]
    fn get_seen_sections(&self) -> Vec<SectionType> {
        self.seen_sections.clone()
    }

    /// Delimiter metadata for current active section.
    #[getter]
    fn get_current_delimiter(&self) -> Option<SectionDelimiter> {
        self.current_delimiter.clone()
    }

    /// Lines processed in the current active section.
    #[getter]
    fn get_current_section_line_count(&self) -> usize {
        self.current_section_line_count
    }

    /// Returns `true` if currently in ASCII data section (`~A`).
    #[getter]
    fn get_is_in_data_section(&self) -> bool {
        self.is_in_data_section()
    }

    /// Returns `true` if state machine reached completed state.
    #[getter]
    fn get_is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// Checks if a given section has been encountered.
    fn has_seen_section(&self, section: SectionType) -> bool {
        self.has_seen(section)
    }

    /// Returns line count for a given section.
    fn get_line_count_for_section(&self, section: SectionType) -> usize {
        self.line_count_for_section(section)
    }

    /// Processes a single line, returning a `PyLineAction`.
    #[pyo3(name = "process_line")]
    fn process_line_py(&mut self, line: &str) -> PyResult<PyLineAction> {
        self.process_line(line).map(Into::into).map_err(Into::into)
    }

    /// Processes a line, returning the `SectionType` if a section started.
    #[pyo3(name = "feed_line")]
    fn feed_line_py(&mut self, line: &str) -> PyResult<Option<SectionType>> {
        self.feed_line(line).map_err(Into::into)
    }

    /// Transitions to a section via `SectionDelimiter`.
    #[pyo3(name = "transition")]
    fn transition_py(&mut self, delimiter: SectionDelimiter) -> PyResult<State> {
        self.transition(delimiter).map_err(Into::into)
    }

    /// Transitions to a section by `SectionType`.
    #[pyo3(name = "transition_to")]
    fn transition_to_py(&mut self, section: SectionType) -> PyResult<State> {
        self.transition_to(section).map_err(Into::into)
    }

    /// Completes the stream and validates required sections.
    #[pyo3(name = "finish")]
    fn finish_py(&mut self) -> PyResult<()> {
        self.finish().map_err(Into::into)
    }

    /// Resets the state machine.
    #[pyo3(name = "reset")]
    fn reset_py(&mut self) {
        self.reset();
    }

    /// Parses a line as a CWLS header item using the state machine's current validation mode.
    ///
    /// # Errors
    /// Returns `LASSectionError` if called outside of a header section (~V, ~W, ~C, ~P),
    /// or `LASHeaderError` if the line itself cannot be parsed.
    pub fn parse_header_line(&mut self, line: &str) -> PyResult<HeaderItem> {
        let current_sec = self.current_section().ok_or_else(|| {
            crate::error::LASSectionError::new_err(
                "Cannot parse header line: not currently in any section",
            )
        })?;
        if !current_sec.is_header_section() {
            return Err(crate::error::LASSectionError::new_err(format!(
                "Cannot parse header line in section ~{} ({}); only header sections (~V, ~W, ~C, ~P) contain header records",
                current_sec.char_code(),
                current_sec.name()
            )));
        }
        let item = HeaderItem::parse(line, self.line_number, self.mode)?;
        if let Some(sentinel) = NullSentinel::from_header_item(&item) {
            self.null_sentinel = sentinel;
            self.is_null_detected = true;
        }
        Ok(item)
    }

    /// Returns the detected or configured NULL sentinel.
    #[must_use]
    #[getter(null_sentinel)]
    pub fn get_null_sentinel(&self) -> NullSentinel {
        self.null_sentinel.clone()
    }

    /// Sets the NULL sentinel.
    #[setter(null_sentinel)]
    pub fn set_null_sentinel(&mut self, sentinel: NullSentinel) {
        self.null_sentinel = sentinel;
    }

    /// Returns the numerical NULL sentinel value.
    #[must_use]
    #[getter(null_value)]
    pub fn get_null_value(&self) -> f64 {
        self.null_sentinel.value()
    }

    /// Returns whether the NULL sentinel was dynamically detected from header records.
    #[must_use]
    #[getter(is_null_detected)]
    pub fn get_is_null_detected(&self) -> bool {
        self.is_null_detected
    }

    /// Parses a line of ASCII log numerical data, dynamically mapping NULL sentinels to IEEE 754 NaN.
    ///
    /// # Errors
    /// Returns `LASSectionError` in strict mode if called outside the ASCII data section (`~A`),
    /// or `PyValueError` if any token cannot be parsed as a float.
    pub fn parse_data_line(&self, line: &str) -> PyResult<Vec<f64>> {
        if self.mode.is_strict() && !self.is_in_data_section() {
            return Err(crate::error::LASSectionError::new_err(
                "Cannot parse data line: not currently in ASCII data section (~A)",
            ));
        }
        self.null_sentinel
            .parse_ascii_line(line)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "SectionStateMachine(state={}, line={}, seen={})",
            self.state.name(),
            self.line_number,
            self.seen_sections.len()
        )
    }
}

#[cfg(test)]
#[allow(clippy::pedantic)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let sm = SectionStateMachine::strict();
        assert_eq!(sm.current_state(), State::Initial);
        assert!(sm.current_state().is_initial());
        assert_eq!(sm.current_section(), None);
        assert!(!sm.is_in_data_section());
        assert_eq!(sm.line_number(), 0);
        assert_eq!(sm.total_lines_processed(), 0);
        assert_eq!(sm.section_count(), 0);
        assert!(sm.seen_sections().is_empty());
    }

    #[test]
    fn test_standard_cwls_sequence_strict() {
        let mut sm = SectionStateMachine::strict();

        // 1. Comments before ~V are allowed
        let res = sm.process_line("# CWLS LAS file").unwrap();
        assert_eq!(res, LineAction::Comment("CWLS LAS file".to_string()));
        assert_eq!(sm.current_state(), State::Initial);

        // Blanks before ~V are allowed
        let res = sm.process_line("").unwrap();
        assert_eq!(res, LineAction::Blank);

        // 2. ~V (Version)
        let res = sm.process_line("~VERSION INFORMATION").unwrap();
        assert!(matches!(
            res,
            LineAction::SectionStart {
                section: SectionType::Version,
                ..
            }
        ));
        assert_eq!(sm.current_state(), State::Version);
        assert_eq!(sm.current_section(), Some(SectionType::Version));

        // Content in ~V
        let res = sm
            .process_line(" VERS. 2.0 : CWLS LOG ASCII STANDARD")
            .unwrap();
        assert!(matches!(
            res,
            LineAction::Content {
                section: SectionType::Version,
                ..
            }
        ));
        let res = sm
            .process_line(" WRAP. NO : ONE LINE PER RECORD STEP")
            .unwrap();
        assert!(matches!(
            res,
            LineAction::Content {
                section: SectionType::Version,
                ..
            }
        ));
        assert_eq!(sm.current_section_line_count(), 2);
        assert_eq!(sm.line_count_for_section(SectionType::Version), 2);

        // 3. ~W (Well)
        let res = sm.process_line("~WELL INFORMATION").unwrap();
        assert!(matches!(
            res,
            LineAction::SectionStart {
                section: SectionType::Well,
                ..
            }
        ));
        assert_eq!(sm.current_state(), State::Well);
        sm.process_line(" STRT.M 100.0 : START DEPTH").unwrap();
        sm.process_line(" STOP.M 200.0 : STOP DEPTH").unwrap();
        assert_eq!(sm.line_count_for_section(SectionType::Well), 2);

        // 4. ~C (Curve)
        let res = sm.process_line("~CURVE INFORMATION").unwrap();
        assert!(matches!(
            res,
            LineAction::SectionStart {
                section: SectionType::Curve,
                ..
            }
        ));
        assert_eq!(sm.current_state(), State::Curve);
        sm.process_line(" DEPT.M : 1 DEPTH").unwrap();
        sm.process_line(" GR.GAPI : 2 GAMMA RAY").unwrap();
        assert_eq!(sm.line_count_for_section(SectionType::Curve), 2);

        // 5. ~P (Parameter) - Optional
        let res = sm.process_line("~PARAMETER INFORMATION").unwrap();
        assert!(matches!(
            res,
            LineAction::SectionStart {
                section: SectionType::Parameter,
                ..
            }
        ));
        assert_eq!(sm.current_state(), State::Parameter);
        sm.process_line(" BHT.DEGC 35.0 : BOTTOM HOLE TEMP")
            .unwrap();
        assert_eq!(sm.line_count_for_section(SectionType::Parameter), 1);

        // 6. ~O (Other) - Optional
        let res = sm.process_line("~OTHER").unwrap();
        assert!(matches!(
            res,
            LineAction::SectionStart {
                section: SectionType::Other,
                ..
            }
        ));
        assert_eq!(sm.current_state(), State::Other);
        sm.process_line("Some logging remarks here").unwrap();
        assert_eq!(sm.line_count_for_section(SectionType::Other), 1);

        // 7. ~A (Ascii Data)
        let res = sm.process_line("~A DEPTH GR").unwrap();
        assert!(matches!(
            res,
            LineAction::SectionStart {
                section: SectionType::Ascii,
                ..
            }
        ));
        assert_eq!(sm.current_state(), State::Ascii);
        assert!(sm.is_in_data_section());
        sm.process_line("100.0 45.2").unwrap();
        sm.process_line("100.5 46.1").unwrap();
        sm.process_line("101.0 47.3").unwrap();
        assert_eq!(sm.line_count_for_section(SectionType::Ascii), 3);

        // Finish
        sm.finish().unwrap();
        assert_eq!(sm.current_state(), State::Complete);
        assert!(sm.current_state().is_complete());
        assert_eq!(sm.section_count(), 6);
    }

    #[test]
    fn test_strict_rejects_non_version_first() {
        let mut sm = SectionStateMachine::strict();
        let err = sm.process_line("~WELL INFORMATION").unwrap_err();
        assert!(matches!(
            err,
            StateMachineError::ExpectedVersionFirst {
                found: SectionType::Well,
                line_number: 1
            }
        ));

        let mut sm2 = SectionStateMachine::strict();
        let err2 = sm2.process_line("~CURVE").unwrap_err();
        assert!(matches!(
            err2,
            StateMachineError::ExpectedVersionFirst {
                found: SectionType::Curve,
                line_number: 1
            }
        ));

        let mut sm3 = SectionStateMachine::strict();
        let err3 = sm3.process_line("~A").unwrap_err();
        assert!(matches!(
            err3,
            StateMachineError::ExpectedVersionFirst {
                found: SectionType::Ascii,
                line_number: 1
            }
        ));
    }

    #[test]
    fn test_strict_rejects_content_before_first_section() {
        let mut sm = SectionStateMachine::strict();
        let err = sm.process_line("STRT.M 100.0:").unwrap_err();
        assert!(matches!(
            err,
            StateMachineError::ContentBeforeFirstSection { line_number: 1, .. }
        ));
    }

    #[test]
    fn test_strict_rejects_duplicate_section() {
        let mut sm = SectionStateMachine::strict();
        sm.process_line("~VERSION").unwrap();
        sm.process_line("~WELL").unwrap();
        let err = sm.process_line("~VERSION").unwrap_err();
        assert!(matches!(
            err,
            StateMachineError::DuplicateSection {
                section: SectionType::Version,
                line_number: 3
            }
        ));
    }

    #[test]
    fn test_strict_rejects_ascii_before_curve() {
        let mut sm = SectionStateMachine::strict();
        sm.process_line("~VERSION").unwrap();
        sm.process_line("~WELL").unwrap();
        let err = sm.process_line("~A").unwrap_err();
        assert!(matches!(
            err,
            StateMachineError::AsciiBeforeCurve { line_number: 3 }
        ));
    }

    #[test]
    fn test_strict_rejects_section_after_ascii() {
        let mut sm = SectionStateMachine::strict();
        sm.process_line("~VERSION").unwrap();
        sm.process_line("~WELL").unwrap();
        sm.process_line("~CURVE").unwrap();
        sm.process_line("~A").unwrap();
        sm.process_line("100.0 50.0").unwrap();
        let err = sm.process_line("~OTHER").unwrap_err();
        assert!(matches!(
            err,
            StateMachineError::SectionAfterAscii {
                section: SectionType::Other,
                line_number: 6
            }
        ));
    }

    #[test]
    fn test_strict_finish_missing_mandatory() {
        let mut sm = SectionStateMachine::strict();
        sm.process_line("~VERSION").unwrap();
        sm.process_line("~WELL").unwrap();
        // Missing Curve and Ascii
        let err = sm.finish().unwrap_err();
        if let StateMachineError::MissingRequiredSections { missing } = err {
            assert!(missing.contains(&SectionType::Curve));
            assert!(missing.contains(&SectionType::Ascii));
        } else {
            panic!("Expected MissingRequiredSections error");
        }
    }

    #[test]
    fn test_permissive_mode() {
        let mut sm = SectionStateMachine::permissive();

        // Content before first section is skipped
        let res = sm.process_line("rogue text before sections").unwrap();
        assert_eq!(res, LineAction::Blank);

        // Can start with ~W instead of ~V
        sm.process_line("~WELL").unwrap();
        assert_eq!(sm.current_state(), State::Well);

        // Can transition to ~A without ~C
        sm.process_line("~A").unwrap();
        assert_eq!(sm.current_state(), State::Ascii);
        sm.process_line("100.0").unwrap();

        // Can transition after ~A
        sm.process_line("~VERSION").unwrap();
        assert_eq!(sm.current_state(), State::Version);

        // Can have duplicate sections
        sm.process_line("~WELL").unwrap();
        assert_eq!(sm.current_state(), State::Well);

        // finish() succeeds even if Curve was missing
        assert!(sm.finish().is_ok());
    }

    #[test]
    fn test_reset() {
        let mut sm = SectionStateMachine::strict();
        sm.process_line("~VERSION").unwrap();
        sm.process_line("VERS. 2.0:").unwrap();
        assert_eq!(sm.section_count(), 1);

        sm.reset();
        assert_eq!(sm.current_state(), State::Initial);
        assert_eq!(sm.line_number(), 0);
        assert_eq!(sm.section_count(), 0);
        assert_eq!(sm.line_count_for_section(SectionType::Version), 0);
    }

    #[test]
    fn test_feed_line_helper() {
        let mut sm = SectionStateMachine::strict();
        assert_eq!(sm.feed_line("# comment").unwrap(), None);
        assert_eq!(sm.feed_line("~V").unwrap(), Some(SectionType::Version));
        assert_eq!(sm.feed_line("VERS. 2.0:").unwrap(), None);
        assert_eq!(sm.feed_line("~W").unwrap(), Some(SectionType::Well));
    }
}
