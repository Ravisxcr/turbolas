//! `TurboLAS` Python Extension.

pub mod error;
pub mod header;
pub mod null;
pub mod section;
pub mod state_machine;

use pyo3::prelude::*;

pub use error::{HeaderError, LASHeaderError, LASSectionError, StateMachineError};
pub use header::{parse_header_line, HeaderItem, HeaderParser};
pub use null::{is_null, map_from_nan, map_to_nan, NullPolicy, NullSentinel};
pub use section::{SectionDelimiter, SectionType};
pub use state_machine::{LineAction, PyLineAction, SectionStateMachine, State, ValidationMode};

/// Python module wrapper for `TurboLAS`.
#[pymodule]
fn _turbolas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<SectionType>()?;
    m.add_class::<SectionDelimiter>()?;
    m.add_class::<State>()?;
    m.add_class::<ValidationMode>()?;
    m.add_class::<PyLineAction>()?;
    m.add_class::<SectionStateMachine>()?;
    m.add_class::<HeaderItem>()?;
    m.add_class::<HeaderParser>()?;
    m.add_class::<NullPolicy>()?;
    m.add_class::<NullSentinel>()?;

    m.add_function(wrap_pyfunction!(parse_header_line, m)?)?;
    m.add_function(wrap_pyfunction!(map_to_nan, m)?)?;
    m.add_function(wrap_pyfunction!(map_from_nan, m)?)?;
    m.add_function(wrap_pyfunction!(is_null, m)?)?;

    // Uppercase aliases for enum variants
    let py = m.py();
    let sec_cls = py.get_type::<SectionType>();
    sec_cls.setattr("VERSION", SectionType::Version)?;
    sec_cls.setattr("WELL", SectionType::Well)?;
    sec_cls.setattr("CURVE", SectionType::Curve)?;
    sec_cls.setattr("PARAMETER", SectionType::Parameter)?;
    sec_cls.setattr("OTHER", SectionType::Other)?;
    sec_cls.setattr("ASCII", SectionType::Ascii)?;

    let state_cls = py.get_type::<State>();
    state_cls.setattr("INITIAL", State::Initial)?;
    state_cls.setattr("VERSION", State::Version)?;
    state_cls.setattr("WELL", State::Well)?;
    state_cls.setattr("CURVE", State::Curve)?;
    state_cls.setattr("PARAMETER", State::Parameter)?;
    state_cls.setattr("OTHER", State::Other)?;
    state_cls.setattr("ASCII", State::Ascii)?;
    state_cls.setattr("COMPLETE", State::Complete)?;

    let val_cls = py.get_type::<ValidationMode>();
    val_cls.setattr("STRICT", ValidationMode::Strict)?;
    val_cls.setattr("PERMISSIVE", ValidationMode::Permissive)?;

    let null_pol_cls = py.get_type::<NullPolicy>();
    null_pol_cls.setattr("STRICT", NullPolicy::Strict)?;
    null_pol_cls.setattr("COMMON", NullPolicy::Common)?;
    null_pol_cls.setattr("AGGRESSIVE", NullPolicy::Aggressive)?;
    null_pol_cls.setattr("NONE", NullPolicy::None)?;
    null_pol_cls.setattr("None_", NullPolicy::None)?;

    m.add("LASSectionError", py.get_type::<LASSectionError>())?;
    m.add("LASHeaderError", py.get_type::<LASHeaderError>())?;

    Ok(())
}
