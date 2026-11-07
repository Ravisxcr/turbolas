//! `TurboLAS` Python Extension.

pub mod columnar;
pub mod encoding;
pub mod error;
pub mod header;
pub mod null;
pub mod section;
pub mod state_machine;
pub mod unwrapper;
pub mod well;

use pyo3::prelude::*;

pub use columnar::{
    estimate_row_capacity, parse_columnar, parse_columnar_with_sentinel, ColumnarBuffer,
    ColumnarParser, PyColumnarBuffer, PyColumnarParser,
};
pub use encoding::{
    decode_buffer, decode_cp1252, detect_encoding, is_valid_utf8, DecodedBuffer, Encoding,
    PyDecodedBuffer,
};
pub use error::{
    DataError, HeaderError, LASDataError, LASHeaderError, LASSectionError, StateMachineError,
};
pub use header::{parse_header_line, HeaderItem, HeaderParser};
pub use null::{is_null, map_from_nan, map_to_nan, NullPolicy, NullSentinel};
pub use section::{SectionDelimiter, SectionType};
pub use state_machine::{LineAction, PyLineAction, SectionStateMachine, State, ValidationMode};
pub use unwrapper::{
    unwrap_and_parse, PyStreamingUnwrapper, PyUnwrappedRecord, StreamingUnwrapper, UnwrappedRecord,
};
pub use well::{read_las, LasCurve, LasWell, OutputFormat};

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
    m.add_class::<Encoding>()?;
    m.add_class::<PyDecodedBuffer>()?;
    m.add_class::<PyColumnarBuffer>()?;
    m.add_class::<PyColumnarParser>()?;
    m.add_class::<PyUnwrappedRecord>()?;
    m.add_class::<PyStreamingUnwrapper>()?;
    m.add_class::<OutputFormat>()?;
    m.add_class::<LasCurve>()?;
    m.add_class::<LasWell>()?;

    m.add_function(wrap_pyfunction!(parse_header_line, m)?)?;
    m.add_function(wrap_pyfunction!(map_to_nan, m)?)?;
    m.add_function(wrap_pyfunction!(map_from_nan, m)?)?;
    m.add_function(wrap_pyfunction!(is_null, m)?)?;
    m.add_function(wrap_pyfunction!(encoding::py_decode_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(encoding::py_decode, m)?)?;
    m.add_function(wrap_pyfunction!(encoding::py_detect_encoding, m)?)?;
    m.add_function(wrap_pyfunction!(encoding::py_is_valid_utf8, m)?)?;
    m.add_function(wrap_pyfunction!(encoding::py_decode_cp1252, m)?)?;
    m.add_function(wrap_pyfunction!(encoding::py_cp1252_to_char, m)?)?;
    m.add_function(wrap_pyfunction!(columnar::py_parse_columnar, m)?)?;
    m.add_function(wrap_pyfunction!(columnar::py_estimate_row_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(unwrapper::py_unwrap_and_parse, m)?)?;
    m.add_function(wrap_pyfunction!(well::read_las, m)?)?;

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

    let enc_cls = py.get_type::<Encoding>();
    enc_cls.setattr("ASCII", Encoding::Ascii)?;
    enc_cls.setattr("UTF8", Encoding::Utf8)?;
    enc_cls.setattr("UTF_8", Encoding::Utf8)?;
    enc_cls.setattr("UTF8_BOM", Encoding::Utf8Bom)?;
    enc_cls.setattr("UTF8BOM", Encoding::Utf8Bom)?;
    enc_cls.setattr("WINDOWS_1252", Encoding::Windows1252)?;
    enc_cls.setattr("WINDOWS1252", Encoding::Windows1252)?;
    enc_cls.setattr("CP1252", Encoding::Windows1252)?;

    let out_fmt_cls = py.get_type::<OutputFormat>();
    out_fmt_cls.setattr("NUMPY", OutputFormat::NumPy)?;
    out_fmt_cls.setattr("ARROW", OutputFormat::Arrow)?;
    out_fmt_cls.setattr("BYTES", OutputFormat::Bytes)?;
    out_fmt_cls.setattr("LIST", OutputFormat::List)?;

    m.add("LASSectionError", py.get_type::<LASSectionError>())?;
    m.add("LASHeaderError", py.get_type::<LASHeaderError>())?;
    m.add("LASDataError", py.get_type::<LASDataError>())?;

    Ok(())
}
