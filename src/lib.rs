//! TurboLAS Python Extension.

use pyo3::prelude::*;

/// Python module wrapper for TurboLAS.
#[pymodule]
fn _turbolas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

