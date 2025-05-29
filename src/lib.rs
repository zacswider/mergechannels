mod blend;
mod cmaps;
mod colorize;
mod errors;
mod interface;

use interface::{dispatch_multi_channel_py, dispatch_single_channel_py};
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn mergechannels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dispatch_single_channel_py, m)?)?;
    m.add_function(wrap_pyfunction!(dispatch_multi_channel_py, m)?)?;
    Ok(())
}
