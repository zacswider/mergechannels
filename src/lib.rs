mod colorize;
use colorize::{apply_color_map_py, create_rgb_from_arr_py};
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn mergechannels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_rgb_from_arr_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_color_map_py, m)?)?;
    Ok(())
}
