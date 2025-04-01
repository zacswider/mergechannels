mod colorize;
use colorize::{apply_color_map_py, apply_colors_and_merge_2c_py, apply_colors_and_merge_3c_py, apply_colors_and_merge_4c_py};
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn mergechannels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_color_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_colors_and_merge_2c_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_colors_and_merge_3c_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_colors_and_merge_4c_py, m)?)?;
    Ok(())
}
