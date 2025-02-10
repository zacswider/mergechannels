mod basic_extension;
mod basic_ndarray;
mod colorize;
use basic_extension::sum_as_string;
use basic_ndarray::{axpy_py, mult_py};
use colorize::{apply_color_map_py, apply_color_map_with_take_py, create_rgb_from_arr_py};
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn mergechannels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(mult_py, m)?)?;
    m.add_function(wrap_pyfunction!(axpy_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_rgb_from_arr_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_color_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_color_map_with_take_py, m)?)?;
    Ok(())
}
