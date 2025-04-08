mod blend;
mod cmaps;
mod colorize;
mod interface;

use interface::{apply_color_map_py, apply_colors_and_merge_nc_py};
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn mergechannels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_color_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_colors_and_merge_nc_py, m)?)?;
    Ok(())
}
