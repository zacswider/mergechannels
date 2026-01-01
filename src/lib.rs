mod blend;
mod cmaps;
mod colorize;
mod errors;
mod interface;

use interface::{dispatch_multi_channel_py, dispatch_single_channel_py, get_cmap_array_py};
use pyo3::prelude::*;

/// This module is thread-safe and supports free-threaded Python (Python 3.13+ without GIL).
/// All functions can be called concurrently from multiple threads without synchronization.
#[pymodule]
fn mergechannels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Declare that this module doesn't rely on the GIL for thread safety
    // This allows the module to run without re-enabling the GIL in free-threaded Python
    m.gil_used(false)?;

    m.add_function(wrap_pyfunction!(dispatch_single_channel_py, m)?)?;
    m.add_function(wrap_pyfunction!(dispatch_multi_channel_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_cmap_array_py, m)?)?;
    Ok(())
}
