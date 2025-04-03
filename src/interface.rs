use crate::cmaps;
use crate::colorize;
use numpy::ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{Bound, Python};

#[pyfunction]
#[pyo3(name = "apply_color_map")]
pub fn apply_color_map_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, u8>,
    cmap_name: &str,
) -> Bound<'py, PyArray3<u8>> {
    // Get the ndarray::ArrayView
    // let array_view: ArrayView = arr.as_array();

    // Get the dtype
    // let dtype = array_view.dtype();
    // println!("{dtype}");
    let arr = arr.as_array();
    let cmap = cmaps::load_cmap(cmap_name);
    let rgb = colorize::apply_color_map(arr, cmap);
    rgb.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "apply_colors_and_merge_nc")]
pub fn apply_colors_and_merge_nc_py<'py>(
    py: Python<'py>,
    py_arrs: Vec<PyReadonlyArray2<'py, u8>>,
    cmap_names: Vec<String>,
    blending: &str,
) -> Bound<'py, PyArray3<u8>> {
    let arrs: Vec<ArrayView2<u8>> = py_arrs.iter().map(|py_arr| py_arr.as_array()).collect();
    let cmaps: Vec<&[[u8; 3]; 256]> = cmap_names
        .iter()
        .map(|name| cmaps::load_cmap(name))
        .collect();
    let rgb = colorize::apply_colors_and_merge(arrs, cmaps, blending);
    rgb.into_pyarray(py)
}
