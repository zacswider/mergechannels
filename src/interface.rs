use crate::cmaps;
use crate::colorize;
use numpy::ndarray::ArrayView2;
use numpy::PyArrayDyn;
use numpy::PyReadonlyArray;
use numpy::PyUntypedArray;
use numpy::PyUntypedArrayMethods;
use numpy::{Element, PyReadonlyArrayDyn};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{Bound, Python};
use pyo3::types::PyAny;

#[pyfunction]
#[pyo3(name = "dispace_single_channel")]
pub fn dispace_single_channel_py(
    py: Python<'_>,
    array_reference: &Bound<'_, PyAny>,
    low: u8,
    high: u8,
    cmap_name: &str,
// ) -> PyResult<(Bound<'py, PyArrayDyn<u8>)> {
) -> PyResult<(())> {
    Ok(())
}

#[pyfunction]
#[pyo3(name = "test_dynamic_arrays")]
pub fn test_dynamic_arrays_py(
    arr_probably: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let untyped_array = arr_probably.downcast::<PyUntypedArray>()?;
    let _dtype = untyped_array.dtype();
    let s: String = _dtype.to_string();
    println!("dtype={:?}", _dtype);
    println!("s={:?}", s);
    Ok(())
}

#[pyfunction]
#[pyo3(name = "apply_color_map")]
pub fn apply_color_map_py<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    low: u8,
    high: u8,
    cmap_name: &str,
) -> PyResult<(Bound<'py, PyArray3<u8>>)> {
    let untyped_array = arr.downcast::<PyUntypedArray>()?;
    let array_dtype = untyped_array.dtype().to_string();
    /*
    * possible dtypes should be uint8, uint16, uint32, uint64, float16, float32, float64
    */
    match array_dtype {
        "uint8" => ...,
        _ => panic!("unsupported!")
    }
    let array_ndim = untyped_array.ndim();
    if array_ndim != 2 {
        panic!("only 2d arrays supported")
    }
    let downcasted = arr.downcast::<PyReadonlyArrayDyn<'py, Element>>()?;
    // let arr = arr.as_array();
    // let cmap = cmaps::load_cmap(cmap_name);
    // let rgb = colorize::apply_color_map(arr, cmap);
    // rgb.into_pyarray(py)
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
