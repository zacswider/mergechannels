use crate::cmaps;
use crate::colorize;
use numpy::ndarray::ArrayView2;
use numpy::npyffi::{npy_uint32, npy_uint8};
use numpy::{PyArrayDyn, PyUntypedArray, PyUntypedArrayMethods, PyReadonlyArray, Element, PyReadonlyArrayDyn, IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{Bound, Python};

#[pyfunction]
#[pyo3(name = "dispatch_single_channel")]
pub fn dispatch_single_channel_py(
    py: Python<'_>,
    array_reference: &Bound<'_, PyAny>,
    low: u8,
    high: u8,
    ndim: usize,
    dtype: &str,
    cmap_name: &str,
    // ) -> PyResult<(Bound<'py, PyArrayDyn<u8>)> {
) -> PyResult<(())> {
    let untyped_array = array_reference.downcast::<PyUntypedArray>()?;
    let arr_dtype = untyped_array.dtype().to_string();
    println!("arr dtype is {:?}", arr_dtype);
    match arr_dtype.as_str() {
        "uint8" => {
            println!("processing uint8 array!");
            let arr_dyn_8bit = untyped_array.extract::<PyReadonlyArrayDyn<'_, u8>>()?;

        },
        "uint16" => println!("processing uint16 array!"),
        _ => panic!("Received unsupported dtype: {:?}", arr_dtype),
    }
    Ok(())
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
