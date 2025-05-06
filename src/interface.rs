use crate::cmaps;
use crate::colorize;
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{Bound, Python};

#[pyfunction]
#[pyo3(name = "dispatch_single_channel")]
pub fn dispatch_single_channel_py<'py>(
    py: Python<'py>,
    array_reference: &Bound<'py, PyAny>,
    cmap_name: &str,
    low: f64,
    high: f64,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let untyped_array = array_reference.downcast::<PyUntypedArray>()?;
    let dtype = untyped_array.dtype().to_string();
    let ndim = untyped_array.ndim();
    match dtype.as_str() {
        "uint8" => {
            println!("doing uint8");
            match ndim {
                2 => {
                    println!("doing 2D");
                    let py_arr = array_reference.extract::<PyReadonlyArray2<u8>>()?;
                    let arr = py_arr.as_array();
                    let cmap = cmaps::load_cmap(cmap_name);
                    let rgb = colorize::colorize_single_channel_8bit(arr, low, high, cmap);
                    return Ok(rgb.into_dyn().into_pyarray(py));
                }
                3 => {
                    println!("doing 3D");
                    let py_arr = array_reference.extract::<PyReadonlyArray3<u8>>()?;
                    let arr = py_arr.as_array();
                    let cmap = cmaps::load_cmap(cmap_name);
                    let rgb = colorize::colorize_stack_8bit(arr, low, high, cmap);
                    return Ok(rgb.into_dyn().into_pyarray(py));
                }
                _ => panic!("Recieved unsupported number of dimensions {:?}", ndim),
            }
        }
        "uint16" => {
            println!("doing uint16");
            match ndim {
                2 => {
                    println!("doing 2D");
                    let py_arr = array_reference.extract::<PyReadonlyArray2<u16>>()?;
                    let arr = py_arr.as_array();
                    let cmap = cmaps::load_cmap(cmap_name);
                    let rgb = colorize::colorize_single_channel_16bit(arr, low, high, cmap);
                    return Ok(rgb.into_dyn().into_pyarray(py));
                }
                3 => {
                    println!("doing 3D");
                    let py_arr = array_reference.extract::<PyReadonlyArray3<u16>>()?;
                    let arr = py_arr.as_array();
                    let cmap = cmaps::load_cmap(cmap_name);
                    let rgb = colorize::colorize_stack_16bit(arr, low, high, cmap);
                    return Ok(rgb.into_dyn().into_pyarray(py));
                }
                _ => panic!("Recieved unsupported number of dimensions {:?}", ndim),
            }
        }
        _ => panic!("Received unsupported dtype: {:?}", dtype),
    }
}

#[pyfunction]
#[pyo3(name = "dispatch_multi_channel")]
pub fn dispatch_multi_channel_py<'py>(
    py: Python<'py>,
    array_references: &Bound<'py, PyAny>,
    cmap_names: Vec<String>,
    lows: Vec<f64>,
    highs: Vec<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let cmaps: Vec<&[[u8; 3]; 256]> = cmap_names
        .iter()
        .map(|name| cmaps::load_cmap(name))
        .collect();
    if let Ok(arrs) = array_references
        .try_iter()
        .map(|arr_ref| arr_ref.extract::<PyReadonlyArray2<u8>>())
        .into_iter()
        .collect()
    {
        println!("Processing 2D u8 arrays");
        let rgb = colorize::merge_2d_u8(arrs, cmaps, blending);
    } else if let Ok(arrs) = array_references
        .try_iter()
        .map(|arr_ref| arr_ref.extract::<PyReadonlyArray3<u8>>())
        .into_iter()
        .collect()
    {
        println!("Processing 3D u8 arrays")
    } else if let Ok(arrs) = array_references
        .try_iter()
        .map(|arr_ref| arr_ref.extract::<PyReadonlyArray2<u16>>())
        .into_iter()
        .collect()
    {
        println!("Processing 2D u16 arrays")
    } else if let Ok(arrs) = array_references
        .try_iter()
        .map(|arr_ref| arr_ref.extract::<PyReadonlyArray3<u16>>())
        .into_iter()
        .collect()
    {
        println!("Processing 3D u16 arrays")
    } else {
        println!("Error!")
    }
    Ok(rgb.into_dyn().into_pyarray(py));
}

// #[pyfunction]
// #[pyo3(name = "apply_colors_and_merge_nc")]
// pub fn apply_colors_and_merge_nc_py<'py>(
//     py: Python<'py>,
//     py_arrs: Vec<PyReadonlyArray2<'py, u8>>,
//     cmap_names: Vec<String>,
//     blending: &str,
// ) -> Bound<'py, PyArray3<u8>> {
//     let arrs: Vec<ArrayView2<u8>> = py_arrs.iter().map(|py_arr| py_arr.as_array()).collect();
//     let cmaps: Vec<&[[u8; 3]; 256]> = cmap_names
//         .iter()
//         .map(|name| cmaps::load_cmap(name))
//         .collect();
//     let rgb = colorize::apply_colors_and_merge(arrs, cmaps, blending);
//     rgb.into_pyarray(py)
// }
