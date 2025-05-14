use crate::cmaps;
use crate::colorize;
use ndarray::ArrayView2;
use ndarray::ArrayView3;
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{Bound, Python};

#[derive(Debug)]
pub enum DispatchError {
    UnsupportedDataType(String),
    UnsupportedNumberOfDimensions(usize),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::UnsupportedDataType(dtype) => {
                write!(f, "Received unsupported dtype: {}", dtype)
            }
            DispatchError::UnsupportedNumberOfDimensions(ndim) => {
                write!(f, "Received unsupported number of dimensions: {}", ndim)
            }
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<DispatchError> for PyErr {
    fn from(err: DispatchError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

#[pyfunction]
#[pyo3(name = "dispatch_single_channel")]
pub fn dispatch_single_channel_py<'py>(
    py: Python<'py>,
    array_reference: &Bound<'py, PyAny>,
    cmap_name: &str,
    limits: [f64; 2],
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let untyped_array = array_reference.downcast::<PyUntypedArray>()?;
    let dtype = untyped_array.dtype().to_string();
    let ndim = untyped_array.ndim();
    match dtype.as_str() {
        "uint8" => {
            println!("doing uint8");
            match ndim {
                2 => {
                    println!("Processing 2D uint8 image");
                    let py_arr = array_reference.extract::<PyReadonlyArray2<u8>>()?;
                    let arr = py_arr.as_array();
                    let cmap = cmaps::load_cmap(cmap_name);
                    let rgb = colorize::colorize_single_channel_8bit(arr, cmap, limits);
                    return Ok(rgb.into_dyn().into_pyarray(py));
                }
                3 => {
                    println!("doing 3D");
                    let py_arr = array_reference.extract::<PyReadonlyArray3<u8>>()?;
                    let arr = py_arr.as_array();
                    let cmap = cmaps::load_cmap(cmap_name);
                    let rgb = colorize::colorize_stack_8bit(arr, cmap, limits);
                    return Ok(rgb.into_dyn().into_pyarray(py));
                }
                _ => {
                    return Err(DispatchError::UnsupportedNumberOfDimensions(ndim).into());
                }
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
                    let rgb = colorize::colorize_single_channel_16bit(arr, cmap, limits);
                    return Ok(rgb.into_dyn().into_pyarray(py));
                }
                3 => {
                    println!("doing 3D");
                    let py_arr = array_reference.extract::<PyReadonlyArray3<u16>>()?;
                    let arr = py_arr.as_array();
                    let cmap = cmaps::load_cmap(cmap_name);
                    let rgb = colorize::colorize_stack_16bit(arr, cmap, limits);
                    return Ok(rgb.into_dyn().into_pyarray(py));
                }
                _ => {
                    return Err(DispatchError::UnsupportedNumberOfDimensions(ndim).into());
                }
            }
        }
        _ => {
            return Err(DispatchError::UnsupportedDataType(dtype).into());
        }
    }
}

#[pyfunction]
#[pyo3(name = "dispatch_multi_channel")]
pub fn dispatch_multi_channel_py<'py>(
    py: Python<'py>,
    array_references: &Bound<'py, PyAny>,
    cmap_names: Vec<String>,
    blending: &str,
    limits: Vec<Vec<f64>>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let cmaps: Vec<&[[u8; 3]; 256]> = cmap_names
        .iter()
        .map(|name| cmaps::load_cmap(name))
        .collect(); // TODO don't panic inside load_cmap
    let limits = limits
        .into_iter()
        .map(|v| {
            v.as_slice()
                .try_into()
                .map_err(|_| format!("Expected a vector of length 2, got {}", v.len()))
        })
        .collect::<Result<Vec<[f64; 2]>, _>>()
        .unwrap();
    if let Ok(arrs) = array_references
        .try_iter()
        .map(|arr_ref| arr_ref.extract::<PyReadonlyArray2<u8>>())
        .into_iter()
        .collect::<Result<Vec<PyReadonlyArray2<u8>>, pyo3::PyErr>>()
    {
        println!("Processing 2D u8 arrays");
        let arrs: Vec<ArrayView2<u8>> = arrs.iter().map(|py_arr| py_arr.as_array()).collect();
        let rgb = colorize::merge_2d_u8(arrs, cmaps, blending, limits).unwrap();
        return Ok(rgb.into_dyn().into_pyarray(py));
    } else if let Ok(arrs) = array_references
        .try_iter()
        .map(|arr_ref| arr_ref.extract::<PyReadonlyArray3<u8>>())
        .into_iter()
        .collect::<Result<Vec<PyReadonlyArray3<u8>>, pyo3::PyErr>>()
    {
        println!("Processing 3D u8 arrays");
        let arrs: Vec<ArrayView3<u8>> = arrs.iter().map(|py_arr| py_arr.as_array()).collect();
        let rgb = colorize::merge_3d_u8(arrs, cmaps, blending, limits).unwrap();
        return Ok(rgb.into_dyn().into_pyarray(py));
    } else if let Ok(arrs) = array_references
        .try_iter()
        .map(|arr_ref| arr_ref.extract::<PyReadonlyArray2<u16>>())
        .into_iter()
        .collect::<Result<Vec<PyReadonlyArray2<u16>>, pyo3::PyErr>>()
    {
        println!("Processing 2D u16 arrays");
        let arrs: Vec<ArrayView2<u16>> = arrs.iter().map(|py_arr| py_arr.as_array()).collect();
        let rgb = colorize::merge_2d_u16(arrs, cmaps, blending, limits).unwrap();
        return Ok(rgb.into_dyn().into_pyarray(py));
    } else if let Ok(arrs) = array_references
        .try_iter()
        .map(|arr_ref| arr_ref.extract::<PyReadonlyArray3<u16>>())
        .into_iter()
        .collect::<Result<Vec<PyReadonlyArray3<u16>>, pyo3::PyErr>>()
    {
        println!("Processing 3D u16 arrays");
        let arrs: Vec<ArrayView3<u16>> = arrs.iter().map(|py_arr| py_arr.as_array()).collect();
        let rgb = colorize::merge_3d_u16(arrs, cmaps, blending, limits).unwrap();
        return Ok(rgb.into_dyn().into_pyarray(py));
    } else {
        println!("Error!");
        panic!("");
    }
}
