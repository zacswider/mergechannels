use crate::cmaps;
use crate::colorize;
use crate::errors;
use ndarray::ArrayView2;
use ndarray::ArrayView3;
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
    limits: [f64; 2],
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let untyped_array = array_reference.downcast::<PyUntypedArray>()?;
    let dtype = untyped_array.dtype().to_string();
    let ndim = untyped_array.ndim();
    match dtype.as_str() {
        "uint8" => match ndim {
            2 => {
                let py_arr = array_reference.extract::<PyReadonlyArray2<u8>>()?;
                let arr = py_arr.as_array();
                let cmap = cmaps::load_cmap(cmap_name);
                let rgb = colorize::colorize_single_channel_8bit(arr, cmap, limits);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arr = array_reference.extract::<PyReadonlyArray3<u8>>()?;
                let arr = py_arr.as_array();
                let cmap = cmaps::load_cmap(cmap_name);
                let rgb = colorize::colorize_stack_8bit(arr, cmap, limits);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(ndim).into()),
        },
        "uint16" => match ndim {
            2 => {
                let py_arr = array_reference.extract::<PyReadonlyArray2<u16>>()?;
                let arr = py_arr.as_array();
                let cmap = cmaps::load_cmap(cmap_name);
                let rgb = colorize::colorize_single_channel_16bit(arr, cmap, limits);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arr = array_reference.extract::<PyReadonlyArray3<u16>>()?;
                let arr = py_arr.as_array();
                let cmap = cmaps::load_cmap(cmap_name);
                let rgb = colorize::colorize_stack_16bit(arr, cmap, limits);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(ndim).into()),
        },
        _ => Err(errors::DispatchError::UnsupportedDataType(dtype).into()),
    }
}

fn consensus_value<T>(dtypes: &[T]) -> Result<&T, String>
where
    T: PartialEq + std::fmt::Debug,
{
    let (first, rest) = dtypes.split_first().ok_or("No dtypes found".to_string())?;
    if !rest.iter().all(|dtype| dtype == first) {
        return Err(format!(
            "Expected all arrays to have the same dtype, got {:?}",
            dtypes
        ));
    }
    Ok(first)
}

fn extract_2d_u8_arrays<'py>(
    array_references: &Bound<'py, PyAny>,
) -> Vec<PyReadonlyArray2<'py, u8>> {
    let mut arrs: Vec<PyReadonlyArray2<'py, u8>> = Vec::new();
    if let Ok(array_iterator) = array_references.try_iter() {
        for py_arr_ref in array_iterator {
            let py_arr = py_arr_ref
                .unwrap()
                .extract::<PyReadonlyArray2<u8>>()
                .unwrap();
            arrs.push(py_arr);
        }
    }
    arrs
}
fn extract_3d_u8_arrays<'py>(
    array_references: &Bound<'py, PyAny>,
) -> Vec<PyReadonlyArray3<'py, u8>> {
    let mut arrs: Vec<PyReadonlyArray3<'py, u8>> = Vec::new();
    if let Ok(array_iterator) = array_references.try_iter() {
        for py_arr_ref in array_iterator {
            let py_arr = py_arr_ref
                .unwrap()
                .extract::<PyReadonlyArray3<u8>>()
                .unwrap();
            arrs.push(py_arr);
        }
    }
    arrs
}
fn extract_2d_u16_arrays<'py>(
    array_references: &Bound<'py, PyAny>,
) -> Vec<PyReadonlyArray2<'py, u16>> {
    let mut arrs: Vec<PyReadonlyArray2<'py, u16>> = Vec::new();
    if let Ok(array_iterator) = array_references.try_iter() {
        for py_arr_ref in array_iterator {
            let py_arr = py_arr_ref
                .unwrap()
                .extract::<PyReadonlyArray2<u16>>()
                .unwrap();
            arrs.push(py_arr);
        }
    }
    arrs
}
fn extract_3d_u16_arrays<'py>(
    array_references: &Bound<'py, PyAny>,
) -> Vec<PyReadonlyArray3<'py, u16>> {
    let mut arrs: Vec<PyReadonlyArray3<'py, u16>> = Vec::new();
    if let Ok(array_iterator) = array_references.try_iter() {
        for py_arr_ref in array_iterator {
            let py_arr = py_arr_ref
                .unwrap()
                .extract::<PyReadonlyArray3<u16>>()
                .unwrap();
            arrs.push(py_arr);
        }
    }
    arrs
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

    let mut dtypes: Vec<String> = Vec::new();
    let mut ndims: Vec<usize> = Vec::new();
    if let Ok(array_iterator) = array_references.try_iter() {
        for array_reference in array_iterator {
            let arr_ref = array_reference?;
            let untyped_array = arr_ref.downcast::<PyUntypedArray>()?;
            dtypes.push(untyped_array.dtype().to_string());
            ndims.push(untyped_array.ndim());
        }
    }
    let dtype = consensus_value(&dtypes).unwrap();
    let ndim = consensus_value(&ndims).unwrap();
    match dtype.as_str() {
        "uint8" => match ndim {
            2 => {
                let py_arrs = extract_2d_u8_arrays(array_references);
                let arrs: Vec<ArrayView2<u8>> =
                    py_arrs.iter().map(|py_arr| py_arr.as_array()).collect();
                let rgb = colorize::merge_2d_u8(arrs, cmaps, blending, limits).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arrs = extract_3d_u8_arrays(array_references);
                let arrs: Vec<ArrayView3<u8>> =
                    py_arrs.iter().map(|py_arr| py_arr.as_array()).collect();
                let rgb = colorize::merge_3d_u8(arrs, cmaps, blending, limits).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(*ndim).into()),
        },
        "uint16" => match ndim {
            2 => {
                let py_arrs = extract_2d_u16_arrays(array_references);
                let arrs: Vec<ArrayView2<u16>> =
                    py_arrs.iter().map(|py_arr| py_arr.as_array()).collect();
                let rgb = colorize::merge_2d_u16(arrs, cmaps, blending, limits).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arrs = extract_3d_u16_arrays(array_references);
                let arrs: Vec<ArrayView3<u16>> =
                    py_arrs.iter().map(|py_arr| py_arr.as_array()).collect();
                let rgb = colorize::merge_3d_u16(arrs, cmaps, blending, limits).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(*ndim).into()),
        },
        _ => Err(errors::DispatchError::UnsupportedDataType(dtype.clone()).into()),
    }
}
