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

fn parse_cmap_from_args<'a>(
    cmap_name: &'a Option<String>,
    cmap_values: &'a Option<[[u8; 3]; 256]>,
) -> &'a [[u8; 3]; 256] {
    let cmap: &'a [[u8; 3]; 256] = match cmap_name {
        Some(valid_name) => cmaps::load_cmap(valid_name),
        None => match cmap_values {
            Some(valid_values) => valid_values,
            None => {
                panic!("Expected either a valid cmap name or a pre-defined colormap, got neither")
            }
        },
    };
    cmap
}

#[pyfunction]
#[pyo3(name = "dispatch_single_channel")]
pub fn dispatch_single_channel_py<'py>(
    py: Python<'py>,
    array_reference: &Bound<'py, PyAny>,
    cmap_name: Option<String>,
    cmap_values: Option<[[u8; 3]; 256]>,
    limits: [f64; 2],
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let untyped_array = array_reference.cast::<PyUntypedArray>()?;
    let dtype = untyped_array.dtype().to_string();
    let ndim = untyped_array.ndim();
    let cmap = parse_cmap_from_args(&cmap_name, &cmap_values);
    match dtype.as_str() {
        "uint8" => match ndim {
            2 => {
                let py_arr = array_reference.extract::<PyReadonlyArray2<u8>>()?;
                let arr = py_arr.as_array();
                let rgb = colorize::colorize_single_channel_8bit(arr, cmap, limits);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arr = array_reference.extract::<PyReadonlyArray3<u8>>()?;
                let arr = py_arr.as_array();
                let rgb = colorize::colorize_stack_8bit(arr, cmap, limits);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(ndim).into()),
        },
        "uint16" => match ndim {
            2 => {
                let py_arr = array_reference.extract::<PyReadonlyArray2<u16>>()?;
                let arr = py_arr.as_array();
                let rgb = colorize::colorize_single_channel_16bit(arr, cmap, limits);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arr = array_reference.extract::<PyReadonlyArray3<u16>>()?;
                let arr = py_arr.as_array();
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
            "Expected all arrays to have the same dtype, got {dtypes:?}"
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
    cmap_names: Vec<Option<String>>,
    cmap_values: Vec<Option<[[u8; 3]; 256]>>,
    blending: &str,
    limits: Vec<Vec<f64>>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let mut cmaps: Vec<&[[u8; 3]; 256]> =
        Vec::with_capacity(std::cmp::min(cmap_names.len(), cmap_values.len()));
    for (cmap_name, cmap_value) in cmap_names.iter().zip(cmap_values.iter()) {
        let cmap = parse_cmap_from_args(cmap_name, cmap_value);
        cmaps.push(cmap)
    }
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
            let untyped_array = arr_ref.cast::<PyUntypedArray>()?;
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
