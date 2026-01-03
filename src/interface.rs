use crate::cmaps;
use crate::colorize;
use crate::errors;
use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{Bound, Python};

fn parse_cmap_from_args<'a>(
    cmap_name: &'a Option<String>,
    cmap_values: &'a Option<[[u8; 3]; 256]>,
) -> Result<&'a [[u8; 3]; 256], String> {
    match cmap_name {
        Some(valid_name) => cmaps::try_load_cmap(valid_name),
        None => match cmap_values {
            Some(valid_values) => Ok(valid_values),
            None => Err(
                "Expected either a valid cmap name or a pre-defined colormap, got neither"
                    .to_string(),
            ),
        },
    }
}

/// Get a colormap array by name
///
/// Returns a (256, 3) numpy array of uint8 RGB values for the specified colormap.
/// Raises ValueError if the colormap name is not found.
#[pyfunction]
#[pyo3(name = "get_cmap_array")]
pub fn get_cmap_array_py<'py>(
    py: Python<'py>,
    cmap_name: String,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let cmap = cmaps::try_load_cmap(&cmap_name).map_err(PyValueError::new_err)?;

    // Convert [[u8; 3]; 256] to Array2<u8> with shape (256, 3)
    let mut arr = Array2::<u8>::zeros((256, 3));
    for (i, rgb) in cmap.iter().enumerate() {
        for (j, &val) in rgb.iter().enumerate() {
            arr[[i, j]] = val;
        }
    }

    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "dispatch_single_channel")]
pub fn dispatch_single_channel_py<'py>(
    py: Python<'py>,
    array_reference: &Bound<'py, PyAny>,
    cmap_name: Option<String>,
    cmap_values: Option<[[u8; 3]; 256]>,
    limits: [f64; 2],
    parallel: bool,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let untyped_array = array_reference.cast::<PyUntypedArray>()?;
    let dtype = untyped_array.dtype().to_string();
    let ndim = untyped_array.ndim();
    let cmap = parse_cmap_from_args(&cmap_name, &cmap_values).map_err(PyValueError::new_err)?;
    match dtype.as_str() {
        "uint8" => match ndim {
            2 => {
                let py_arr = array_reference.extract::<PyReadonlyArray2<u8>>()?;
                let arr = py_arr.as_array();
                let config = colorize::ChannelConfigU82D { arr, cmap, limits };
                let rgb = colorize::colorize_single_channel_8bit(config, parallel);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arr = array_reference.extract::<PyReadonlyArray3<u8>>()?;
                let arr = py_arr.as_array();
                let config = colorize::ChannelConfigU83D { arr, cmap, limits };
                let rgb = colorize::colorize_stack_8bit(config, parallel);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(ndim).into()),
        },
        "uint16" => match ndim {
            2 => {
                let py_arr = array_reference.extract::<PyReadonlyArray2<u16>>()?;
                let arr = py_arr.as_array();
                let config = colorize::ChannelConfigU162D { arr, cmap, limits };
                let rgb = colorize::colorize_single_channel_16bit(config, parallel);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arr = array_reference.extract::<PyReadonlyArray3<u16>>()?;
                let arr = py_arr.as_array();
                let config = colorize::ChannelConfigU163D { arr, cmap, limits };
                let rgb = colorize::colorize_stack_16bit(config, parallel);
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

fn build_configs<'a, A>(
    arrays: impl Iterator<Item = A>,
    cmaps: &[&'a [[u8; 3]; 256]],
    limits: &[[f64; 2]],
) -> Vec<colorize::ChannelConfig<'a, A>> {
    arrays
        .zip(cmaps.iter())
        .zip(limits.iter())
        .map(|((arr, &cmap), &limits)| colorize::ChannelConfig { arr, cmap, limits })
        .collect()
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
    parallel: bool,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let mut cmaps: Vec<&[[u8; 3]; 256]> =
        Vec::with_capacity(std::cmp::min(cmap_names.len(), cmap_values.len()));
    for (cmap_name, cmap_value) in cmap_names.iter().zip(cmap_values.iter()) {
        let cmap = parse_cmap_from_args(cmap_name, cmap_value).map_err(PyValueError::new_err)?;
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
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);
                let rgb = colorize::merge_2d_u8(configs, blending, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arrs = extract_3d_u8_arrays(array_references);
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);
                let rgb = colorize::merge_3d_u8(configs, blending, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(*ndim).into()),
        },
        "uint16" => match ndim {
            2 => {
                let py_arrs = extract_2d_u16_arrays(array_references);
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);
                let rgb = colorize::merge_2d_u16(configs, blending, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arrs = extract_3d_u16_arrays(array_references);
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);
                let rgb = colorize::merge_3d_u16(configs, blending, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(*ndim).into()),
        },
        _ => Err(errors::DispatchError::UnsupportedDataType(dtype.clone()).into()),
    }
}
