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

/// Parse and load a colormap from name or values
///
/// Attempts to load a colormap by name if provided, otherwise uses pre-defined values.
/// Returns an error if neither a valid name nor values are provided.
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

/// Verify that all values in a slice are identical
///
/// Checks that all elements equal the first element. Returns the first element
/// if all are equal, or an error if any differ.
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

/// Extract arrays from a Python iterable with a given type and dimensionality
///
/// Generic function that extracts readonly numpy arrays of any extractable type
/// from a Python iterable. Works for both 2D and 3D arrays.
fn extract_arrays<'py, T>(array_references: &Bound<'py, PyAny>) -> PyResult<Vec<T>>
where
    for<'a> T: pyo3::FromPyObject<'a, 'py>,
{
    let array_iterator = array_references
        .try_iter()
        .map_err(|_| PyValueError::new_err("Expected an iterable of arrays"))?;

    let mut arrs: Vec<T> = Vec::new();
    for (i, py_arr_ref) in array_iterator.enumerate() {
        let item = py_arr_ref?;
        let py_arr = item.extract::<T>().map_err(|_| {
            let dtype = item
                .getattr("dtype")
                .map(|d| d.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            let shape = item
                .getattr("shape")
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            PyValueError::new_err(format!(
                "Failed to extract array at index {i}: got dtype={dtype}, shape={shape}"
            ))
        })?;
        arrs.push(py_arr);
    }
    Ok(arrs)
}

/// Build channel configurations from arrays, colormaps, and limits
///
/// Zips together arrays, colormaps, and limit ranges to create ChannelConfig objects.
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

/// Container for extracted 2D mask arrays to manage lifetimes
/// Contains separate vectors (bool and i32) of read-only references to python memory which must
/// live until the end of the function call. The "extracted arrays" are held separately to ensure
/// that their data remains valid and referenced where needed.
struct ExtractedMasks2D<'py> {
    bool_masks: Vec<PyReadonlyArray2<'py, bool>>,
    i32_masks: Vec<PyReadonlyArray2<'py, i32>>,
    /// mask_info tracks metadata about each array
    /// for (is_bool, idx, color, alpha) in &vec {...}
    mask_info: Vec<(
        bool,    // whether the array is bool. If not, it must be i32
        usize,   // the index of the array in the vector containing it (bool_masks or i32_masks)
        [u8; 3], // the RGB color to use for the mask
        f32,     // the alpha blending value
    )>,
}

impl<'py> ExtractedMasks2D<'py> {
    /// Create a new instance
    fn new() -> Self {
        Self {
            bool_masks: Vec::new(),
            i32_masks: Vec::new(),
            mask_info: Vec::new(),
        }
    }

    /// Build Mask2D vec from the extracted arrays
    /// The returned Vec borrows from self, so self must outlive the returned masks
    fn build_masks<'a>(&'a self) -> Vec<colorize::Mask2D<'a>> {
        self.mask_info
            .iter()
            .map(|&(is_bool, idx, color, alpha)| {
                if is_bool {
                    colorize::Mask2D::Bool(colorize::MaskConfig {
                        arr: self.bool_masks[idx].as_array(),
                        color,
                        alpha,
                    })
                } else {
                    colorize::Mask2D::I32(colorize::MaskConfig {
                        arr: self.i32_masks[idx].as_array(),
                        color,
                        alpha,
                    })
                }
            })
            .collect()
    }
}

/// Container for extracted 3D mask arrays to manage lifetimes
/// Contains separate vectors (bool and i32) of read-only references to python memory which must
/// live until the end of the function call. The "extracted arrays" are held separately to ensure
/// that their data remains valid and referenced where needed.
struct ExtractedMasks3D<'py> {
    bool_masks: Vec<PyReadonlyArray3<'py, bool>>,
    i32_masks: Vec<PyReadonlyArray3<'py, i32>>,
    /// mask_info tracks metadata about each array
    /// for (is_bool, idx, color, alpha) in &vec {...}
    mask_info: Vec<(
        bool,    // whether the array is bool. If not, it must be i32
        usize,   // the index of the array in the vector containing it (bool_masks or i32_masks)
        [u8; 3], // the RGB color to use for the mask
        f32,     // the alpha blending value
    )>,
}

impl<'py> ExtractedMasks3D<'py> {
    /// Create a new instance
    fn new() -> Self {
        Self {
            bool_masks: Vec::new(),
            i32_masks: Vec::new(),
            mask_info: Vec::new(),
        }
    }

    /// Build Mask2D vec from the extracted arrays
    /// The returned Vec borrows from self, so self must outlive the returned masks
    fn build_masks<'a>(&'a self) -> Vec<colorize::Mask3D<'a>> {
        self.mask_info
            .iter()
            .map(|&(is_bool, idx, color, alpha)| {
                if is_bool {
                    colorize::Mask3D::Bool(colorize::MaskConfig {
                        arr: self.bool_masks[idx].as_array(),
                        color,
                        alpha,
                    })
                } else {
                    colorize::Mask3D::I32(colorize::MaskConfig {
                        arr: self.i32_masks[idx].as_array(),
                        color,
                        alpha,
                    })
                }
            })
            .collect()
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

/// Colorize a single channel image
///
/// Applies a colormap to a single-channel 2D or 3D array and returns an RGB image.
/// Supports uint8 and uint16 data types with optional parallel processing.
/// Raises ValueError if the colormap name is invalid or array type is unsupported.
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

/// Merge and blend multiple single-channel images into an RGB composite
///
/// Colorizes multiple channels using specified colormaps and blends them together.
/// Supports uint8 and uint16 data types with various blending modes.
/// All arrays must have the same dimensionality (2D or 3D) and data type.
/// Raises ValueError if colormaps are invalid or array properties are inconsistent.
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
                let py_arrs: Vec<PyReadonlyArray2<u8>> = extract_arrays(array_references)?;
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);
                let rgb = colorize::merge_2d_u8(configs, blending, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arrs: Vec<PyReadonlyArray3<u8>> = extract_arrays(array_references)?;
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);
                let rgb = colorize::merge_3d_u8(configs, blending, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(*ndim).into()),
        },
        "uint16" => match ndim {
            2 => {
                let py_arrs: Vec<PyReadonlyArray2<u16>> = extract_arrays(array_references)?;
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);
                let rgb = colorize::merge_2d_u16(configs, blending, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arrs: Vec<PyReadonlyArray3<u16>> = extract_arrays(array_references)?;
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);
                let rgb = colorize::merge_3d_u16(configs, blending, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(*ndim).into()),
        },
        _ => Err(errors::DispatchError::UnsupportedDataType(dtype.clone()).into()),
    }
}
