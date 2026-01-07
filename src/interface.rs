use crate::cmaps;
use crate::colorize;
use crate::errors;
use crate::process;
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

/// Enum to track which dtype vector a mask belongs to
#[derive(Clone, Copy)]
enum MaskDtype {
    Bool,
    I32,
    U8,
    U16,
}

/// Container for extracted 2D mask arrays to manage lifetimes
/// Contains separate vectors for each dtype of read-only references to python memory which must
/// live until the end of the function call. The "extracted arrays" are held separately to ensure
/// that their data remains valid and referenced where needed.
struct ExtractedMasks2D<'py> {
    bool_masks: Vec<PyReadonlyArray2<'py, bool>>,
    i32_masks: Vec<PyReadonlyArray2<'py, i32>>,
    u8_masks: Vec<PyReadonlyArray2<'py, u8>>,
    u16_masks: Vec<PyReadonlyArray2<'py, u16>>,
    /// mask_info tracks metadata about each array
    mask_info: Vec<(
        MaskDtype, // which dtype vector this mask belongs to
        usize,     // the index of the array in the appropriate vector
        [u8; 3],   // the RGB color to use for the mask
        f32,       // the alpha blending value
    )>,
}

impl<'py> ExtractedMasks2D<'py> {
    /// Create a new instance
    fn new() -> Self {
        Self {
            bool_masks: Vec::new(),
            i32_masks: Vec::new(),
            u8_masks: Vec::new(),
            u16_masks: Vec::new(),
            mask_info: Vec::new(),
        }
    }

    /// Build MaskConfig2D slice from the extracted arrays
    /// The returned Vec borrows from self, so self must outlive the returned masks
    fn build_masks<'a>(&'a self) -> Vec<colorize::MaskConfig2D<'a>> {
        self.mask_info
            .iter()
            .map(|&(dtype, idx, color, alpha)| match dtype {
                MaskDtype::Bool => colorize::MaskConfig2D::Bool(colorize::MaskConfig {
                    arr: self.bool_masks[idx].as_array(),
                    color,
                    alpha,
                }),
                MaskDtype::I32 => colorize::MaskConfig2D::I32(colorize::MaskConfig {
                    arr: self.i32_masks[idx].as_array(),
                    color,
                    alpha,
                }),
                MaskDtype::U8 => colorize::MaskConfig2D::U8(colorize::MaskConfig {
                    arr: self.u8_masks[idx].as_array(),
                    color,
                    alpha,
                }),
                MaskDtype::U16 => colorize::MaskConfig2D::U16(colorize::MaskConfig {
                    arr: self.u16_masks[idx].as_array(),
                    color,
                    alpha,
                }),
            })
            .collect()
    }

    /// Compute boundary detection on selected masks and return owned boundary arrays.
    /// The `boundaries_only` slice indicates which masks should have boundary detection applied.
    /// Returns a tuple of (boundary_arrays, indices) where indices maps back to which masks
    /// in mask_info had boundaries computed.
    fn compute_boundaries(&self, boundaries_only: &[bool]) -> Vec<Array2<bool>> {
        let mut boundaries = Vec::new();

        for (i, &(dtype, idx, _color, _alpha)) in self.mask_info.iter().enumerate() {
            if boundaries_only.get(i).copied().unwrap_or(false) {
                let boundary = match dtype {
                    MaskDtype::Bool => process::find_boundaries(self.bool_masks[idx].as_array()),
                    MaskDtype::I32 => process::find_boundaries(self.i32_masks[idx].as_array()),
                    MaskDtype::U8 => process::find_boundaries(self.u8_masks[idx].as_array()),
                    MaskDtype::U16 => process::find_boundaries(self.u16_masks[idx].as_array()),
                };
                boundaries.push(boundary);
            }
        }

        boundaries
    }

    /// Build MaskConfig2D slice from the extracted arrays, using boundary arrays for masks
    /// where boundaries_only is true.
    /// The boundary_arrays must outlive the returned Vec.
    fn build_masks_with_boundaries<'a>(
        &'a self,
        boundaries_only: &[bool],
        boundary_arrays: &'a [Array2<bool>],
    ) -> Vec<colorize::MaskConfig2D<'a>> {
        let mut boundary_idx = 0;
        self.mask_info
            .iter()
            .enumerate()
            .map(|(i, &(dtype, idx, color, alpha))| {
                if boundaries_only.get(i).copied().unwrap_or(false) {
                    // Use the pre-computed boundary array
                    let mask = colorize::MaskConfig2D::Bool(colorize::MaskConfig {
                        arr: boundary_arrays[boundary_idx].view(),
                        color,
                        alpha,
                    });
                    boundary_idx += 1;
                    mask
                } else {
                    // Use the original mask
                    match dtype {
                        MaskDtype::Bool => colorize::MaskConfig2D::Bool(colorize::MaskConfig {
                            arr: self.bool_masks[idx].as_array(),
                            color,
                            alpha,
                        }),
                        MaskDtype::I32 => colorize::MaskConfig2D::I32(colorize::MaskConfig {
                            arr: self.i32_masks[idx].as_array(),
                            color,
                            alpha,
                        }),
                        MaskDtype::U8 => colorize::MaskConfig2D::U8(colorize::MaskConfig {
                            arr: self.u8_masks[idx].as_array(),
                            color,
                            alpha,
                        }),
                        MaskDtype::U16 => colorize::MaskConfig2D::U16(colorize::MaskConfig {
                            arr: self.u16_masks[idx].as_array(),
                            color,
                            alpha,
                        }),
                    }
                }
            })
            .collect()
    }
}

/// Container for extracted 3D mask arrays to manage lifetimes
/// Contains separate vectors for each dtype of read-only references to python memory which must
/// live until the end of the function call. The "extracted arrays" are held separately to ensure
/// that their data remains valid and referenced where needed.
struct ExtractedMasks3D<'py> {
    bool_masks: Vec<PyReadonlyArray3<'py, bool>>,
    i32_masks: Vec<PyReadonlyArray3<'py, i32>>,
    u8_masks: Vec<PyReadonlyArray3<'py, u8>>,
    u16_masks: Vec<PyReadonlyArray3<'py, u16>>,
    /// mask_info tracks metadata about each array
    mask_info: Vec<(
        MaskDtype, // which dtype vector this mask belongs to
        usize,     // the index of the array in the appropriate vector
        [u8; 3],   // the RGB color to use for the mask
        f32,       // the alpha blending value
    )>,
}

impl<'py> ExtractedMasks3D<'py> {
    /// Create a new instance
    fn new() -> Self {
        Self {
            bool_masks: Vec::new(),
            i32_masks: Vec::new(),
            u8_masks: Vec::new(),
            u16_masks: Vec::new(),
            mask_info: Vec::new(),
        }
    }

    /// Build MaskConfig3D vec from the extracted arrays
    /// The returned Vec borrows from self, so self must outlive the returned masks
    fn build_masks<'a>(&'a self) -> Vec<colorize::MaskConfig3D<'a>> {
        self.mask_info
            .iter()
            .map(|&(dtype, idx, color, alpha)| match dtype {
                MaskDtype::Bool => colorize::MaskConfig3D::Bool(colorize::MaskConfig {
                    arr: self.bool_masks[idx].as_array(),
                    color,
                    alpha,
                }),
                MaskDtype::I32 => colorize::MaskConfig3D::I32(colorize::MaskConfig {
                    arr: self.i32_masks[idx].as_array(),
                    color,
                    alpha,
                }),
                MaskDtype::U8 => colorize::MaskConfig3D::U8(colorize::MaskConfig {
                    arr: self.u8_masks[idx].as_array(),
                    color,
                    alpha,
                }),
                MaskDtype::U16 => colorize::MaskConfig3D::U16(colorize::MaskConfig {
                    arr: self.u16_masks[idx].as_array(),
                    color,
                    alpha,
                }),
            })
            .collect()
    }
}

/// Extract 2D mask arrays from Python, handling bool, i32, u8, and u16 dtypes
fn extract_masks_2d<'py>(
    mask_arrays: Option<&Bound<'py, PyAny>>,
    mask_colors: Option<Vec<[u8; 3]>>,
    mask_alphas: Option<Vec<f32>>,
) -> PyResult<Option<ExtractedMasks2D<'py>>> {
    let mask_arrays = match mask_arrays {
        Some(arr) => arr,
        None => return Ok(None),
    };

    let mask_iterator = mask_arrays
        .try_iter()
        .map_err(|_| PyValueError::new_err("Expected an iterable of mask arrays"))?;

    let colors = mask_colors.unwrap_or_default();
    let alphas = mask_alphas.unwrap_or_default();

    let mut extracted = ExtractedMasks2D::new();

    for (i, mask_ref) in mask_iterator.enumerate() {
        let mask_item = mask_ref?;
        let untyped = mask_item.cast::<PyUntypedArray>()?;
        let dtype = untyped.dtype().to_string();

        let color = colors.get(i).copied().unwrap_or([128, 0, 128]); // default purple
        let alpha = alphas.get(i).copied().unwrap_or(0.5);

        // raise an error if we get an alpha value less than 0 or greater than 1
        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err(format!(
                "Alpha value at index {} must be between 0 and 1, got {}",
                i, alpha
            )));
        }

        match dtype.as_str() {
            "bool" => {
                let py_arr = mask_item.extract::<PyReadonlyArray2<bool>>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Failed to extract bool mask at index {i} as 2D array"
                    ))
                })?;
                let idx = extracted.bool_masks.len();
                extracted.bool_masks.push(py_arr);
                extracted
                    .mask_info
                    .push((MaskDtype::Bool, idx, color, alpha));
            }
            "int32" => {
                let py_arr = mask_item.extract::<PyReadonlyArray2<i32>>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Failed to extract int32 mask at index {i} as 2D array"
                    ))
                })?;
                let idx = extracted.i32_masks.len();
                extracted.i32_masks.push(py_arr);
                extracted
                    .mask_info
                    .push((MaskDtype::I32, idx, color, alpha));
            }
            "uint8" => {
                let py_arr = mask_item.extract::<PyReadonlyArray2<u8>>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Failed to extract uint8 mask at index {i} as 2D array"
                    ))
                })?;
                let idx = extracted.u8_masks.len();
                extracted.u8_masks.push(py_arr);
                extracted.mask_info.push((MaskDtype::U8, idx, color, alpha));
            }
            "uint16" => {
                let py_arr = mask_item.extract::<PyReadonlyArray2<u16>>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Failed to extract uint16 mask at index {i} as 2D array"
                    ))
                })?;
                let idx = extracted.u16_masks.len();
                extracted.u16_masks.push(py_arr);
                extracted
                    .mask_info
                    .push((MaskDtype::U16, idx, color, alpha));
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Mask at index {i} has unsupported dtype '{dtype}': expected bool, int32, uint8, or uint16"
                )));
            }
        }
    }

    if extracted.mask_info.is_empty() {
        Ok(None)
    } else {
        Ok(Some(extracted))
    }
}

/// Extract 3D mask arrays from Python, handling bool, i32, u8, and u16 dtypes
fn extract_masks_3d<'py>(
    mask_arrays: Option<&Bound<'py, PyAny>>,
    mask_colors: Option<Vec<[u8; 3]>>,
    mask_alphas: Option<Vec<f32>>,
) -> PyResult<Option<ExtractedMasks3D<'py>>> {
    let mask_arrays = match mask_arrays {
        Some(arr) => arr,
        None => return Ok(None),
    };

    let mask_iterator = mask_arrays
        .try_iter()
        .map_err(|_| PyValueError::new_err("Expected an iterable of mask arrays"))?;

    let colors = mask_colors.unwrap_or_default();
    let alphas = mask_alphas.unwrap_or_default();

    let mut extracted = ExtractedMasks3D::new();

    for (i, mask_ref) in mask_iterator.enumerate() {
        let mask_item = mask_ref?;
        let untyped = mask_item.cast::<PyUntypedArray>()?;
        let dtype = untyped.dtype().to_string();

        let color = colors.get(i).copied().unwrap_or([128, 0, 128]);
        let alpha = alphas.get(i).copied().unwrap_or(0.5);

        // raise an error if we get an alpha value less than 0 or greater than 1
        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err(format!(
                "Alpha value at index {} must be between 0 and 1, got {}",
                i, alpha
            )));
        }

        match dtype.as_str() {
            "bool" => {
                let py_arr = mask_item.extract::<PyReadonlyArray3<bool>>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Failed to extract bool mask at index {i} as 3D array"
                    ))
                })?;
                let idx = extracted.bool_masks.len();
                extracted.bool_masks.push(py_arr);
                extracted
                    .mask_info
                    .push((MaskDtype::Bool, idx, color, alpha));
            }
            "int32" => {
                let py_arr = mask_item.extract::<PyReadonlyArray3<i32>>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Failed to extract int32 mask at index {i} as 3D array"
                    ))
                })?;
                let idx = extracted.i32_masks.len();
                extracted.i32_masks.push(py_arr);
                extracted
                    .mask_info
                    .push((MaskDtype::I32, idx, color, alpha));
            }
            "uint8" => {
                let py_arr = mask_item.extract::<PyReadonlyArray3<u8>>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Failed to extract uint8 mask at index {i} as 3D array"
                    ))
                })?;
                let idx = extracted.u8_masks.len();
                extracted.u8_masks.push(py_arr);
                extracted.mask_info.push((MaskDtype::U8, idx, color, alpha));
            }
            "uint16" => {
                let py_arr = mask_item.extract::<PyReadonlyArray3<u16>>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Failed to extract uint16 mask at index {i} as 3D array"
                    ))
                })?;
                let idx = extracted.u16_masks.len();
                extracted.u16_masks.push(py_arr);
                extracted
                    .mask_info
                    .push((MaskDtype::U16, idx, color, alpha));
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Mask at index {i} has unsupported dtype '{dtype}': expected bool, int32, uint8, or uint16"
                )));
            }
        }
    }

    if extracted.mask_info.is_empty() {
        Ok(None)
    } else {
        Ok(Some(extracted))
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
/// Optional masks can overlay colored regions with alpha blending.
/// When boundaries_only contains true values (only for 2D), mask boundaries are detected
/// for those masks and only boundary pixels are overlaid.
/// Raises ValueError if the colormap name is invalid or array type is unsupported.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "dispatch_single_channel", signature = (array_reference, cmap_name, cmap_values, limits, parallel=false, mask_arrays=None, mask_colors=None, mask_alphas=None, boundaries_only=None))]
pub fn dispatch_single_channel_py<'py>(
    py: Python<'py>,
    array_reference: &Bound<'py, PyAny>,
    cmap_name: Option<String>,
    cmap_values: Option<[[u8; 3]; 256]>,
    limits: [f64; 2],
    parallel: bool,
    mask_arrays: Option<&Bound<'py, PyAny>>,
    mask_colors: Option<Vec<[u8; 3]>>,
    mask_alphas: Option<Vec<f32>>,
    boundaries_only: Option<Vec<bool>>,
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

                // read-only views into python memory - these must live until the function returns
                let extracted = extract_masks_2d(mask_arrays, mask_colors, mask_alphas)?;

                // Get boundaries_only flags, defaulting to empty if None
                let boundaries_flags = boundaries_only.as_deref().unwrap_or(&[]);

                // Compute boundaries for masks where boundaries_only is true
                let boundary_arrays = extracted
                    .as_ref()
                    .map(|ext| ext.compute_boundaries(boundaries_flags));

                // Build masks, using boundary arrays where applicable
                let masks = match (&extracted, &boundary_arrays) {
                    (Some(ext), Some(bounds)) => {
                        Some(ext.build_masks_with_boundaries(boundaries_flags, bounds))
                    }
                    (Some(ext), None) => Some(ext.build_masks()),
                    (None, _) => None,
                };
                let masks_slice = masks.as_deref();

                let rgb = colorize::colorize_single_channel_8bit(config, masks_slice, parallel);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arr = array_reference.extract::<PyReadonlyArray3<u8>>()?;
                let arr = py_arr.as_array();
                let config = colorize::ChannelConfigU83D { arr, cmap, limits };

                // For 3D, boundaries_only is ignored (Python emits warning)
                let extracted = extract_masks_3d(mask_arrays, mask_colors, mask_alphas)?;
                let masks = extracted.as_ref().map(|e| e.build_masks());
                let masks_slice = masks.as_deref();

                let rgb = colorize::colorize_stack_8bit(config, masks_slice, parallel);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(ndim).into()),
        },
        "uint16" => match ndim {
            2 => {
                let py_arr = array_reference.extract::<PyReadonlyArray2<u16>>()?;
                let arr = py_arr.as_array();
                let config = colorize::ChannelConfigU162D { arr, cmap, limits };

                let extracted = extract_masks_2d(mask_arrays, mask_colors, mask_alphas)?;

                // Get boundaries_only flags, defaulting to empty if None
                let boundaries_flags = boundaries_only.as_deref().unwrap_or(&[]);

                // Compute boundaries for masks where boundaries_only is true
                let boundary_arrays = extracted
                    .as_ref()
                    .map(|ext| ext.compute_boundaries(boundaries_flags));

                // Build masks, using boundary arrays where applicable
                let masks = match (&extracted, &boundary_arrays) {
                    (Some(ext), Some(bounds)) => {
                        Some(ext.build_masks_with_boundaries(boundaries_flags, bounds))
                    }
                    (Some(ext), None) => Some(ext.build_masks()),
                    (None, _) => None,
                };
                let masks_slice = masks.as_deref();

                let rgb = colorize::colorize_single_channel_16bit(config, masks_slice, parallel);
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arr = array_reference.extract::<PyReadonlyArray3<u16>>()?;
                let arr = py_arr.as_array();
                let config = colorize::ChannelConfigU163D { arr, cmap, limits };

                // For 3D, boundaries_only is ignored (Python emits warning)
                let extracted = extract_masks_3d(mask_arrays, mask_colors, mask_alphas)?;
                let masks = extracted.as_ref().map(|e| e.build_masks());
                let masks_slice = masks.as_deref();

                let rgb = colorize::colorize_stack_16bit(config, masks_slice, parallel);
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
/// Optional masks can overlay colored regions with alpha blending on the final result.
/// When boundaries_only contains true values (only for 2D), mask boundaries are detected
/// for those masks and only boundary pixels are overlaid.
/// All arrays must have the same dimensionality (2D or 3D) and data type.
/// Raises ValueError if colormaps are invalid or array properties are inconsistent.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "dispatch_multi_channel", signature = (array_references, cmap_names, cmap_values, blending, limits, parallel=false, mask_arrays=None, mask_colors=None, mask_alphas=None, boundaries_only=None))]
pub fn dispatch_multi_channel_py<'py>(
    py: Python<'py>,
    array_references: &Bound<'py, PyAny>,
    cmap_names: Vec<Option<String>>,
    cmap_values: Vec<Option<[[u8; 3]; 256]>>,
    blending: &str,
    limits: Vec<Vec<f64>>,
    parallel: bool,
    mask_arrays: Option<&Bound<'py, PyAny>>,
    mask_colors: Option<Vec<[u8; 3]>>,
    mask_alphas: Option<Vec<f32>>,
    boundaries_only: Option<Vec<bool>>,
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

                let extracted = extract_masks_2d(mask_arrays, mask_colors, mask_alphas)?;

                // Get boundaries_only flags, defaulting to empty if None
                let boundaries_flags = boundaries_only.as_deref().unwrap_or(&[]);

                // Compute boundaries for masks where boundaries_only is true
                let boundary_arrays = extracted
                    .as_ref()
                    .map(|ext| ext.compute_boundaries(boundaries_flags));

                // Build masks, using boundary arrays where applicable
                let masks = match (&extracted, &boundary_arrays) {
                    (Some(ext), Some(bounds)) => {
                        Some(ext.build_masks_with_boundaries(boundaries_flags, bounds))
                    }
                    (Some(ext), None) => Some(ext.build_masks()),
                    (None, _) => None,
                };
                let masks_slice = masks.as_deref();

                let rgb = colorize::merge_2d_u8(configs, blending, masks_slice, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arrs: Vec<PyReadonlyArray3<u8>> = extract_arrays(array_references)?;
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);

                // For 3D, boundaries_only is ignored (Python emits warning)
                let extracted = extract_masks_3d(mask_arrays, mask_colors, mask_alphas)?;
                let masks = extracted.as_ref().map(|e| e.build_masks());
                let masks_slice = masks.as_deref();

                let rgb = colorize::merge_3d_u8(configs, blending, masks_slice, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(*ndim).into()),
        },
        "uint16" => match ndim {
            2 => {
                let py_arrs: Vec<PyReadonlyArray2<u16>> = extract_arrays(array_references)?;
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);

                let extracted = extract_masks_2d(mask_arrays, mask_colors, mask_alphas)?;

                // Get boundaries_only flags, defaulting to empty if None
                let boundaries_flags = boundaries_only.as_deref().unwrap_or(&[]);

                // Compute boundaries for masks where boundaries_only is true
                let boundary_arrays = extracted
                    .as_ref()
                    .map(|ext| ext.compute_boundaries(boundaries_flags));

                // Build masks, using boundary arrays where applicable
                let masks = match (&extracted, &boundary_arrays) {
                    (Some(ext), Some(bounds)) => {
                        Some(ext.build_masks_with_boundaries(boundaries_flags, bounds))
                    }
                    (Some(ext), None) => Some(ext.build_masks()),
                    (None, _) => None,
                };
                let masks_slice = masks.as_deref();

                let rgb = colorize::merge_2d_u16(configs, blending, masks_slice, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            3 => {
                let py_arrs: Vec<PyReadonlyArray3<u16>> = extract_arrays(array_references)?;
                let configs = build_configs(py_arrs.iter().map(|p| p.as_array()), &cmaps, &limits);

                // For 3D, boundaries_only is ignored (Python emits warning)
                let extracted = extract_masks_3d(mask_arrays, mask_colors, mask_alphas)?;
                let masks = extracted.as_ref().map(|e| e.build_masks());
                let masks_slice = masks.as_deref();

                let rgb = colorize::merge_3d_u16(configs, blending, masks_slice, parallel).unwrap();
                Ok(rgb.into_dyn().into_pyarray(py))
            }
            _ => Err(errors::DispatchError::UnsupportedNumberOfDimensions(*ndim).into()),
        },
        _ => Err(errors::DispatchError::UnsupportedDataType(dtype.clone()).into()),
    }
}

/// Create a boundary mask from a 2D array.
///
/// Detects boundary pixels where not all neighbors in a 3x3 window have the same value.
/// This is equivalent to: max_filter(arr, 3, 'reflect') != min_filter(arr, 3, 'reflect')
/// but computed in a single pass without intermediate arrays.
///
/// Supports bool, uint8, int32, and uint16 input arrays.
/// Returns a boolean array where True indicates a boundary pixel.
#[pyfunction]
#[pyo3(name = "create_mask_boundaries")]
pub fn create_mask_boundaries_py<'py>(
    py: Python<'py>,
    array_reference: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let untyped_array = array_reference.cast::<PyUntypedArray>()?;
    let dtype = untyped_array.dtype().to_string();
    let ndim = untyped_array.ndim();

    if ndim != 2 {
        return Err(errors::DispatchError::UnsupportedNumberOfDimensions(ndim).into());
    }

    match dtype.as_str() {
        "bool" => {
            let py_arr = array_reference.extract::<PyReadonlyArray2<bool>>()?;
            Ok(process::find_boundaries(py_arr.as_array()).into_pyarray(py))
        }
        "uint8" => {
            let py_arr = array_reference.extract::<PyReadonlyArray2<u8>>()?;
            Ok(process::find_boundaries(py_arr.as_array()).into_pyarray(py))
        }
        "int32" => {
            let py_arr = array_reference.extract::<PyReadonlyArray2<i32>>()?;
            Ok(process::find_boundaries(py_arr.as_array()).into_pyarray(py))
        }
        "uint16" => {
            let py_arr = array_reference.extract::<PyReadonlyArray2<u16>>()?;
            Ok(process::find_boundaries(py_arr.as_array()).into_pyarray(py))
        }
        _ => Err(PyValueError::new_err(format!(
            "create_mask_boundaries only supports bool, uint8, int32, and uint16 arrays, got dtype '{}'",
            dtype
        ))),
    }
}
