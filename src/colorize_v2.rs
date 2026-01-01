//! Refactored colorize module using inline helpers and closures to reduce code duplication.
//! This version extracts common patterns while preserving all fast-path optimizations.

use crate::blend::{self};
use crate::errors::MergeError;
use numpy::ndarray::{Array, Array3, Array4, ArrayView2, ArrayView3};
use rayon::prelude::*;
use smallvec::SmallVec;

// ============================================================================
// Core Helpers
// ============================================================================

/// Create a (y, x, 3) array with ones
#[inline(always)]
fn img_to_rgb<T>(a: ArrayView2<T>) -> Array3<u8> {
    Array::ones((a.shape()[0], a.shape()[1], 3))
}

/// Create a (n, y, x, 3) array with ones
#[inline(always)]
fn stack_to_rgb<T>(a: ArrayView3<T>) -> Array4<u8> {
    Array::ones((a.shape()[0], a.shape()[1], a.shape()[2], 3))
}

/// Pre-calculate the offset and scale factors for normalization
#[inline(always)]
fn offset_and_scale(lowhigh: [f64; 2]) -> [f32; 2] {
    let [low, high] = lowhigh;
    let offset = low as f32;
    let range = high - low;
    let scale = if range.abs() > 1e-9 {
        255.0 / range
    } else {
        0.0
    };
    [offset, scale as f32]
}

/// Pre-calculate offset and scale for each channel
#[inline]
fn per_ch_offset_and_scale(limits: Vec<[f64; 2]>) -> SmallVec<[[f32; 2]; blend::MAX_N_CH]> {
    limits.into_iter().map(offset_and_scale).collect()
}

/// Normalize a value to a colormap index
#[inline(always)]
fn as_idx<T: Into<f32>>(val: T, offset: f32, scale: f32) -> usize {
    let normalized = (val.into() - offset) * scale;
    normalized.clamp(0.0, 255.0) as usize
}

/// Check if all limits allow direct lookup (0-255 range for u8)
#[inline(always)]
fn all_normalized(limits: &[[f64; 2]]) -> bool {
    limits
        .iter()
        .all(|&[low, high]| low == 0.0 && high == 255.0)
}

/// Resolve blending mode string to function pointer
#[inline]
fn get_blend_fn(blending: &str) -> Result<blend::BlendFn, MergeError> {
    match blending {
        "max" => Ok(blend::max_blending),
        "sum" => Ok(blend::sum_blending),
        "min" => Ok(blend::min_blending),
        "mean" => Ok(blend::mean_blending),
        _ => Err(MergeError::InvalidBlendingMode(blending.to_string())),
    }
}

// ============================================================================
// Pixel Processing Closures - Single Channel
// ============================================================================

/// Process a 2D image with a given pixel operation (parallel or sequential)
#[inline]
fn process_2d<F>(rgb: &mut Array3<u8>, shape_y: usize, shape_x: usize, parallel: bool, pixel_fn: F)
where
    F: Fn(usize, usize) -> [u8; 3] + Sync,
{
    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..shape_x {
                    let color = pixel_fn(y, x);
                    row[[x, 0]] = color[0];
                    row[[x, 1]] = color[1];
                    row[[x, 2]] = color[2];
                }
            });
    } else {
        for y in 0..shape_y {
            for x in 0..shape_x {
                let color = pixel_fn(y, x);
                rgb[[y, x, 0]] = color[0];
                rgb[[y, x, 1]] = color[1];
                rgb[[y, x, 2]] = color[2];
            }
        }
    }
}

/// Process a 3D stack with a given pixel operation (parallel or sequential)
#[inline]
fn process_3d<F>(
    rgb: &mut Array4<u8>,
    shape_n: usize,
    shape_y: usize,
    shape_x: usize,
    parallel: bool,
    pixel_fn: F,
) where
    F: Fn(usize, usize, usize) -> [u8; 3] + Sync,
{
    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(n, mut plane)| {
                for y in 0..shape_y {
                    for x in 0..shape_x {
                        let color = pixel_fn(n, y, x);
                        plane[[y, x, 0]] = color[0];
                        plane[[y, x, 1]] = color[1];
                        plane[[y, x, 2]] = color[2];
                    }
                }
            });
    } else {
        for n in 0..shape_n {
            for y in 0..shape_y {
                for x in 0..shape_x {
                    let color = pixel_fn(n, y, x);
                    rgb[[n, y, x, 0]] = color[0];
                    rgb[[n, y, x, 1]] = color[1];
                    rgb[[n, y, x, 2]] = color[2];
                }
            }
        }
    }
}

// ============================================================================
// Single Channel Colorize Functions
// ============================================================================

/// Apply a colormap to a single 8-bit image
pub fn colorize_single_channel_8bit(
    arr: ArrayView2<u8>,
    cmap: &[[u8; 3]; 256],
    limits: [f64; 2],
    parallel: bool,
) -> Array3<u8> {
    let shape_y = arr.shape()[0];
    let shape_x = arr.shape()[1];
    let mut rgb = img_to_rgb(arr);

    if limits[0] == 0.0 && limits[1] == 255.0 {
        // Fast path - direct lookup
        process_2d(&mut rgb, shape_y, shape_x, parallel, |y, x| {
            cmap[arr[[y, x]] as usize]
        });
    } else {
        // Normalization path
        let [offset, scale] = offset_and_scale(limits);
        process_2d(&mut rgb, shape_y, shape_x, parallel, |y, x| {
            cmap[as_idx(arr[[y, x]], offset, scale)]
        });
    }
    rgb
}

/// Apply a colormap to a single 16-bit image
pub fn colorize_single_channel_16bit(
    arr: ArrayView2<u16>,
    cmap: &[[u8; 3]; 256],
    limits: [f64; 2],
    parallel: bool,
) -> Array3<u8> {
    let shape_y = arr.shape()[0];
    let shape_x = arr.shape()[1];
    let mut rgb = img_to_rgb(arr);
    let [offset, scale] = offset_and_scale(limits);

    process_2d(&mut rgb, shape_y, shape_x, parallel, |y, x| {
        cmap[as_idx(arr[[y, x]], offset, scale)]
    });
    rgb
}

/// Apply a colormap to a stack of 8-bit images
pub fn colorize_stack_8bit(
    arr: ArrayView3<u8>,
    cmap: &[[u8; 3]; 256],
    limits: [f64; 2],
    parallel: bool,
) -> Array4<u8> {
    let shape_n = arr.shape()[0];
    let shape_y = arr.shape()[1];
    let shape_x = arr.shape()[2];
    let mut rgb = stack_to_rgb(arr);

    if limits[0] == 0.0 && limits[1] == 255.0 {
        // Fast path - direct lookup
        process_3d(&mut rgb, shape_n, shape_y, shape_x, parallel, |n, y, x| {
            cmap[arr[[n, y, x]] as usize]
        });
    } else {
        // Normalization path
        let [offset, scale] = offset_and_scale(limits);
        process_3d(&mut rgb, shape_n, shape_y, shape_x, parallel, |n, y, x| {
            cmap[as_idx(arr[[n, y, x]], offset, scale)]
        });
    }
    rgb
}

/// Apply a colormap to a stack of 16-bit images
pub fn colorize_stack_16bit(
    arr: ArrayView3<u16>,
    cmap: &[[u8; 3]; 256],
    limits: [f64; 2],
    parallel: bool,
) -> Array4<u8> {
    let shape_n = arr.shape()[0];
    let shape_y = arr.shape()[1];
    let shape_x = arr.shape()[2];
    let mut rgb = stack_to_rgb(arr);
    let [offset, scale] = offset_and_scale(limits);

    process_3d(&mut rgb, shape_n, shape_y, shape_x, parallel, |n, y, x| {
        cmap[as_idx(arr[[n, y, x]], offset, scale)]
    });
    rgb
}

// ============================================================================
// Merge Functions - Using thread-local SmallVec for pixel accumulation
// ============================================================================

/// Merge n 2D u8 arrays together
pub fn merge_2d_u8(
    arrs: Vec<ArrayView2<u8>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    blending: &str,
    limits: Vec<[f64; 2]>,
    parallel: bool,
) -> Result<Array3<u8>, MergeError> {
    let first_arr = arrs[0];
    let shape_y = first_arr.shape()[0];
    let shape_x = first_arr.shape()[1];
    let mut rgb = img_to_rgb(first_arr);
    let blend_fn = get_blend_fn(blending)?;

    if all_normalized(&limits) {
        // Fast path - direct lookup
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(y, mut row)| {
                    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                    for x in 0..shape_x {
                        px_vals.clear();
                        for (arr, cmap) in arrs.iter().zip(cmaps.iter()) {
                            px_vals.push(cmap[arr[[y, x]] as usize]);
                        }
                        let color = blend_fn(&px_vals);
                        row[[x, 0]] = color[0];
                        row[[x, 1]] = color[1];
                        row[[x, 2]] = color[2];
                    }
                });
        } else {
            let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
            for y in 0..shape_y {
                for x in 0..shape_x {
                    px_vals.clear();
                    for (arr, cmap) in arrs.iter().zip(cmaps.iter()) {
                        px_vals.push(cmap[arr[[y, x]] as usize]);
                    }
                    let color = blend_fn(&px_vals);
                    rgb[[y, x, 0]] = color[0];
                    rgb[[y, x, 1]] = color[1];
                    rgb[[y, x, 2]] = color[2];
                }
            }
        }
    } else {
        // Normalization path
        let offsets_and_scales = per_ch_offset_and_scale(limits);
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(y, mut row)| {
                    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                    for x in 0..shape_x {
                        px_vals.clear();
                        for ((arr, cmap), [offset, scale]) in
                            arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                        {
                            px_vals.push(cmap[as_idx(arr[[y, x]], *offset, *scale)]);
                        }
                        let color = blend_fn(&px_vals);
                        row[[x, 0]] = color[0];
                        row[[x, 1]] = color[1];
                        row[[x, 2]] = color[2];
                    }
                });
        } else {
            let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
            for y in 0..shape_y {
                for x in 0..shape_x {
                    px_vals.clear();
                    for ((arr, cmap), [offset, scale]) in
                        arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                    {
                        px_vals.push(cmap[as_idx(arr[[y, x]], *offset, *scale)]);
                    }
                    let color = blend_fn(&px_vals);
                    rgb[[y, x, 0]] = color[0];
                    rgb[[y, x, 1]] = color[1];
                    rgb[[y, x, 2]] = color[2];
                }
            }
        }
    }
    Ok(rgb)
}

/// Merge n 2D u16 arrays together
pub fn merge_2d_u16(
    arrs: Vec<ArrayView2<u16>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    blending: &str,
    limits: Vec<[f64; 2]>,
    parallel: bool,
) -> Result<Array3<u8>, MergeError> {
    let first_arr = arrs[0];
    let shape_y = first_arr.shape()[0];
    let shape_x = first_arr.shape()[1];
    let mut rgb = img_to_rgb(first_arr);
    let blend_fn = get_blend_fn(blending)?;
    let offsets_and_scales = per_ch_offset_and_scale(limits);

    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                for x in 0..shape_x {
                    px_vals.clear();
                    for ((arr, cmap), [offset, scale]) in
                        arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                    {
                        px_vals.push(cmap[as_idx(arr[[y, x]], *offset, *scale)]);
                    }
                    let color = blend_fn(&px_vals);
                    row[[x, 0]] = color[0];
                    row[[x, 1]] = color[1];
                    row[[x, 2]] = color[2];
                }
            });
    } else {
        let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
        for y in 0..shape_y {
            for x in 0..shape_x {
                px_vals.clear();
                for ((arr, cmap), [offset, scale]) in
                    arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                {
                    px_vals.push(cmap[as_idx(arr[[y, x]], *offset, *scale)]);
                }
                let color = blend_fn(&px_vals);
                rgb[[y, x, 0]] = color[0];
                rgb[[y, x, 1]] = color[1];
                rgb[[y, x, 2]] = color[2];
            }
        }
    }
    Ok(rgb)
}

/// Merge n 3D u8 arrays together
pub fn merge_3d_u8(
    arrs: Vec<ArrayView3<u8>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    blending: &str,
    limits: Vec<[f64; 2]>,
    parallel: bool,
) -> Result<Array4<u8>, MergeError> {
    let first_arr = arrs[0];
    let shape_n = first_arr.shape()[0];
    let shape_y = first_arr.shape()[1];
    let shape_x = first_arr.shape()[2];
    let mut rgb = stack_to_rgb(first_arr);
    let blend_fn = get_blend_fn(blending)?;

    if all_normalized(&limits) {
        // Fast path - direct lookup
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(n, mut plane)| {
                    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                    for y in 0..shape_y {
                        for x in 0..shape_x {
                            px_vals.clear();
                            for (arr, cmap) in arrs.iter().zip(cmaps.iter()) {
                                px_vals.push(cmap[arr[[n, y, x]] as usize]);
                            }
                            let color = blend_fn(&px_vals);
                            plane[[y, x, 0]] = color[0];
                            plane[[y, x, 1]] = color[1];
                            plane[[y, x, 2]] = color[2];
                        }
                    }
                });
        } else {
            let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
            for n in 0..shape_n {
                for y in 0..shape_y {
                    for x in 0..shape_x {
                        px_vals.clear();
                        for (arr, cmap) in arrs.iter().zip(cmaps.iter()) {
                            px_vals.push(cmap[arr[[n, y, x]] as usize]);
                        }
                        let color = blend_fn(&px_vals);
                        rgb[[n, y, x, 0]] = color[0];
                        rgb[[n, y, x, 1]] = color[1];
                        rgb[[n, y, x, 2]] = color[2];
                    }
                }
            }
        }
    } else {
        // Normalization path
        let offsets_and_scales = per_ch_offset_and_scale(limits);
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(n, mut plane)| {
                    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                    for y in 0..shape_y {
                        for x in 0..shape_x {
                            px_vals.clear();
                            for ((arr, cmap), [offset, scale]) in
                                arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                            {
                                px_vals.push(cmap[as_idx(arr[[n, y, x]], *offset, *scale)]);
                            }
                            let color = blend_fn(&px_vals);
                            plane[[y, x, 0]] = color[0];
                            plane[[y, x, 1]] = color[1];
                            plane[[y, x, 2]] = color[2];
                        }
                    }
                });
        } else {
            let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
            for n in 0..shape_n {
                for y in 0..shape_y {
                    for x in 0..shape_x {
                        px_vals.clear();
                        for ((arr, cmap), [offset, scale]) in
                            arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                        {
                            px_vals.push(cmap[as_idx(arr[[n, y, x]], *offset, *scale)]);
                        }
                        let color = blend_fn(&px_vals);
                        rgb[[n, y, x, 0]] = color[0];
                        rgb[[n, y, x, 1]] = color[1];
                        rgb[[n, y, x, 2]] = color[2];
                    }
                }
            }
        }
    }
    Ok(rgb)
}

/// Merge n 3D u16 arrays together
pub fn merge_3d_u16(
    arrs: Vec<ArrayView3<u16>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    blending: &str,
    limits: Vec<[f64; 2]>,
    parallel: bool,
) -> Result<Array4<u8>, MergeError> {
    let first_arr = arrs[0];
    let shape_n = first_arr.shape()[0];
    let shape_y = first_arr.shape()[1];
    let shape_x = first_arr.shape()[2];
    let mut rgb = stack_to_rgb(first_arr);
    let blend_fn = get_blend_fn(blending)?;
    let offsets_and_scales = per_ch_offset_and_scale(limits);

    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(n, mut plane)| {
                let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                for y in 0..shape_y {
                    for x in 0..shape_x {
                        px_vals.clear();
                        for ((arr, cmap), [offset, scale]) in
                            arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                        {
                            px_vals.push(cmap[as_idx(arr[[n, y, x]], *offset, *scale)]);
                        }
                        let color = blend_fn(&px_vals);
                        plane[[y, x, 0]] = color[0];
                        plane[[y, x, 1]] = color[1];
                        plane[[y, x, 2]] = color[2];
                    }
                }
            });
    } else {
        let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
        for n in 0..shape_n {
            for y in 0..shape_y {
                for x in 0..shape_x {
                    px_vals.clear();
                    for ((arr, cmap), [offset, scale]) in
                        arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                    {
                        px_vals.push(cmap[as_idx(arr[[n, y, x]], *offset, *scale)]);
                    }
                    let color = blend_fn(&px_vals);
                    rgb[[n, y, x, 0]] = color[0];
                    rgb[[n, y, x, 1]] = color[1];
                    rgb[[n, y, x, 2]] = color[2];
                }
            }
        }
    }
    Ok(rgb)
}
