use crate::blend::{self};
use crate::errors::MergeError;
use numpy::ndarray::{Array, Array3, Array4, ArrayView2, ArrayView3};
use smallvec::SmallVec;

/// Create a (y, x, 3) array with ones
fn img_to_rgb<T>(a: ArrayView2<T>) -> Array3<u8> {
    Array::ones((a.shape()[0], a.shape()[1], 3))
}

/// Create a (n, y, x, 3) array with ones, where n represents a time, z, or batch dimension
fn stack_to_rgb<T>(a: ArrayView3<T>) -> Array4<u8> {
    Array::ones((a.shape()[0], a.shape()[1], a.shape()[2], 3))
}

///pre-calculate the offset and scale factors for a given channel
fn offset_and_scale(lowhigh: [f64; 2]) -> [f32; 2] {
    let [low, high] = lowhigh;
    let offset = low as f32;
    let range = high - low;
    let scale = if range.abs() > 1e-9 {
        255.0 / range
    } else {
        0.0
    };
    let scale = scale as f32;
    [offset, scale]
}

///pre-calculate the offset and scale factors for each channel
///returned object is a SmallVec containing arrays with the offset and scale for each channel
fn per_ch_offset_and_scale(limits: Vec<[f64; 2]>) -> SmallVec<[[f32; 2]; blend::MAX_N_CH]> {
    let mut limit_vals: SmallVec<[[f32; 2]; blend::MAX_N_CH]> = SmallVec::new();
    for lowhigh in limits.into_iter() {
        let offset_scale = offset_and_scale(lowhigh);
        limit_vals.push(offset_scale);
    }
    limit_vals
}

/// normalize a u8 value as a colormap index given pre-calculated offset and scale values
fn as_idx<T>(val: T, offset: f32, scale: f32) -> usize
where
    T: Into<f32>,
{
    let normalized_value = ((val.into()) - offset) * scale;
    normalized_value.clamp(0.0, 255.0) as usize
}

///apply a colormap to a single 8-bit image
pub fn colorize_single_channel_8bit(
    arr: ArrayView2<u8>,
    cmap: &[[u8; 3]; 256],
    limits: [f64; 2],
) -> Array3<u8> {
    let shape_y = arr.shape()[0];
    let shape_x = arr.shape()[1];
    let mut rgb = img_to_rgb(arr);

    if limits[0] == 0.0 && limits[1] == 255.0 {
        // fast path - direct lookup
        for y in 0..shape_y {
            for x in 0..shape_x {
                let idx = arr[[y, x]] as usize;
                let color = cmap[idx];
                rgb[[y, x, 0]] = color[0];
                rgb[[y, x, 1]] = color[1];
                rgb[[y, x, 2]] = color[2];
            }
        }
    } else {
        // normalize on the fly
        let [offset, scale] = offset_and_scale(limits);
        for y in 0..shape_y {
            for x in 0..shape_x {
                let val = arr[[y, x]];
                let idx = as_idx(val, offset, scale);
                let color = cmap[idx];
                rgb[[y, x, 0]] = color[0];
                rgb[[y, x, 1]] = color[1];
                rgb[[y, x, 2]] = color[2];
            }
        }
    }
    rgb
}

/// apply a colormap to a stack of 8 bit images
pub fn colorize_stack_8bit(
    arr: ArrayView3<u8>,
    cmap: &[[u8; 3]; 256],
    limits: [f64; 2],
) -> Array4<u8> {
    let shape_n = arr.shape()[0];
    let shape_y = arr.shape()[1];
    let shape_x = arr.shape()[2];
    let mut rgb = stack_to_rgb(arr);

    if limits[0] == 0.0 && limits[1] == 255.0 {
        // fast path - direct lookup
        for n in 0..shape_n {
            for y in 0..shape_y {
                for x in 0..shape_x {
                    let idx = arr[[n, y, x]] as usize;
                    let color = cmap[idx];
                    rgb[[n, y, x, 0]] = color[0];
                    rgb[[n, y, x, 1]] = color[1];
                    rgb[[n, y, x, 2]] = color[2];
                }
            }
        }
    } else {
        // normalize on the fly
        let [offset, scale] = offset_and_scale(limits);
        for n in 0..shape_n {
            for y in 0..shape_y {
                for x in 0..shape_x {
                    let val = arr[[n, y, x]];
                    let idx = as_idx(val, offset, scale);
                    let color = cmap[idx];
                    rgb[[n, y, x, 0]] = color[0];
                    rgb[[n, y, x, 1]] = color[1];
                    rgb[[n, y, x, 2]] = color[2];
                }
            }
        }
    }
    rgb
}

/// apply a colormap to a single 16 bit image, normalizing the intensity lookups on the fly
pub fn colorize_single_channel_16bit(
    arr: ArrayView2<u16>,
    cmap: &[[u8; 3]; 256],
    limits: [f64; 2],
) -> Array3<u8> {
    let shape_y = arr.shape()[0];
    let shape_x = arr.shape()[1];
    let mut rgb = img_to_rgb(arr);
    let [offset, scale] = offset_and_scale(limits);
    for y in 0..shape_y {
        for x in 0..shape_x {
            let val = arr[[y, x]];
            let idx = as_idx(val, offset, scale);
            let color = cmap[idx];
            rgb[[y, x, 0]] = color[0];
            rgb[[y, x, 1]] = color[1];
            rgb[[y, x, 2]] = color[2];
        }
    }
    rgb
}

/// apply a colormap to a stack of 16 bit images, normalizing the intensity lookups on the fly
pub fn colorize_stack_16bit(
    arr: ArrayView3<u16>,
    cmap: &[[u8; 3]; 256],
    limits: [f64; 2],
) -> Array4<u8> {
    let shape_n = arr.shape()[0];
    let shape_y = arr.shape()[1];
    let shape_x = arr.shape()[2];
    let mut rgb = stack_to_rgb(arr);
    let [offset, scale] = offset_and_scale(limits);
    for n in 0..shape_n {
        for y in 0..shape_y {
            for x in 0..shape_x {
                let val = arr[[n, y, x]];
                let idx = as_idx(val, offset, scale);
                let color = cmap[idx];
                rgb[[n, y, x, 0]] = color[0];
                rgb[[n, y, x, 1]] = color[1];
                rgb[[n, y, x, 2]] = color[2];
            }
        }
    }
    rgb
}

/// check if all limits for a series of u8 ArrayViews are 0.0 and 255.0
fn all_normalized(limits: &[[f64; 2]]) -> bool {
    limits
        .iter()
        .all(|&[low, high]| low == 0.0 && high == 255.0)
}

/// Merge n 2d arrays together
pub fn merge_2d_u8(
    arrs: Vec<ArrayView2<u8>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    blending: &str,
    limits: Vec<[f64; 2]>,
) -> Result<Array3<u8>, MergeError> {
    let first_arr = arrs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_y = first_arr.shape()[0];
    let shape_x = first_arr.shape()[1];
    let mut rgb = img_to_rgb(first_arr);
    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    if all_normalized(&limits) {
        // fast path - direct lookup
        for i in 0..shape_y {
            for j in 0..shape_x {
                px_vals.clear();
                for (arr, cmap) in arrs.iter().zip(cmaps.iter()) {
                    let idx = arr[[i, j]] as usize;
                    let ch_color = cmap[idx];
                    px_vals.push(ch_color);
                }
                let px_color: [u8; 3] = blend_fn(&px_vals);
                rgb[[i, j, 0]] = px_color[0];
                rgb[[i, j, 1]] = px_color[1];
                rgb[[i, j, 2]] = px_color[2];
            }
        }
    } else {
        // slow path - normalize on the fly
        let offsets_and_scales = per_ch_offset_and_scale(limits);
        for i in 0..shape_y {
            for j in 0..shape_x {
                px_vals.clear();
                for ((arr, cmap), offset_scale) in
                    arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                {
                    let [offset, scale] = offset_scale;
                    let val = arr[[i, j]];
                    let idx = as_idx(val, *offset, *scale);
                    let ch_color = cmap[idx];
                    px_vals.push(ch_color);
                    let px_color = blend_fn(&px_vals);
                    rgb[[i, j, 0]] = px_color[0];
                    rgb[[i, j, 1]] = px_color[1];
                    rgb[[i, j, 2]] = px_color[2];
                }
            }
        }
    }
    Ok(rgb)
}

/// Merge n 3d arrays together
pub fn merge_3d_u8(
    arrs: Vec<ArrayView3<u8>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    blending: &str,
    limits: Vec<[f64; 2]>,
) -> Result<Array4<u8>, MergeError> {
    let first_arr = arrs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_n = first_arr.shape()[0];
    let shape_y = first_arr.shape()[1];
    let shape_x = first_arr.shape()[2];
    let mut rgb = stack_to_rgb(first_arr);
    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    if all_normalized(&limits) {
        // fast path - direct lookup
        for n in 0..shape_n {
            for i in 0..shape_y {
                for j in 0..shape_x {
                    px_vals.clear();
                    for (arr, cmap) in arrs.iter().zip(cmaps.iter()) {
                        let idx = arr[[n, i, j]] as usize;
                        let ch_color = cmap[idx];
                        px_vals.push(ch_color);
                    }
                    let px_color: [u8; 3] = blend_fn(&px_vals);
                    rgb[[n, i, j, 0]] = px_color[0];
                    rgb[[n, i, j, 1]] = px_color[1];
                    rgb[[n, i, j, 2]] = px_color[2];
                }
            }
        }
    } else {
        // slow path - normalize on the fly
        let offsets_and_scales = per_ch_offset_and_scale(limits);
        for n in 0..shape_n {
            for i in 0..shape_y {
                for j in 0..shape_x {
                    px_vals.clear();
                    for ((arr, cmap), offset_scale) in
                        arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                    {
                        let [offset, scale] = offset_scale;
                        let val = arr[[n, i, j]];
                        let idx = as_idx(val, *offset, *scale);
                        let ch_color = cmap[idx];
                        px_vals.push(ch_color);
                    }
                    let px_color = blend_fn(&px_vals);
                    rgb[[n, i, j, 0]] = px_color[0];
                    rgb[[n, i, j, 1]] = px_color[1];
                    rgb[[n, i, j, 2]] = px_color[2];
                }
            }
        }
    }
    Ok(rgb)
}

pub fn merge_2d_u16(
    arrs: Vec<ArrayView2<u16>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    blending: &str,
    limits: Vec<[f64; 2]>,
) -> Result<Array3<u8>, MergeError> {
    let first_arr = arrs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_y = first_arr.shape()[0];
    let shape_x = first_arr.shape()[1];
    let mut rgb = img_to_rgb(first_arr);
    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };
    // slow path - normalize on the fly
    let offsets_and_scales = per_ch_offset_and_scale(limits);
    for i in 0..shape_y {
        for j in 0..shape_x {
            px_vals.clear();
            for ((arr, cmap), offset_scale) in
                arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
            {
                let [offset, scale] = offset_scale;
                let val = arr[[i, j]];
                let idx = as_idx(val, *offset, *scale);
                let ch_color = cmap[idx];
                px_vals.push(ch_color);
                let px_color = blend_fn(&px_vals);
                rgb[[i, j, 0]] = px_color[0];
                rgb[[i, j, 1]] = px_color[1];
                rgb[[i, j, 2]] = px_color[2];
            }
        }
    }
    Ok(rgb)
}

pub fn merge_3d_u16(
    arrs: Vec<ArrayView3<u16>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    blending: &str,
    limits: Vec<[f64; 2]>,
) -> Result<Array4<u8>, MergeError> {
    let first_arr = arrs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_n = first_arr.shape()[0];
    let shape_y = first_arr.shape()[1];
    let shape_x = first_arr.shape()[2];
    let mut rgb = stack_to_rgb(first_arr);
    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    // slow path - normalize on the fly
    let offsets_and_scales = per_ch_offset_and_scale(limits);
    for n in 0..shape_n {
        for i in 0..shape_y {
            for j in 0..shape_x {
                px_vals.clear();
                for ((arr, cmap), offset_scale) in
                    arrs.iter().zip(cmaps.iter()).zip(offsets_and_scales.iter())
                {
                    let [offset, scale] = offset_scale;
                    let val = arr[[n, i, j]];
                    let idx = as_idx(val, *offset, *scale);
                    let ch_color = cmap[idx];
                    px_vals.push(ch_color);
                }
                let px_color = blend_fn(&px_vals);
                rgb[[n, i, j, 0]] = px_color[0];
                rgb[[n, i, j, 1]] = px_color[1];
                rgb[[n, i, j, 2]] = px_color[2];
            }
        }
    }
    Ok(rgb)
}
