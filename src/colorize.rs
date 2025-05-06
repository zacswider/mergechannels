use std::collections::HashMap;

use crate::blend;
use numpy::ndarray::{Array, Array3, Array4, ArrayView2, ArrayView3};
use pyo3::panic;

/// Create a (y, x, 3) array with ones
fn img_to_rgb<T>(a: ArrayView2<T>) -> Array3<u8> {
    Array::ones((a.shape()[0], a.shape()[1], 3))
}

/// Create a (n, y, x, 3) array with ones, where n represents a time, z, or batch dimension
fn stack_to_rgb<T>(a: ArrayView3<T>) -> Array4<u8> {
    Array::ones((a.shape()[0], a.shape()[1], a.shape()[2], 3))
}

///pre-calculate the offset and scale factors for a given channel
fn offset_and_scale(low: f64, high: f64) -> [f32; 2] {
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
fn per_ch_offset_and_scale(
    limits: Vec<&[f64; 2]>,
) -> HashMap<usize, [f32; 2]> {
    let mut hm = HashMap::new();
    for (idx, tup) in limits.iter().enumerate() {
        let offset_scale = offset_and_scale(tup[0], tup[1]);
        hm.insert(idx, offset_scale);
    }
    hm
}

///apply a colormap to a single 8-bit image
pub fn colorize_single_channel_8bit(
    arr: ArrayView2<u8>,
    low: f64,
    high: f64,
    cmap: &[[u8; 3]; 256],
) -> Array3<u8> {
    let shape_y = arr.shape()[0];
    let shape_x = arr.shape()[1];
    let mut rgb = img_to_rgb(arr);

    if low == 0.0 && high == 255.0 {
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
        let [offset, scale] = offset_and_scale(low, high);
        for y in 0..shape_y {
            for x in 0..shape_x {
                let value = arr[[y, x]];
                let normalized_value = ((value as f32) - offset) * scale;
                let idx = normalized_value.clamp(0.0, 255.0) as usize;
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
    low: f64,
    high: f64,
    cmap: &[[u8; 3]; 256],
) -> Array4<u8> {
    let shape_n = arr.shape()[0];
    let shape_y = arr.shape()[1];
    let shape_x = arr.shape()[2];
    let mut rgb = stack_to_rgb(arr);

    if low == 0.0 && high == 255.0 {
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
        let [offset, scale] = offset_and_scale(low, high);
        for n in 0..shape_n {
            for y in 0..shape_y {
                for x in 0..shape_x {
                    let value = arr[[n, y, x]];
                    let normalized_value = ((value as f32) - offset) * scale;
                    let idx = normalized_value.clamp(0.0, 255.0) as usize;
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
    low: f64,
    high: f64,
    cmap: &[[u8; 3]; 256],
) -> Array3<u8> {
    let shape_y = arr.shape()[0];
    let shape_x = arr.shape()[1];
    let mut rgb = img_to_rgb(arr);
    let [offset, scale] = offset_and_scale(low, high);
    for y in 0..shape_y {
        for x in 0..shape_x {
            let value = arr[[y, x]];
            let normalized_value = ((value as f32) - offset) * scale;
            let idx = normalized_value.clamp(0.0, 255.0) as usize;
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
    low: f64,
    high: f64,
    cmap: &[[u8; 3]; 256],
) -> Array4<u8> {
    let shape_n = arr.shape()[0];
    let shape_y = arr.shape()[1];
    let shape_x = arr.shape()[2];
    let mut rgb = stack_to_rgb(arr);
    let [offset, scale] = offset_and_scale(low, high);
    for n in 0..shape_n {
        for y in 0..shape_y {
            for x in 0..shape_x {
                let value = arr[[n, y, x]];
                let normalized_value = ((value as f32) - offset) * scale;
                let idx = normalized_value.clamp(0.0, 255.0) as usize;
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

fn all_normalized(limits: &Vec<&[f64; 2]>) -> bool {
    limits.iter().all(|&[low, high]| *low == 0.0 && *high == 255.0)
}

pub fn merge_2d_u8(
    arrs: Vec<ArrayView2<u8>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    limits: Vec<&[f64; 2]>,
    blending: &str,
) -> Array3<u8> {
    let first_arr = arrs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_y = first_arr.shape()[0];
    let shape_x = first_arr.shape()[1];
    let mut rgb = img_to_rgb(first_arr);

    if all_normalized(&limits) {
        // fast path - direct lookup
        println!("doing the fast path")
    } else {
        let per_ch_norms = per_ch_offset_and_scale(limits);
    }


    for i in 0..shape_y {
        for j in 0..shape_x {
            let mut px_vals: Vec<[u8; 3]> = Vec::with_capacity(arrs.len());
            for (ch, (arr, cmap)) in arrs.iter().zip(cmaps.iter()).enumerate() {
                if let Some(ch_norms) = per_ch_norms.get(&ch) {
                    let [offset, scale] = ch_norms;
                } else {
                    panic!("Tried to find the norm values for {}, but they could not be found", ch);
                }
                let idx = arr[[i, j]] as usize;
                let color = cmap[idx];
                px_vals.push(color);
                let color: [u8; 3] = match blending {
                    "max" => {
                        blend::max_blending(&px_vals)
                    }
                    "sum" => {
                        blend::sum_blending(&px_vals)
                    }
                    "min" => {
                        blend::min_blending(&px_vals)
                    }
                    "mean" => {
                        blend::mean_blending(&px_vals)
                    }
                    _ => panic!("received invalid argument for `blending`: {blending}, valid arguments are 'max', 'sum', 'min', and 'mean'")
                };
                rgb[[i, j, 0]] = color[0];
                rgb[[i, j, 1]] = color[1];
                rgb[[i, j, 2]] = color[2];
            }
        }
    }
    rgb
}
