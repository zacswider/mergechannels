use crate::blend::{self};
use crate::errors::MergeError;
use numpy::ndarray::{Array, Array3, Array4, ArrayView2, ArrayView3};
use rayon::prelude::*;
use smallvec::SmallVec;

pub struct ChannelConfig<'a, A> {
    pub arr: A,
    pub cmap: &'a [[u8; 3]; 256],
    pub limits: [f64; 2],
}

impl<'a, A> ChannelConfig<'a, A> {
    pub fn is_normalized(&self) -> bool {
        self.limits[0] == 0.0 && self.limits[1] == 255.0
    }
}

// Type aliases for clarity
pub type ChannelConfigU82D<'a> = ChannelConfig<'a, ArrayView2<'a, u8>>;
pub type ChannelConfigU83D<'a> = ChannelConfig<'a, ArrayView3<'a, u8>>;
pub type ChannelConfigU162D<'a> = ChannelConfig<'a, ArrayView2<'a, u16>>;
pub type ChannelConfigU163D<'a> = ChannelConfig<'a, ArrayView3<'a, u16>>;

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
pub fn colorize_single_channel_8bit(config: ChannelConfigU82D, parallel: bool) -> Array3<u8> {
    let shape_y = config.arr.shape()[0];
    let shape_x = config.arr.shape()[1];
    let mut rgb = img_to_rgb(config.arr);

    if config.is_normalized() {
        // fast path - direct lookup
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(y, mut row)| {
                    for x in 0..shape_x {
                        let idx = config.arr[[y, x]] as usize;
                        let color = config.cmap[idx];
                        row[[x, 0]] = color[0];
                        row[[x, 1]] = color[1];
                        row[[x, 2]] = color[2];
                    }
                });
        } else {
            for y in 0..shape_y {
                for x in 0..shape_x {
                    let idx = config.arr[[y, x]] as usize;
                    let color = config.cmap[idx];
                    rgb[[y, x, 0]] = color[0];
                    rgb[[y, x, 1]] = color[1];
                    rgb[[y, x, 2]] = color[2];
                }
            }
        }
    } else {
        // normalize on the fly
        let [offset, scale] = offset_and_scale(config.limits);
        // NOTE: I think we could define a "pixel value fn conditioanlly that is applied in an
        // otherwise identical loop. This would cut a ton of boilerplate and the compiler should
        // monomorphize"
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(y, mut row)| {
                    for x in 0..shape_x {
                        let val = config.arr[[y, x]];
                        let idx = as_idx(val, offset, scale);
                        let color = config.cmap[idx];
                        row[[x, 0]] = color[0];
                        row[[x, 1]] = color[1];
                        row[[x, 2]] = color[2];
                    }
                });
        } else {
            for y in 0..shape_y {
                for x in 0..shape_x {
                    let val = config.arr[[y, x]];
                    let idx = as_idx(val, offset, scale);
                    let color = config.cmap[idx];
                    rgb[[y, x, 0]] = color[0];
                    rgb[[y, x, 1]] = color[1];
                    rgb[[y, x, 2]] = color[2];
                }
            }
        }
    }
    rgb
}

/// apply a colormap to a stack of 8 bit images
pub fn colorize_stack_8bit(config: ChannelConfigU83D, parallel: bool) -> Array4<u8> {
    let shape_n = config.arr.shape()[0];
    let shape_y = config.arr.shape()[1];
    let shape_x = config.arr.shape()[2];
    let mut rgb = stack_to_rgb(config.arr);

    if config.is_normalized() {
        // fast path - direct lookup
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(n, mut plane)| {
                    for y in 0..shape_y {
                        for x in 0..shape_x {
                            let idx = config.arr[[n, y, x]] as usize;
                            let color = config.cmap[idx];
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
                        let idx = config.arr[[n, y, x]] as usize;
                        let color = config.cmap[idx];
                        rgb[[n, y, x, 0]] = color[0];
                        rgb[[n, y, x, 1]] = color[1];
                        rgb[[n, y, x, 2]] = color[2];
                    }
                }
            }
        }
    } else {
        // normalize on the fly
        let [offset, scale] = offset_and_scale(config.limits);
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(n, mut plane)| {
                    for y in 0..shape_y {
                        for x in 0..shape_x {
                            let val = config.arr[[n, y, x]];
                            let idx = as_idx(val, offset, scale);
                            let color = config.cmap[idx];
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
                        let val = config.arr[[n, y, x]];
                        let idx = as_idx(val, offset, scale);
                        let color = config.cmap[idx];
                        rgb[[n, y, x, 0]] = color[0];
                        rgb[[n, y, x, 1]] = color[1];
                        rgb[[n, y, x, 2]] = color[2];
                    }
                }
            }
        }
    }
    rgb
}

/// apply a colormap to a single 16 bit image, normalizing the intensity lookups on the fly
pub fn colorize_single_channel_16bit(config: ChannelConfigU162D, parallel: bool) -> Array3<u8> {
    let shape_y = config.arr.shape()[0];
    let shape_x = config.arr.shape()[1];
    let mut rgb = img_to_rgb(config.arr);
    let [offset, scale] = offset_and_scale(config.limits);
    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..shape_x {
                    let val = config.arr[[y, x]];
                    let idx = as_idx(val, offset, scale);
                    let color = config.cmap[idx];
                    row[[x, 0]] = color[0];
                    row[[x, 1]] = color[1];
                    row[[x, 2]] = color[2];
                }
            });
    } else {
        for y in 0..shape_y {
            for x in 0..shape_x {
                let val = config.arr[[y, x]];
                let idx = as_idx(val, offset, scale);
                let color = config.cmap[idx];
                rgb[[y, x, 0]] = color[0];
                rgb[[y, x, 1]] = color[1];
                rgb[[y, x, 2]] = color[2];
            }
        }
    }
    rgb
}

/// apply a colormap to a stack of 16 bit images, normalizing the intensity lookups on the fly
pub fn colorize_stack_16bit(config: ChannelConfigU163D, parallel: bool) -> Array4<u8> {
    let shape_n = config.arr.shape()[0];
    let shape_y = config.arr.shape()[1];
    let shape_x = config.arr.shape()[2];
    let mut rgb = stack_to_rgb(config.arr);
    let [offset, scale] = offset_and_scale(config.limits);
    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(n, mut plane)| {
                for y in 0..shape_y {
                    for x in 0..shape_x {
                        let val = config.arr[[n, y, x]];
                        let idx = as_idx(val, offset, scale);
                        let color = config.cmap[idx];
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
                    let val = config.arr[[n, y, x]];
                    let idx = as_idx(val, offset, scale);
                    let color = config.cmap[idx];
                    rgb[[n, y, x, 0]] = color[0];
                    rgb[[n, y, x, 1]] = color[1];
                    rgb[[n, y, x, 2]] = color[2];
                }
            }
        }
    }
    rgb
}

/// Merge n 2d arrays together
pub fn merge_2d_u8(
    configs: Vec<ChannelConfigU82D>,
    blending: &str,
    parallel: bool,
) -> Result<Array3<u8>, MergeError> {
    let first_config = &configs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_y = first_config.arr.shape()[0];
    let shape_x = first_config.arr.shape()[1];
    let mut rgb = img_to_rgb(first_config.arr);
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    if configs.iter().all(|c| c.is_normalized()) {
        // fast path - direct lookup
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                    for j in 0..shape_x {
                        px_vals.clear();
                        for config in configs.iter() {
                            let idx = config.arr[[i, j]] as usize;
                            let ch_color = config.cmap[idx];
                            px_vals.push(ch_color);
                        }
                        let px_color: [u8; 3] = blend_fn(&px_vals);
                        row[[j, 0]] = px_color[0];
                        row[[j, 1]] = px_color[1];
                        row[[j, 2]] = px_color[2];
                    }
                });
        } else {
            let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
            for i in 0..shape_y {
                for j in 0..shape_x {
                    px_vals.clear();
                    for config in configs.iter() {
                        let idx = config.arr[[i, j]] as usize;
                        let ch_color = config.cmap[idx];
                        px_vals.push(ch_color);
                    }
                    let px_color: [u8; 3] = blend_fn(&px_vals);
                    rgb[[i, j, 0]] = px_color[0];
                    rgb[[i, j, 1]] = px_color[1];
                    rgb[[i, j, 2]] = px_color[2];
                }
            }
        }
    } else {
        // slow path - normalize on the fly
        let limits: Vec<[f64; 2]> = configs.iter().map(|c| c.limits).collect();
        let offsets_and_scales = per_ch_offset_and_scale(limits);
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                    for j in 0..shape_x {
                        px_vals.clear();
                        for (config, offset_scale) in configs.iter().zip(offsets_and_scales.iter())
                        {
                            let [offset, scale] = offset_scale;
                            let val = config.arr[[i, j]];
                            let idx = as_idx(val, *offset, *scale);
                            let ch_color = config.cmap[idx];
                            px_vals.push(ch_color);
                        }
                        let px_color = blend_fn(&px_vals);
                        row[[j, 0]] = px_color[0];
                        row[[j, 1]] = px_color[1];
                        row[[j, 2]] = px_color[2];
                    }
                });
        } else {
            let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
            for i in 0..shape_y {
                for j in 0..shape_x {
                    px_vals.clear();
                    for (config, offset_scale) in configs.iter().zip(offsets_and_scales.iter()) {
                        let [offset, scale] = offset_scale;
                        let val = config.arr[[i, j]];
                        let idx = as_idx(val, *offset, *scale);
                        let ch_color = config.cmap[idx];
                        px_vals.push(ch_color);
                    }
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
    configs: Vec<ChannelConfigU83D>,
    blending: &str,
    parallel: bool,
) -> Result<Array4<u8>, MergeError> {
    let first_config = &configs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_n = first_config.arr.shape()[0];
    let shape_y = first_config.arr.shape()[1];
    let shape_x = first_config.arr.shape()[2];
    let mut rgb = stack_to_rgb(first_config.arr);
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    if configs.iter().all(|c| c.is_normalized()) {
        // fast path - direct lookup
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(n, mut plane)| {
                    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                    for i in 0..shape_y {
                        for j in 0..shape_x {
                            px_vals.clear();
                            for config in configs.iter() {
                                let idx = config.arr[[n, i, j]] as usize;
                                let ch_color = config.cmap[idx];
                                px_vals.push(ch_color);
                            }
                            let px_color: [u8; 3] = blend_fn(&px_vals);
                            plane[[i, j, 0]] = px_color[0];
                            plane[[i, j, 1]] = px_color[1];
                            plane[[i, j, 2]] = px_color[2];
                        }
                    }
                });
        } else {
            let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
            for n in 0..shape_n {
                for i in 0..shape_y {
                    for j in 0..shape_x {
                        px_vals.clear();
                        for config in configs.iter() {
                            let idx = config.arr[[n, i, j]] as usize;
                            let ch_color = config.cmap[idx];
                            px_vals.push(ch_color);
                        }
                        let px_color: [u8; 3] = blend_fn(&px_vals);
                        rgb[[n, i, j, 0]] = px_color[0];
                        rgb[[n, i, j, 1]] = px_color[1];
                        rgb[[n, i, j, 2]] = px_color[2];
                    }
                }
            }
        }
    } else {
        // slow path - normalize on the fly
        let limits: Vec<[f64; 2]> = configs.iter().map(|c| c.limits).collect();
        let offsets_and_scales = per_ch_offset_and_scale(limits);
        if parallel {
            rgb.axis_iter_mut(numpy::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(n, mut plane)| {
                    let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                    for i in 0..shape_y {
                        for j in 0..shape_x {
                            px_vals.clear();
                            for (config, offset_scale) in
                                configs.iter().zip(offsets_and_scales.iter())
                            {
                                let [offset, scale] = offset_scale;
                                let val = config.arr[[n, i, j]];
                                let idx = as_idx(val, *offset, *scale);
                                let ch_color = config.cmap[idx];
                                px_vals.push(ch_color);
                            }
                            let px_color = blend_fn(&px_vals);
                            plane[[i, j, 0]] = px_color[0];
                            plane[[i, j, 1]] = px_color[1];
                            plane[[i, j, 2]] = px_color[2];
                        }
                    }
                });
        } else {
            let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
            for n in 0..shape_n {
                for i in 0..shape_y {
                    for j in 0..shape_x {
                        px_vals.clear();
                        for (config, offset_scale) in configs.iter().zip(offsets_and_scales.iter())
                        {
                            let [offset, scale] = offset_scale;
                            let val = config.arr[[n, i, j]];
                            let idx = as_idx(val, *offset, *scale);
                            let ch_color = config.cmap[idx];
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
    }
    Ok(rgb)
}

pub fn merge_2d_u16(
    configs: Vec<ChannelConfigU162D>,
    blending: &str,
    parallel: bool,
) -> Result<Array3<u8>, MergeError> {
    let first_config = &configs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_y = first_config.arr.shape()[0];
    let shape_x = first_config.arr.shape()[1];
    let mut rgb = img_to_rgb(first_config.arr);
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };
    // slow path - normalize on the fly
    let limits: Vec<[f64; 2]> = configs.iter().map(|c| c.limits).collect();
    let offsets_and_scales = per_ch_offset_and_scale(limits);
    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                for j in 0..shape_x {
                    px_vals.clear();
                    for (config, offset_scale) in configs.iter().zip(offsets_and_scales.iter()) {
                        let [offset, scale] = offset_scale;
                        let val = config.arr[[i, j]];
                        let idx = as_idx(val, *offset, *scale);
                        let ch_color = config.cmap[idx];
                        px_vals.push(ch_color);
                    }
                    let px_color = blend_fn(&px_vals);
                    row[[j, 0]] = px_color[0];
                    row[[j, 1]] = px_color[1];
                    row[[j, 2]] = px_color[2];
                }
            });
    } else {
        let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
        for i in 0..shape_y {
            for j in 0..shape_x {
                px_vals.clear();
                for (config, offset_scale) in configs.iter().zip(offsets_and_scales.iter()) {
                    let [offset, scale] = offset_scale;
                    let val = config.arr[[i, j]];
                    let idx = as_idx(val, *offset, *scale);
                    let ch_color = config.cmap[idx];
                    px_vals.push(ch_color);
                }
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
    configs: Vec<ChannelConfigU163D>,
    blending: &str,
    parallel: bool,
) -> Result<Array4<u8>, MergeError> {
    let first_config = &configs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_n = first_config.arr.shape()[0];
    let shape_y = first_config.arr.shape()[1];
    let shape_x = first_config.arr.shape()[2];
    let mut rgb = stack_to_rgb(first_config.arr);
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    // slow path - normalize on the fly
    let limits: Vec<[f64; 2]> = configs.iter().map(|c| c.limits).collect();
    let offsets_and_scales = per_ch_offset_and_scale(limits);
    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(n, mut plane)| {
                let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
                for i in 0..shape_y {
                    for j in 0..shape_x {
                        px_vals.clear();
                        for (config, offset_scale) in configs.iter().zip(offsets_and_scales.iter())
                        {
                            let [offset, scale] = offset_scale;
                            let val = config.arr[[n, i, j]];
                            let idx = as_idx(val, *offset, *scale);
                            let ch_color = config.cmap[idx];
                            px_vals.push(ch_color);
                        }
                        let px_color = blend_fn(&px_vals);
                        plane[[i, j, 0]] = px_color[0];
                        plane[[i, j, 1]] = px_color[1];
                        plane[[i, j, 2]] = px_color[2];
                    }
                }
            });
    } else {
        let mut px_vals: SmallVec<[[u8; 3]; blend::MAX_N_CH]> = SmallVec::new();
        for n in 0..shape_n {
            for i in 0..shape_y {
                for j in 0..shape_x {
                    px_vals.clear();
                    for (config, offset_scale) in configs.iter().zip(offsets_and_scales.iter()) {
                        let [offset, scale] = offset_scale;
                        let val = config.arr[[n, i, j]];
                        let idx = as_idx(val, *offset, *scale);
                        let ch_color = config.cmap[idx];
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
