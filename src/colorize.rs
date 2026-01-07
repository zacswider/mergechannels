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

/// Configuration for a binary mask overlay
/// The mask is applied on top of the merged result using alpha blending
/// For bool masks: true = apply mask color, false = no change
/// For i32 masks: any non-zero value = apply mask color, zero = no change
pub struct MaskConfig<A> {
    pub arr: A,
    pub color: [u8; 3], // RGB color for the mask
    pub alpha: f32,     // Alpha value for blending (0.0-1.0)
}

/// Enum to hold either bool or i32 mask types for 2D arrays
#[allow(dead_code)] // Variants constructed via Python interface
pub enum MaskConfig2D<'a> {
    Bool(MaskConfig<ArrayView2<'a, bool>>),
    I32(MaskConfig<ArrayView2<'a, i32>>),
}

/// Enum to hold either bool or i32 mask types for 3D arrays
#[allow(dead_code)] // Variants constructed via Python interface
pub enum MaskConfig3D<'a> {
    Bool(MaskConfig<ArrayView3<'a, bool>>),
    I32(MaskConfig<ArrayView3<'a, i32>>),
}

impl<'a> MaskConfig2D<'a> {
    #[inline]
    pub fn is_active(&self, i: usize, j: usize) -> bool {
        match self {
            MaskConfig2D::Bool(m) => m.arr[[i, j]],
            MaskConfig2D::I32(m) => m.arr[[i, j]] != 0,
        }
    }

    #[inline]
    pub fn color(&self) -> [u8; 3] {
        match self {
            MaskConfig2D::Bool(m) => m.color,
            MaskConfig2D::I32(m) => m.color,
        }
    }

    #[inline]
    pub fn alpha(&self) -> f32 {
        match self {
            MaskConfig2D::Bool(m) => m.alpha,
            MaskConfig2D::I32(m) => m.alpha,
        }
    }
}

impl<'a> MaskConfig3D<'a> {
    #[inline]
    pub fn is_active(&self, n: usize, i: usize, j: usize) -> bool {
        match self {
            MaskConfig3D::Bool(m) => m.arr[[n, i, j]],
            MaskConfig3D::I32(m) => m.arr[[n, i, j]] != 0,
        }
    }

    #[inline]
    pub fn color(&self) -> [u8; 3] {
        match self {
            MaskConfig3D::Bool(m) => m.color,
            MaskConfig3D::I32(m) => m.color,
        }
    }

    #[inline]
    pub fn alpha(&self) -> f32 {
        match self {
            MaskConfig3D::Bool(m) => m.alpha,
            MaskConfig3D::I32(m) => m.alpha,
        }
    }
}

/// Trait for optional mask application - allows monomorphization
/// When M = NoMasks, the apply method is a no-op that compiles away
/// When M = MaskSlice2D/MaskSlice3D, actual mask blending occurs
trait MaskApplicator2D: Sync {
    fn apply(&self, color: [u8; 3], y: usize, x: usize) -> [u8; 3];
}

trait MaskApplicator3D: Sync {
    fn apply(&self, color: [u8; 3], n: usize, y: usize, x: usize) -> [u8; 3];
}

/// Zero-cost abstraction for "no masks" case - compiles to nothing
#[derive(Clone, Copy)]
struct NoMasks;

impl MaskApplicator2D for NoMasks {
    #[inline(always)]
    fn apply(&self, color: [u8; 3], _y: usize, _x: usize) -> [u8; 3] {
        color
    }
}

impl MaskApplicator3D for NoMasks {
    #[inline(always)]
    fn apply(&self, color: [u8; 3], _n: usize, _y: usize, _x: usize) -> [u8; 3] {
        color
    }
}

/// Wrapper for actual mask slice - performs blending
struct MaskSlice2D<'a>(&'a [MaskConfig2D<'a>]);

impl<'a> MaskApplicator2D for MaskSlice2D<'a> {
    #[inline]
    fn apply(&self, mut color: [u8; 3], y: usize, x: usize) -> [u8; 3] {
        for mask in self.0 {
            if mask.is_active(y, x) {
                color = blend::alpha_blend(color, mask.color(), mask.alpha());
            }
        }
        color
    }
}

struct MaskSlice3D<'a>(&'a [MaskConfig3D<'a>]);

impl<'a> MaskApplicator3D for MaskSlice3D<'a> {
    #[inline]
    fn apply(&self, mut color: [u8; 3], n: usize, y: usize, x: usize) -> [u8; 3] {
        for mask in self.0 {
            if mask.is_active(n, y, x) {
                color = blend::alpha_blend(color, mask.color(), mask.alpha());
            }
        }
        color
    }
}

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

/// Inner implementation for applying a colormap to a 2D array with generic mask applicator
/// The M generic parameter allows monomorphization:
/// - NoMasks: mask application compiles away to nothing
/// - MaskSlice2D: actual mask blending is performed
fn apply_colormap_2d_inner<T, F, M>(
    arr: ArrayView2<T>,
    cmap: &[[u8; 3]; 256],
    masks: &M,
    rgb: &mut Array3<u8>,
    parallel: bool,
    idx_fn: F,
) where
    T: Copy + Sync + Send,
    F: Fn(T) -> usize + Sync + Send,
    M: MaskApplicator2D,
{
    let shape_x = arr.shape()[1];
    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..shape_x {
                    let val = arr[[y, x]];
                    let idx = idx_fn(val);
                    let color = cmap[idx];
                    let color = masks.apply(color, y, x);
                    row[[x, 0]] = color[0];
                    row[[x, 1]] = color[1];
                    row[[x, 2]] = color[2];
                }
            });
    } else {
        let shape_y = arr.shape()[0];
        for y in 0..shape_y {
            for x in 0..shape_x {
                let val = arr[[y, x]];
                let idx = idx_fn(val);
                let color = cmap[idx];
                let color = masks.apply(color, y, x);
                rgb[[y, x, 0]] = color[0];
                rgb[[y, x, 1]] = color[1];
                rgb[[y, x, 2]] = color[2];
            }
        }
    }
}

/// Apply a colormap to a 2D array with optional mask overlay
/// Dispatches to monomorphized inner function based on mask presence
fn apply_colormap_2d<'a, T, F>(
    arr: ArrayView2<T>,
    cmap: &[[u8; 3]; 256],
    masks: Option<&'a [MaskConfig2D<'a>]>,
    rgb: &mut Array3<u8>,
    parallel: bool,
    idx_fn: F,
) where
    T: Copy + Sync + Send,
    F: Fn(T) -> usize + Sync + Send + Clone,
{
    match masks {
        Some(m) if !m.is_empty() => {
            apply_colormap_2d_inner(arr, cmap, &MaskSlice2D(m), rgb, parallel, idx_fn);
        }
        _ => {
            apply_colormap_2d_inner(arr, cmap, &NoMasks, rgb, parallel, idx_fn);
        }
    }
}

///apply a colormap to a single 8-bit image
pub fn colorize_single_channel_8bit<'a>(
    config: ChannelConfigU82D<'a>,
    masks: Option<&'a [MaskConfig2D<'a>]>,
    parallel: bool,
) -> Array3<u8> {
    let mut rgb = img_to_rgb(config.arr);

    if config.is_normalized() {
        apply_colormap_2d(config.arr, config.cmap, masks, &mut rgb, parallel, |v| {
            v as usize
        });
    } else {
        let [offset, scale] = offset_and_scale(config.limits);
        apply_colormap_2d(config.arr, config.cmap, masks, &mut rgb, parallel, |v| {
            as_idx(v, offset, scale)
        });
    }
    rgb
}

/// Inner implementation for applying a colormap to a 3D array with generic mask applicator
/// The M generic parameter allows monomorphization:
/// - NoMasks: mask application compiles away to nothing
/// - MaskSlice3D: actual mask blending is performed
fn apply_colormap_3d_inner<T, F, M>(
    arr: ArrayView3<T>,
    cmap: &[[u8; 3]; 256],
    masks: &M,
    rgb: &mut Array4<u8>,
    parallel: bool,
    idx_fn: F,
) where
    T: Copy + Sync + Send,
    F: Fn(T) -> usize + Sync + Send,
    M: MaskApplicator3D,
{
    let shape_y = arr.shape()[1];
    let shape_x = arr.shape()[2];
    if parallel {
        rgb.axis_iter_mut(numpy::ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(n, mut plane)| {
                for y in 0..shape_y {
                    for x in 0..shape_x {
                        let val = arr[[n, y, x]];
                        let idx = idx_fn(val);
                        let color = cmap[idx];
                        let color = masks.apply(color, n, y, x);
                        plane[[y, x, 0]] = color[0];
                        plane[[y, x, 1]] = color[1];
                        plane[[y, x, 2]] = color[2];
                    }
                }
            });
    } else {
        let shape_n = arr.shape()[0];
        for n in 0..shape_n {
            for y in 0..shape_y {
                for x in 0..shape_x {
                    let val = arr[[n, y, x]];
                    let idx = idx_fn(val);
                    let color = cmap[idx];
                    let color = masks.apply(color, n, y, x);
                    rgb[[n, y, x, 0]] = color[0];
                    rgb[[n, y, x, 1]] = color[1];
                    rgb[[n, y, x, 2]] = color[2];
                }
            }
        }
    }
}

/// Apply a colormap to a 3D array with optional mask overlay
/// Dispatches to monomorphized inner function based on mask presence
fn apply_colormap_3d<'a, T, F>(
    arr: ArrayView3<T>,
    cmap: &[[u8; 3]; 256],
    masks: Option<&'a [MaskConfig3D<'a>]>,
    rgb: &mut Array4<u8>,
    parallel: bool,
    idx_fn: F,
) where
    T: Copy + Sync + Send,
    F: Fn(T) -> usize + Sync + Send + Clone,
{
    match masks {
        Some(m) if !m.is_empty() => {
            apply_colormap_3d_inner(arr, cmap, &MaskSlice3D(m), rgb, parallel, idx_fn);
        }
        _ => {
            apply_colormap_3d_inner(arr, cmap, &NoMasks, rgb, parallel, idx_fn);
        }
    }
}

/// apply a colormap to a stack of 8 bit images
pub fn colorize_stack_8bit<'a>(
    config: ChannelConfigU83D<'a>,
    masks: Option<&'a [MaskConfig3D<'a>]>,
    parallel: bool,
) -> Array4<u8> {
    let mut rgb = stack_to_rgb(config.arr);

    if config.is_normalized() {
        apply_colormap_3d(config.arr, config.cmap, masks, &mut rgb, parallel, |v| {
            v as usize
        });
    } else {
        let [offset, scale] = offset_and_scale(config.limits);
        apply_colormap_3d(config.arr, config.cmap, masks, &mut rgb, parallel, |v| {
            as_idx(v, offset, scale)
        });
    }
    rgb
}

/// apply a colormap to a single 16 bit image, normalizing the intensity lookups on the fly
pub fn colorize_single_channel_16bit<'a>(
    config: ChannelConfigU162D<'a>,
    masks: Option<&'a [MaskConfig2D<'a>]>,
    parallel: bool,
) -> Array3<u8> {
    let mut rgb = img_to_rgb(config.arr);
    let [offset, scale] = offset_and_scale(config.limits);
    apply_colormap_2d(config.arr, config.cmap, masks, &mut rgb, parallel, |v| {
        as_idx(v, offset, scale)
    });
    rgb
}

/// apply a colormap to a stack of 16 bit images, normalizing the intensity lookups on the fly
pub fn colorize_stack_16bit<'a>(
    config: ChannelConfigU163D<'a>,
    masks: Option<&'a [MaskConfig3D<'a>]>,
    parallel: bool,
) -> Array4<u8> {
    let mut rgb = stack_to_rgb(config.arr);
    let [offset, scale] = offset_and_scale(config.limits);
    apply_colormap_3d(config.arr, config.cmap, masks, &mut rgb, parallel, |v| {
        as_idx(v, offset, scale)
    });
    rgb
}

/// Inner implementation for merge_2d_u8 with generic mask applicator
fn merge_2d_u8_inner<M: MaskApplicator2D>(
    configs: &[ChannelConfigU82D],
    blend_fn: blend::BlendFn,
    masks: &M,
    rgb: &mut Array3<u8>,
    parallel: bool,
) {
    let shape_y = configs[0].arr.shape()[0];
    let shape_x = configs[0].arr.shape()[1];

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
                        let px_color = blend_fn(&px_vals);
                        let px_color = masks.apply(px_color, i, j);
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
                    let px_color = blend_fn(&px_vals);
                    let px_color = masks.apply(px_color, i, j);
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
                        let px_color = masks.apply(px_color, i, j);
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
                    let px_color = masks.apply(px_color, i, j);
                    rgb[[i, j, 0]] = px_color[0];
                    rgb[[i, j, 1]] = px_color[1];
                    rgb[[i, j, 2]] = px_color[2];
                }
            }
        }
    }
}

/// Merge n 2d arrays together
pub fn merge_2d_u8<'a>(
    configs: Vec<ChannelConfigU82D<'a>>,
    blending: &str,
    masks: Option<&'a [MaskConfig2D<'a>]>,
    parallel: bool,
) -> Result<Array3<u8>, MergeError> {
    let first_config = &configs[0]; // we guarantee that all arrays have the same shape before calling
    let mut rgb = img_to_rgb(first_config.arr);
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    match masks {
        Some(m) if !m.is_empty() => {
            merge_2d_u8_inner(&configs, blend_fn, &MaskSlice2D(m), &mut rgb, parallel);
        }
        _ => {
            merge_2d_u8_inner(&configs, blend_fn, &NoMasks, &mut rgb, parallel);
        }
    }
    Ok(rgb)
}

/// Inner implementation for merge_3d_u8 with generic mask applicator
fn merge_3d_u8_inner<M: MaskApplicator3D>(
    configs: &[ChannelConfigU83D],
    blend_fn: blend::BlendFn,
    masks: &M,
    rgb: &mut Array4<u8>,
    parallel: bool,
) {
    let shape_n = configs[0].arr.shape()[0];
    let shape_y = configs[0].arr.shape()[1];
    let shape_x = configs[0].arr.shape()[2];

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
                            let px_color = blend_fn(&px_vals);
                            let px_color = masks.apply(px_color, n, i, j);
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
                        let px_color = blend_fn(&px_vals);
                        let px_color = masks.apply(px_color, n, i, j);
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
                            let px_color = masks.apply(px_color, n, i, j);
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
                        let px_color = masks.apply(px_color, n, i, j);
                        rgb[[n, i, j, 0]] = px_color[0];
                        rgb[[n, i, j, 1]] = px_color[1];
                        rgb[[n, i, j, 2]] = px_color[2];
                    }
                }
            }
        }
    }
}

/// Merge n 3d arrays together
pub fn merge_3d_u8<'a>(
    configs: Vec<ChannelConfigU83D<'a>>,
    blending: &str,
    masks: Option<&'a [MaskConfig3D<'a>]>,
    parallel: bool,
) -> Result<Array4<u8>, MergeError> {
    let first_config = &configs[0]; // we guarantee that all arrays have the same shape before calling
    let mut rgb = stack_to_rgb(first_config.arr);
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    match masks {
        Some(m) if !m.is_empty() => {
            merge_3d_u8_inner(&configs, blend_fn, &MaskSlice3D(m), &mut rgb, parallel);
        }
        _ => {
            merge_3d_u8_inner(&configs, blend_fn, &NoMasks, &mut rgb, parallel);
        }
    }
    Ok(rgb)
}

/// Inner implementation for merge_2d_u16 with generic mask applicator
fn merge_2d_u16_inner<M: MaskApplicator2D>(
    configs: &[ChannelConfigU162D],
    blend_fn: blend::BlendFn,
    masks: &M,
    rgb: &mut Array3<u8>,
    parallel: bool,
) {
    let shape_y = configs[0].arr.shape()[0];
    let shape_x = configs[0].arr.shape()[1];

    // slow path - normalize on the fly (always needed for u16)
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
                    let px_color = masks.apply(px_color, i, j);
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
                let px_color = masks.apply(px_color, i, j);
                rgb[[i, j, 0]] = px_color[0];
                rgb[[i, j, 1]] = px_color[1];
                rgb[[i, j, 2]] = px_color[2];
            }
        }
    }
}

pub fn merge_2d_u16<'a>(
    configs: Vec<ChannelConfigU162D<'a>>,
    blending: &str,
    masks: Option<&'a [MaskConfig2D<'a>]>,
    parallel: bool,
) -> Result<Array3<u8>, MergeError> {
    let first_config = &configs[0]; // we guarantee that all arrays have the same shape before calling
    let mut rgb = img_to_rgb(first_config.arr);
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    match masks {
        Some(m) if !m.is_empty() => {
            merge_2d_u16_inner(&configs, blend_fn, &MaskSlice2D(m), &mut rgb, parallel);
        }
        _ => {
            merge_2d_u16_inner(&configs, blend_fn, &NoMasks, &mut rgb, parallel);
        }
    }
    Ok(rgb)
}

/// Inner implementation for merge_3d_u16 with generic mask applicator
fn merge_3d_u16_inner<M: MaskApplicator3D>(
    configs: &[ChannelConfigU163D],
    blend_fn: blend::BlendFn,
    masks: &M,
    rgb: &mut Array4<u8>,
    parallel: bool,
) {
    let shape_n = configs[0].arr.shape()[0];
    let shape_y = configs[0].arr.shape()[1];
    let shape_x = configs[0].arr.shape()[2];

    // slow path - normalize on the fly (always needed for u16)
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
                        let px_color = masks.apply(px_color, n, i, j);
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
                    let px_color = masks.apply(px_color, n, i, j);
                    rgb[[n, i, j, 0]] = px_color[0];
                    rgb[[n, i, j, 1]] = px_color[1];
                    rgb[[n, i, j, 2]] = px_color[2];
                }
            }
        }
    }
}

pub fn merge_3d_u16<'a>(
    configs: Vec<ChannelConfigU163D<'a>>,
    blending: &str,
    masks: Option<&'a [MaskConfig3D<'a>]>,
    parallel: bool,
) -> Result<Array4<u8>, MergeError> {
    let first_config = &configs[0]; // we guarantee that all arrays have the same shape before calling
    let mut rgb = stack_to_rgb(first_config.arr);
    let blend_fn: blend::BlendFn = match blending {
        "max" => blend::max_blending,
        "sum" => blend::sum_blending,
        "min" => blend::min_blending,
        "mean" => blend::mean_blending,
        _ => return Err(MergeError::InvalidBlendingMode(blending.to_string())),
    };

    match masks {
        Some(m) if !m.is_empty() => {
            merge_3d_u16_inner(&configs, blend_fn, &MaskSlice3D(m), &mut rgb, parallel);
        }
        _ => {
            merge_3d_u16_inner(&configs, blend_fn, &NoMasks, &mut rgb, parallel);
        }
    }
    Ok(rgb)
}
