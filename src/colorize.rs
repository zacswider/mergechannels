use crate::blend;
use numpy::ndarray::{Array, Array3, ArrayView2};

// Create a (y, x, 3) array with ones
fn rgb_with_arr_shape(x: ArrayView2<u8>) -> Array3<u8> {
    Array::ones((x.shape()[0], x.shape()[1], 3))
}

pub fn apply_color_map(arr: ArrayView2<u8>, cmap: &[[u8; 3]; 256]) -> Array3<u8> {
    let mut rgb = rgb_with_arr_shape(arr);
    let shape_y = arr.shape()[0];
    let shape_x = arr.shape()[1];

    for i in 0..shape_y {
        for j in 0..shape_x {
            let idx = arr[[i, j]] as usize;
            let color = cmap[idx];
            rgb[[i, j, 0]] = color[0];
            rgb[[i, j, 1]] = color[1];
            rgb[[i, j, 2]] = color[2];
        }
    }
    rgb
}

pub fn apply_colors_and_merge(
    arrs: Vec<ArrayView2<u8>>,
    cmaps: Vec<&[[u8; 3]; 256]>,
    blending: &str,
) -> Array3<u8> {
    let first_arr = arrs[0]; // we guarantee that all arrays have the same shape before calling
    let shape_y = first_arr.shape()[0];
    let shape_x = first_arr.shape()[1];
    let mut rgb = rgb_with_arr_shape(first_arr);
    for i in 0..shape_y {
        for j in 0..shape_x {
            let mut px_vals: Vec<[u8; 3]> = Vec::with_capacity(arrs.len());
            for (arr, cmap) in arrs.iter().zip(cmaps.iter()) {
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
