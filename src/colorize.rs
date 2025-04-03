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
                        max_blending(&px_vals)
                    }
                    "sum" => {
                        sum_blending(&px_vals)
                    }
                    "min" => {
                        min_blending(&px_vals)
                    }
                    "mean" => {
                        mean_blending(&px_vals)
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

fn max_blending(px_vals: &Vec<[u8; 3]>) -> [u8; 3] {
    let mut r: u8 = 0;
    let mut g: u8 = 0;
    let mut b: u8 = 0;
    for px_val in px_vals {
        if px_val[0] > r {
            r = px_val[0]
        }
        if px_val[1] > g {
            g = px_val[1]
        }
        if px_val[2] > b {
            b = px_val[2]
        }
    }
    [r, g, b]
}

fn sum_blending(px_vals: &Vec<[u8; 3]>) -> [u8; 3] {
    let mut r: u8 = 0;
    let mut g: u8 = 0;
    let mut b: u8 = 0;
    for px_val in px_vals {
        r = r.saturating_add(px_val[0]);
        g = g.saturating_add(px_val[1]);
        b = b.saturating_add(px_val[2]);
    }
    [r, g, b]
}

fn min_blending(px_vals: &Vec<[u8; 3]>) -> [u8; 3] {
    let mut r: u8 = 255;
    let mut g: u8 = 255;
    let mut b: u8 = 255;
    for px_val in px_vals {
        if px_val[0] < r {
            r = px_val[0]
        }
        if px_val[1] < g {
            g = px_val[1]
        }
        if px_val[2] < b {
            b = px_val[2]
        }
    }
    [r, g, b]
}

fn mean_blending(px_vals: &Vec<[u8; 3]>) -> [u8; 3] {
    let mut r: u16 = 0;
    let mut g: u16 = 0;
    let mut b: u16 = 0;
    for px_val in px_vals {
        r = r.saturating_add(px_val[0] as u16);
        g = g.saturating_add(px_val[1] as u16);
        b = b.saturating_add(px_val[2] as u16);
    }
    let n_channels: u16 = px_vals.len() as u16;
    let r: u8 = (r / n_channels) as u8;
    let g: u8 = (g / n_channels) as u8;
    let b: u8 = (b / n_channels) as u8;
    [r, g, b]
}
