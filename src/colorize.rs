mod cmaps;
use cmaps::BETTERBLUE;
use numpy::ndarray::{Array, Array3, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::{Bound, Python};

// Create a (y, x, 3) array with ones
fn create_rgb_from_arr(x: ArrayView2<u8>) -> Array3<u8> {
    Array::ones((x.shape()[0], x.shape()[1], 3))
}

fn apply_color_map(x: ArrayView2<u8>, cmap: &[[u8; 3]; 256]) -> Array3<u8> {
    let mut rgb = create_rgb_from_arr(x);
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            let value = x[[i, j]];
            let idx = value as usize;
            let color = cmap[idx];
            rgb[[i, j, 0]] = color[0];
            rgb[[i, j, 1]] = color[1];
            rgb[[i, j, 2]] = color[2];
        }
    }
    rgb
}

fn load_cmap(cmap_name: &str) -> &[[u8; 3]; 256] {
    match cmap_name {
        "better_blue" => &BETTERBLUE,
        _ => panic!("Invalid colormap name"),
    }
}

#[pyfunction]
#[pyo3(name = "create_rgb_from_arr")]
pub fn create_rgb_from_arr_py<'py>(
    py: Python<'py>,
    x: &Bound<PyArray2<u8>>,
) -> Bound<'py, PyArray3<u8>> {
    let x = unsafe { x.as_array() };
    let z = create_rgb_from_arr(x);
    z.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "apply_color_map")]
pub fn apply_color_map_py<'py>(
    py: Python<'py>,
    x: &Bound<PyArray2<u8>>,
    cmap_name: &str,
) -> Bound<'py, PyArray3<u8>> {
    let x = unsafe { x.as_array() };
    let rgb = apply_color_map(x, &load_cmap(cmap_name));
    rgb.into_pyarray(py)
}
