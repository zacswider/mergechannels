mod cmaps;
use cmaps::CMAPS;
use numpy::ndarray::{Array, Array3, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::{Bound, Python};

// Create a (y, x, 3) array with ones
fn rgb_with_arr_shape(x: ArrayView2<u8>) -> Array3<u8> {
    Array::ones((x.shape()[0], x.shape()[1], 3))
}

fn apply_color_map(arr: ArrayView2<u8>, cmap: &[[u8; 3]; 256]) -> Array3<u8> {
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

fn load_cmap(cmap_name: &str) -> &[[u8; 3]; 256] {
    CMAPS
        .get(cmap_name)
        .unwrap_or_else(|| panic!("Invalid colormap name: {}", cmap_name))
}

#[pyfunction]
#[pyo3(name = "apply_color_map")]
pub fn apply_color_map_py<'py>(
    py: Python<'py>,
    x: &Bound<PyArray2<u8>>,
    cmap_name: &str,
) -> Bound<'py, PyArray3<u8>> {
    let x = unsafe { x.as_array() };
    let cmap = load_cmap(cmap_name);
    let rgb = apply_color_map(x, cmap);
    rgb.into_pyarray(py)
}
