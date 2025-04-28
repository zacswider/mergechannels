use crate::cmaps;
use crate::colorize;
use numpy::ndarray::ArrayView2;
use numpy::npyffi::{npy_uint32, npy_uint8};
use numpy::{PyArrayDyn, PyUntypedArray, PyUntypedArrayMethods, PyReadonlyArray, Element, PyReadonlyArrayDyn, IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{Bound, Python};

#[pyfunction]
#[pyo3(name = "dispatch_single_channel")]
pub fn dispatch_single_channel_py(
    py: Python<'_>,
    array_reference: &Bound<'_, PyAny>,
    low: u8,
    high: u8,
    ndim: usize,
    dtype: &str,
    cmap_name: &str,
    // ) -> PyResult<(Bound<'py, PyArrayDyn<u8>)> {
) -> PyResult<(())> {
    match dtype {
        "uint8" => {
            println!("doing uint8");
            match ndim {
                2 => {
                    println!("doing 2D");
                    let arr = array_reference.extract::<PyReadonlyArray2<u8>>()?;
                },
                3 => {
                    println!("doing 3D");
                    let arr = array_reference.extract::<PyReadonlyArray3<u8>>()?;
                },
                _ => panic!("Recieved unsupported number of dimensions {:?}", ndim),
            }
        },
        "uint16" => {
            println!("doing uint16");
            match ndim {
                2 => {
                    println!("doing 2D");
                    let arr = array_reference.extract::<PyReadonlyArray2<u16>>()?;
                },
                3 => {
                    println!("doing 3D");
                    let arr = array_reference.extract::<PyReadonlyArray3<u16>>()?;
                },
                _ => panic!("Recieved unsupported number of dimensions {:?}", ndim),
            }
        },
        _ => panic!("Received unsupported dtype: {:?}", dtype),
    }
    Ok(())
}

// #[pyfunction]
// #[pyo3(name = "apply_colors_and_merge_nc")]
// pub fn apply_colors_and_merge_nc_py<'py>(
//     py: Python<'py>,
//     py_arrs: Vec<PyReadonlyArray2<'py, u8>>,
//     cmap_names: Vec<String>,
//     blending: &str,
// ) -> Bound<'py, PyArray3<u8>> {
//     let arrs: Vec<ArrayView2<u8>> = py_arrs.iter().map(|py_arr| py_arr.as_array()).collect();
//     let cmaps: Vec<&[[u8; 3]; 256]> = cmap_names
//         .iter()
//         .map(|name| cmaps::load_cmap(name))
//         .collect();
//     let rgb = colorize::apply_colors_and_merge(arrs, cmaps, blending);
//     rgb.into_pyarray(py)
// }
