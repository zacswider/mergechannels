use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};

fn type_name_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}

#[pyfunction]
fn print_array_size(arr: PyReadonlyArray2<f64>) {
    println!("Array type: {}", type_name_of(&arr));
    let shape = arr.shape();
    println!("Array shape: {:?}", shape);
}

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn make_rgb(arr: PyReadonlyArray2<f64>) {
    let mut rgb_dims = arr.shape().to_vec();
    rgb_dims.insert(0, 3);
    let rgb_arr = Array::<u8, _>::ones(rgb_dims);
    let rgb_min = rgb_arr.min().unwrap();
    let rgb_max = rgb_arr.max().unwrap();
    println!("min is {}", rgb_min);
    println!("max is {}", rgb_max);
}

#[pyfunction]
fn load_lut() -> Py<PyArray2<u8>> {
    let lut_path = "/Applications/Fiji.app/luts/sepia.lut";

    let file = File::open(&lut_path).expect("Failed to open lut path");
    let reader = io::BufReader::new(file);
    let mut lut: Array2<u8> = Array2::zeros((3, 256));

    for (index, line_result) in reader.lines().enumerate() {
        let line = line_result.expect("Failed to read line");
        let words: Vec<&str> = line.split_whitespace().collect();
        let nums: Vec<u8> = words.iter().map(|x| x.parse::<u8>().unwrap()).collect();

        for (ch, lut_val) in nums.iter().enumerate() {
            lut[[ch, index]] = *lut_val;
        }
    }

    let numpy_array = lut.into_pyarray_bound(py);
    numpy_array
}

// #[pyfunction]
// fn apply_lut() -> Array3<u8> {
//     let lut = load_lut();
//     let a2 = array![
//         [1, 2, 3, 4, 5],
//         [2, 3, 4, 5, 6],
//         [3, 4, 5, 6, 7],
//         [4, 5, 6, 7, 8]
//     ];
//     let data = a2
//         .as_slice()
//         .expect("Failed to create slice because array was not in the standard layout");
//     let res = lut.select(Axis(1), &data).into_owned();
//     // let mut rgb_shape = vec![3];
//     let rgb_shape = [3, a2.shape()[0], a2.shape()[1]];
//     // rgb_shape.extend(a2.shape());
//     let res = res
//         .into_shape_with_order(rgb_shape)
//         .expect("Failed to reshape");

//     res
// }

#[pymodule]
fn mergechannels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(print_array_size, m)?)?;
    m.add_function(wrap_pyfunction!(make_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(load_lut, m)?)?;
    // m.add_function(wrap_pyfunction!(apply_lut, m)?)?;
    Ok(())
}
