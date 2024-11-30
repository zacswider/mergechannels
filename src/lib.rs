use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
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

#[pymodule]
fn mergechannels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(print_array_size, m)?)?;
    m.add_function(wrap_pyfunction!(make_rgb, m)?)?;
    Ok(())
}
// use ndarray::{prelude::*, stack};
// use std::fs::File;
// use std::io::{self, BufRead};

// // fn type_name_of<T>(_: &T) -> &'static str {
// //     std::any::type_name::<T>()
// // }

// fn main() -> io::Result<()> {
//     let path = "/Applications/Fiji.app/luts/sepia.lut";

//     if let Ok(file) = File::open(&path) {
//         let reader = io::BufReader::new(file);

//         let mut lut: Array2<u8> = Array::zeros((3, 256));

//         for (index, line) in reader.lines().enumerate() {
//             let line = line?;
//             let words: Vec<&str> = line.split(' ').collect();
//             let nums: Vec<u8> = words.iter().map(|x| x.parse::<u8>().unwrap()).collect();

//             for (ch, lut_val) in nums.iter().enumerate() {
//                 lut[[ch, index]] = *lut_val;
//             }
//         }

//         println!("shape is {:?}", lut.shape())
//     } else {
//         println!("Failed to open the file");
//     }

//     Ok(())
// }
