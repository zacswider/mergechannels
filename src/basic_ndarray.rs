use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::{Bound, Python};

// example using immutable borrows producing a new array
fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    a * &x + &y
}

// example using a mutable borrow to modify an array in-place
fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
    x *= a;
}

// wrapper of `axpy`
#[pyfunction]
#[pyo3(name = "axpy")]
pub fn axpy_py<'py>(
    py: Python<'py>,
    a: f64,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = axpy(a, x, y);
    z.into_pyarray(py)
}

// wrapper of `mult`
#[pyfunction]
#[pyo3(name = "mult")]
pub fn mult_py<'py>(a: f64, x: &Bound<'py, PyArrayDyn<f64>>) {
    let x = unsafe { x.as_array_mut() };
    mult(a, x);
}
