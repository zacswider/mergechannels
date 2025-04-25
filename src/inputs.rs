use numpy::{npyffi::PyArray_Descr, Element, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::PyResult;


pub struct ArrayContainer<'py, T: Element> {
    array_ref: PyReadonlyArrayDyn<'py, T>,
    ndim: usize,
    dtype_name: String,
}

impl<'py, T: Element> ArrayContainer<'py, T> {
    pub fn new(py_array: PyReadonlyArrayDyn<'py, T>) -> PyResult<Self> {
        let dimensions: usize = py_array.ndim();
        let dtype: &PyArray_Descr = py_array.dtype();
        let dtype_name: String = dtype.name()?.to_string();

        Ok(ArrayContainer {
            array_ref: py_array,
            ndim: dimensions,
            dtype_name
        })
    }
}
