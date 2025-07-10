use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::error::Error;

#[derive(Debug)]
pub enum DispatchError {
    UnsupportedDataType(String),
    UnsupportedNumberOfDimensions(usize),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::UnsupportedDataType(dtype) => {
                write!(f, "Received unsupported dtype: {dtype}")
            }
            DispatchError::UnsupportedNumberOfDimensions(ndim) => {
                write!(f, "Received unsupported number of dimensions: {ndim}")
            }
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<DispatchError> for PyErr {
    fn from(err: DispatchError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

#[derive(Debug)]
pub enum MergeError {
    InvalidBlendingMode(String),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::InvalidBlendingMode(mode) => {
                write!(
                    f,
                    "Invalid blending mode: `{mode}`. Valid modes are 'max', 'sum', 'min', and 'mean'."
                )
            }
        }
    }
}

impl Error for MergeError {}
