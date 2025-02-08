import numpy as np
import mergechannels

def test_default_function() -> None:
    res = mergechannels.sum_as_string(10, 20)
    assert res == '30'

def test_basic_multiply() -> None:
    a = np.ones(shape=(3,3))
    mergechannels.mult(2.0, a)
    assert np.allclose(a, np.ones((3,3)) * 2)

def test_axpy() -> None:
    x = np.ones(shape=(3,3))
    y = np.ones(shape=(3,3)) * 2
    res = mergechannels.axpy(2.0, x, y)
    assert np.allclose(res, np.ones((3,3)) * 4)
