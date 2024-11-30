import numpy as np
from skimage import data

import mergechannels

def test_default_function() -> None:
    res = mergechannels.sum_as_string(10, 20)
    assert res == '30'

def test_call_print_array_size() -> None:
    mergechannels.print_array_size(
        arr=np.ones(shape=(3, 3)),
    )

def test_make_rgb() -> None:
    mergechannels.make_rgb(
        arr=np.ones(shape=(3,3)),
    )
