from typing import Sequence, Literal

import numpy as np

from luts import COLORMAPS
from _internal import BLENDING_OPTIONS

def merge(
	arrs: Sequence[np.ndarray],
	colors: Sequence[COLORMAPS] = (),
	blending: Literal['sum', 'max', 'min'] = 'max',
) -> np.ndarray:
    ...

def apply_color_map(arr: np.ndarray, cmap_name: COLORMAPS) -> np.ndarray: ...

def apply_colors_and_merge_2c(
    arr1: np.ndarray,
    arr2: np.ndarray,
    cmap1_name: COLORMAPS,
    cmap2_name: COLORMAPS,
    blending: BLENDING_OPTIONS,
) -> np.ndarray: ...

