from typing import Sequence, Literal

import numpy as np

from luts import COLORMAPS
from _blending import BLENDING_OPTIONS

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

def apply_colors_and_merge_3c(
    arr1: np.ndarray,
    arr2: np.ndarray,
    arr3: np.ndarray,
    cmap1_name: COLORMAPS,
    cmap2_name: COLORMAPS,
    cmap3_name: COLORMAPS,
    blending: BLENDING_OPTIONS,
) -> np.ndarray: ...

def apply_colors_and_merge_4c(
    arr1: np.ndarray,
    arr2: np.ndarray,
    arr3: np.ndarray,
    arr4: np.ndarray,
    cmap1_name: COLORMAPS,
    cmap2_name: COLORMAPS,
    cmap3_name: COLORMAPS,
    cmap4_name: COLORMAPS,
    blending: BLENDING_OPTIONS,
) -> np.ndarray: ...
