from typing import Sequence, Literal

import numpy as np

from .luts import COLORMAPS


def merge(
	arrs: Sequence[np.ndarray],
	colors: Sequence[COLORMAPS] = (),
	blending: Literal['sum', 'max', 'min'] = 'max',
) -> np.ndarray:
    ...

def apply_color_map(x: np.ndarray, cmap: COLORMAPS) -> np.ndarray: ...

