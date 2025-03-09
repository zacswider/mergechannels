import numpy as np

from .luts import COLORMAPS

def create_rgb_from_arr(x: np.ndarray) -> np.ndarray: ...

def apply_color_map(x: np.ndarray, cmap: COLORMAPS) -> np.ndarray: ...