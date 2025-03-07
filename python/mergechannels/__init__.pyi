from typing import Literal

import numpy as np

def create_rgb_from_arr(x: np.ndarray) -> np.ndarray: ...

def apply_color_map(x: np.ndarray, cmap: Literal["better_blue"]) -> np.ndarray: ...