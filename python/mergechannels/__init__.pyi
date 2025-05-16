
from typing import Literal, Sequence
import numpy as np

from python.mergechannels._blending import BLENDING_OPTIONS
from python.mergechannels._luts import COLORMAPS


def merge(
    arrs: np.ndarray | Sequence[np.ndarray],
    colors: Sequence[COLORMAPS] = (),
    blending: Literal[BLENDING_OPTIONS] = 'max',
    saturation_limits: tuple[float, float] = (0.2, 99.8),
) -> np.ndarray:
    ...

def dispatch_single_channel(
    array_reference: np.ndarray,
    cmap_name: str,
    limits: tuple[float, float],
) -> np.ndarray:
    ...

def dispatch_multi_channel(
    array_references: Sequence[np.ndarray],
    cmap_names: Sequence[str],
    blending: Literal[BLENDING_OPTIONS],
    limits: Sequence[tuple[float, float]],
) -> np.ndarray:
    ...
