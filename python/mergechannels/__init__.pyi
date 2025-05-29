
from typing import Literal, Sequence
import numpy as np

from python.mergechannels._blending import BLENDING_OPTIONS
from python.mergechannels._luts import COLORMAPS


def apply_color_map(
	arr: np.ndarray,
	color: COLORMAPS,
	percentiles: tuple[float, float] | None = None,
	saturation_limits: tuple[float, float] | None = None,
) -> np.ndarray:
    ...

def merge(
	arrs: Sequence[np.ndarray],
	colors: Sequence[COLORMAPS],
	blending: BLENDING_OPTIONS = 'max',
	percentiles: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
	saturation_limits: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
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
