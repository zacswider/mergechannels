from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence, Union

import numpy as np

from mergechannels._blending import BLENDING_OPTIONS
from mergechannels._luts import COLORMAPS

if TYPE_CHECKING:
    from cmap import Colormap as CmapColormap
    from matplotlib.colors import Colormap as MatplotlibColormap
    from nptyping import (
        NDArray,
        Shape,
        UInt8,
    )

Number = Union[int, float]

def apply_color_map(
    arr: np.ndarray,
    color: Union[
        COLORMAPS,
        NDArray[Shape['3, 255'], UInt8],
        MatplotlibColormap,
        CmapColormap,
    ],
    percentiles: tuple[Number, Number] | None = None,
    saturation_limits: tuple[Number, Number] | None = None,
    parallel: bool = True,
) -> np.ndarray: ...
def merge(
    arrs: Sequence[np.ndarray],
    colors: Sequence[COLORMAPS],
    blending: BLENDING_OPTIONS = 'max',
    percentiles: Sequence[tuple[float, float]] | None = None,
    saturation_limits: Sequence[tuple[float, float]] | None = None,
    parallel: bool = True,
) -> np.ndarray: ...
def dispatch_single_channel(
    array_reference: np.ndarray,
    cmap_name: Union[COLORMAPS, None],
    cmap_values: Union[
        NDArray[Shape['3, 255'], UInt8],
        MatplotlibColormap,
        CmapColormap,
        None,
    ],
    limits: tuple[Number, Number] | None = None,
    parallel: bool = False,
) -> np.ndarray: ...
def dispatch_multi_channel(
    array_references: Sequence[np.ndarray],
    cmap_names: Sequence[Union[COLORMAPS, None]],
    cmap_values: Sequence[
        Union[
            NDArray[Shape['3, 255'], UInt8],
            MatplotlibColormap,
            CmapColormap,
            None,
        ]
    ],
    blending: Literal[BLENDING_OPTIONS],
    limits: Sequence[tuple[Number, Number]] | None = None,
    parallel: bool = False,
) -> np.ndarray: ...
