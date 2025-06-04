from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Sequence, Union, overload
from nptyping import NDArray, Shape, UInt8
import numpy as np

from mergechannels._blending import BLENDING_OPTIONS
from mergechannels._luts import COLORMAPS


if TYPE_CHECKING:
    from matplotlib.colors import Colormap as MatplotlibColormap
    from cmap import Colormap as CmapColormap

Number = Union[int, float]

ColorLiteral = COLORMAPS
ColorType = Union[
    ColorLiteral,
    NDArray[Shape['3, 255'], UInt8],
    MatplotlibColormap,
    CmapColormap,
]

def apply_color_map(
    arr: np.ndarray,
    color: ColorType,
    percentiles: tuple[Number, Number] | None = None,
    saturation_limits: tuple[Number, Number] | None = None,
) -> np.ndarray:
    ...

@overload
def merge(
    arrs: Sequence[np.ndarray],
    colors: Sequence[ColorLiteral],
    blending: BLENDING_OPTIONS = 'max',
    percentiles: Sequence[tuple[float, float]] | None = None,
    saturation_limits: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    ...

@overload
def merge(
    arrs: Sequence[np.ndarray],
    colors: Sequence[
        NDArray[Shape['3, 255'], UInt8]
        | MatplotlibColormap
        | CmapColormap
    ],
    blending: BLENDING_OPTIONS = 'max',
    percentiles: Sequence[tuple[float, float]] | None = None,
    saturation_limits: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    ...

def merge(
    arrs: Sequence[np.ndarray],
    colors: Sequence[ColorType],
    blending: BLENDING_OPTIONS = 'max',
    percentiles: Sequence[tuple[float, float]] | None = None,
    saturation_limits: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    ...

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
) -> np.ndarray:
    ...

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
) -> np.ndarray:
    ...
