from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from mergechannels._blending import BLENDING_OPTIONS
from mergechannels._luts import COLORMAPS

if TYPE_CHECKING:
    from cmap import Colormap as CmapColormap
    from matplotlib.colors import Colormap as MatplotlibColormap
    from matplotlib.colors import ListedColormap
    from nptyping import (
        NDArray,
        Shape,
        UInt8,
    )

Number = Union[int, float]

# Type alias for mask color specification
MaskColor = Union[str, Tuple[int, int, int], Sequence[int]]

def apply_color_map(
    arr: np.ndarray,
    color: Union[
        COLORMAPS,
        NDArray[Shape['256, 3'], UInt8],
        MatplotlibColormap,
        CmapColormap,
    ],
    percentiles: tuple[Number, Number] | None = None,
    saturation_limits: tuple[Number, Number] | None = None,
    masks: Sequence[np.ndarray] | np.ndarray | None = None,
    mask_colors: Sequence[MaskColor] | MaskColor | None = None,
    mask_alphas: Sequence[float] | float | None = None,
    parallel: bool = True,
) -> np.ndarray: ...
def merge(
    arrs: Sequence[np.ndarray],
    colors: Sequence[COLORMAPS],
    blending: BLENDING_OPTIONS = 'max',
    percentiles: Sequence[tuple[float, float]] | None = None,
    saturation_limits: Sequence[tuple[float, float]] | None = None,
    masks: Sequence[np.ndarray] | np.ndarray | None = None,
    mask_colors: Sequence[MaskColor] | MaskColor | None = None,
    mask_alphas: Sequence[float] | float | None = None,
    parallel: bool = True,
) -> np.ndarray: ...
def dispatch_single_channel(
    array_reference: np.ndarray,
    cmap_name: Union[COLORMAPS, None],
    cmap_values: Union[
        NDArray[Shape['256, 3'], UInt8],
        MatplotlibColormap,
        CmapColormap,
        None,
    ],
    limits: tuple[Number, Number] | None = None,
    parallel: bool = False,
    mask_arrays: list[np.ndarray] | None = None,
    mask_colors: list[Tuple[int, int, int]] | None = None,
    mask_alphas: list[float] | None = None,
) -> np.ndarray: ...
def dispatch_multi_channel(
    array_references: Sequence[np.ndarray],
    cmap_names: Sequence[Union[COLORMAPS, None]],
    cmap_values: Sequence[
        Union[
            NDArray[Shape['256, 3'], UInt8],
            MatplotlibColormap,
            CmapColormap,
            None,
        ]
    ],
    blending: Literal[BLENDING_OPTIONS],
    limits: Sequence[tuple[Number, Number]] | None = None,
    parallel: bool = False,
    mask_arrays: list[np.ndarray] | None = None,
    mask_colors: list[Tuple[int, int, int]] | None = None,
    mask_alphas: list[float] | None = None,
) -> np.ndarray: ...
def get_cmap_array(name: COLORMAPS) -> NDArray[Shape['256, 3'], UInt8]: ...
def get_mpl_cmap(name: COLORMAPS) -> ListedColormap: ...
