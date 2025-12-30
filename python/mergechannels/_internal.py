from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ._blending import BLENDING_OPTIONS
from ._luts import COLORMAPS
from .mergechannels import (  # type: ignore
    dispatch_multi_channel,
    dispatch_single_channel,
)

if TYPE_CHECKING:
    from cmap import Colormap as CmapColormap
    from matplotlib.colors import Colormap as MatplotlibColormap
    from nptyping import (
        NDArray,
        Shape,
        UInt8,
    )


def _parse_cmap_arguments(
    color: Union[
        COLORMAPS,
        NDArray[Shape['3, 255'], UInt8],
        MatplotlibColormap,
        CmapColormap,
    ],
) -> Tuple[Union[COLORMAPS, None], Union[NDArray[Shape['3, 256'], UInt8], None]]:
    """
    Parse the color argument and return the corresponding cmap name and cmap values

    Args:
            color: a user-specified argument which may be the name of mergechannels colormap,
            a ndarray lookup table, and matplotlib colormap, or a cmap colormap.

    Returns:
            A tuple specifying the corresponding mergechannels colormap name (or None if N/A),
            and an array of the lookup table (or None if N/A)
    """
    if isinstance(color, str):
        return color, None
    else:
        try:  # try to convert from a matplotlib colormap
            if not color._isinit:  # type: ignore
                color._init()  # type: ignore
            cmap_values = (color._lut[: color.N, :3] * 255).astype('uint8')  # type: ignore
        except AttributeError:  # try to convert from a cmaps ColorMap
            try:
                cmap_values = (np.asarray(color.lut()[:, :3]) * 255).astype('uint8')  # type: ignore
            except AttributeError:  # must be a list of lists or an array castable to u8 (256, 3)
                cmap_values = np.asarray(color).astype('uint8')  # type: ignore

    if not (
        isinstance(cmap_values, np.ndarray)
        and cmap_values.shape == (256, 3)
        and cmap_values.dtype == np.uint8
    ):
        raise ValueError(
            'Expected a matplotlib colormap, a cmaps colormap, or an object directly castable to '
            f'an 8-bit array of shape (256, 3), got {type(cmap_values)}: {color}'
        )
    return None, cmap_values


def apply_color_map(
    arr: np.ndarray,
    color: Union[
        COLORMAPS,
        NDArray[Shape['3, 255'], UInt8],
        MatplotlibColormap,
        CmapColormap,
    ],
    percentiles: Union[tuple[float, float], None] = None,
    saturation_limits: Union[tuple[float, float], None] = None,
    parallel: bool = True,
) -> np.ndarray:
    """
    apply a color map to an array
    """
    if saturation_limits is None:
        if percentiles is None:
            percentiles = (1.1, 99.9)
        low, high = np.percentile(arr, percentiles)
        saturation_limits = (low, high)

    cmap_name, cmap_values = _parse_cmap_arguments(color)

    return dispatch_single_channel(
        array_reference=arr,
        cmap_name=cmap_name,
        cmap_values=cmap_values,
        limits=saturation_limits,
        parallel=parallel,
    )


def merge(
    arrs: Sequence[np.ndarray],
    colors: Sequence[COLORMAPS],
    blending: BLENDING_OPTIONS = 'max',
    percentiles: Sequence[tuple[float, float]] | None = None,
    saturation_limits: Sequence[tuple[float, float]] | None = None,
    parallel: bool = True,
) -> np.ndarray:
    """
    apply cmaps to arrays and blend the colors
    """
    cmap_names, cmap_values = zip(*[_parse_cmap_arguments(color) for color in colors])
    if saturation_limits is None:
        if percentiles is None:
            percentiles = [(1.1, 99.9)] * len(arrs)
        saturation_limits = tuple(
            np.percentile(arr, ch_percentiles)
            for arr, ch_percentiles in zip(arrs, percentiles)  # type: ignore
        )
    return dispatch_multi_channel(
        array_references=arrs,
        cmap_names=cmap_names,
        cmap_values=cmap_values,
        blending=blending,
        limits=saturation_limits,  # type: ignore
        parallel=parallel,
    )
