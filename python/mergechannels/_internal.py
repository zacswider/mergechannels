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
from .mergechannels import (
    get_cmap_array as _get_cmap_array,  # just aliasing this to inject a docstring
)

if TYPE_CHECKING:
    from cmap import Colormap as CmapColormap
    from matplotlib.colors import Colormap as MatplotlibColormap
    from matplotlib.colors import ListedColormap
    from nptyping import (
        NDArray,
        Shape,
        UInt8,
    )


def _parse_cmap_arguments(
    color: Union[
        COLORMAPS,
        NDArray[Shape['256, 3'], UInt8],
        MatplotlibColormap,
        CmapColormap,
    ],
) -> Tuple[Union[COLORMAPS, None], Union[NDArray[Shape['256, 3'], UInt8], None]]:
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
        NDArray[Shape['256, 3'], UInt8],
        MatplotlibColormap,
        CmapColormap,
    ],
    percentiles: Union[tuple[float, float], None] = None,
    saturation_limits: Union[tuple[float, float], None] = None,
    parallel: bool = True,
) -> np.ndarray:
    """
    Apply a colormap to a grayscale array.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (H, W) or (Z, H, W) with dtype uint8 or uint16.
    color : COLORMAPS or NDArray or MatplotlibColormap or CmapColormap
        The colormap to apply. Can be:
        - A built-in colormap name (see mergechannels.COLORMAPS)
        - A (256, 3) uint8 numpy array
        - A matplotlib Colormap object
        - A cmap Colormap object
    percentiles : tuple[float, float] | None, optional
        Percentile values (low, high) for auto-scaling intensity. Ignored if saturation_limits is
        provided. Default is (1.1, 99.9).
    saturation_limits : tuple[float, float] | None, optional
        Explicit intensity limits (low, high) to set the black and white points.
    parallel : bool, optional
        Whether to use a Rayon threadpool on the Rust side for parallel processing. Default is True.

    Returns
    -------
    np.ndarray
        RGB array with shape (..., 3) and dtype uint8.

    Raises
    ------
    ValueError
        If the colormap name is not found or color format is invalid.

    Examples
    --------
    >>> import mergechannels as mc
    >>> import numpy as np
    >>> arr = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    >>> rgb = mc.apply_color_map(arr, 'betterBlue', saturation_limits=(0, 255))
    >>> rgb.shape
    (512, 512, 3)
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
    Apply colormaps to multiple arrays and blend them into a single RGB image.

    Parameters
    ----------
    arrs : Sequence[np.ndarray]
        Sequence of input arrays, each with shape (H, W) or (Z, H, W).
        All arrays must have the same shape and dtype (uint8 or uint16).
    colors : Sequence[COLORMAPS]
        Sequence of colormap names or colormap objects, one per input array. Can be built-in names
        (see mergechannels.COLORMAPS), (256, 3) uint8 arrays, matplotlib Colormap objects, or cmap
        Colormap objects.
    blending : BLENDING_OPTIONS, optional
        Blending mode for combining colored channels. One of:
        - 'max': Maximum intensity projection (default)
        - 'sum': Additive blending (clamped to 255)
        - 'min': Minimum intensity projection
        - 'mean': Average of all channels
    percentiles : Sequence[tuple[float, float]] | None, optional
        Per-channel percentile values (low, high) for auto-scaling. Ignored if saturation_limits is
        provided. Default is (1.1, 99.9) for each.
    saturation_limits : Sequence[tuple[float, float]] | None, optional
        Per-channel explicit intensity limits (low, high) for scaling.
    parallel : bool, optional
        Whether to use a Rayon threadpool on the Rust side for parallel processing. Default is True.

    Returns
    -------
    np.ndarray
        Blended RGB array with shape (..., 3) and dtype uint8.

    Raises
    ------
    ValueError
        If a colormap name is not found or color format is invalid.

    Examples
    --------
    >>> import mergechannels as mc
    >>> import numpy as np
    >>> ch1 = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    >>> ch2 = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    >>> rgb = mc.merge(
    ...     [ch1, ch2],
    ...     ['betterBlue', 'betterOrange'],
    ...     blending='max',
    ...     saturation_limits=[(0, 255), (0, 255)],
    ... )
    >>> rgb.shape
    (512, 512, 3)
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


def get_cmap_array(name: COLORMAPS) -> np.ndarray:
    """
    Get the RGB values for a built-in colormap.

    Parameters
    ----------
    name : COLORMAPS
        The name of the colormap to retrieve. Use mergechannels.COLORMAPS
        to see available colormap names.

    Returns
    -------
    np.ndarray
        A (256, 3) uint8 array of RGB values, where each row represents
        the RGB color for that intensity level (0-255).

    Raises
    ------
    ValueError
        If the colormap name is not found.

    Examples
    --------
    >>> import mergechannels as mc
    >>> cmap = mc.get_cmap_array('betterBlue')
    >>> cmap.shape
    (256, 3)
    >>> cmap.dtype
    dtype('uint8')
    """
    return _get_cmap_array(name)


def get_mpl_cmap(name: COLORMAPS) -> ListedColormap:
    """
    Get a built-in colormap as a matplotlib ListedColormap.

    Parameters
    ----------
    name : COLORMAPS
        The name of the colormap to retrieve. Use mergechannels.COLORMAPS
        to see available colormap names.

    Returns
    -------
    matplotlib.colors.ListedColormap
        A matplotlib ListedColormap object that can be used with matplotlib
        plotting functions.

    Raises
    ------
    ImportError
        If matplotlib is not installed. Install it with:
        ``uv pip install matplotlib`` or
        ``uv pip install "mergechannels[matplotlib]>=0.5.5"``
    ValueError
        If the colormap name is not found.

    Examples
    --------
    >>> import mergechannels as mc
    >>> cmap = mc.get_mpl_cmap('betterBlue')
    >>> cmap.name
    'betterBlue'
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(data, cmap=cmap)  # doctest: +SKIP
    """
    try:
        from matplotlib.colors import ListedColormap
    except ImportError as e:
        raise ImportError(
            'matplotlib is required for get_mpl_cmap(). '
            'Install it with: uv pip install matplotlib '
            'or uv pip install "mergechannels[matplotlib]>=0.5.5"'
        ) from e

    cmap_array = get_cmap_array(name)
    colors = cmap_array / 255.0  # Convert from uint8 (0-255) to float (0-1) for matplotlib
    return ListedColormap(colors, name=name)
