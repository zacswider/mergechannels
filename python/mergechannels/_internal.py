from __future__ import annotations

import re
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

# Type alias for mask color specification
MaskColor = Union[COLORMAPS, Tuple[int, int, int], Sequence[int]]

# Default mask color (purple) and alpha
DEFAULT_MASK_COLOR: Tuple[int, int, int] = (128, 0, 128)
DEFAULT_MASK_ALPHA: float = 0.5


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


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert a hex color string to RGB tuple.

    Args:
        hex_color: Hex color string like '#FF00FF', 'FF00FF', '#f0f', or 'f0f'

    Returns:
        Tuple of (R, G, B) values in range 0-255

    Raises:
        ValueError: If the hex string is invalid
    """
    # Remove leading '#' if present
    hex_color = hex_color.lstrip('#')

    # Handle shorthand hex (e.g., 'f0f' -> 'ff00ff')
    if len(hex_color) == 3:
        hex_color = ''.join(c * 2 for c in hex_color)

    if len(hex_color) != 6:
        raise ValueError(
            f"Invalid hex color '{hex_color}': expected 3 or 6 hex digits"
            " (with optional '#' prefix)"
        )

    # Validate hex characters
    if not re.match(r'^[0-9a-fA-F]{6}$', hex_color):
        raise ValueError(f"Invalid hex color '{hex_color}': contains non-hexadecimal characters")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def _parse_mask_color(color: MaskColor | None) -> Tuple[int, int, int]:
    """
    Parse a mask color specification into an RGB tuple.

    Args:
        color: Can be:
            - None: returns default purple (128, 0, 128)
            - A colormap name (str): uses the color at index 255 of that colormap
            - A hex string: '#FF00FF', 'FF00FF', '#f0f', 'f0f'
            - An RGB tuple/sequence: (R, G, B) with values 0-255

    Returns:
        Tuple of (R, G, B) values in range 0-255

    Raises:
        ValueError: If the color specification is invalid
    """
    if color is None:
        return DEFAULT_MASK_COLOR

    if isinstance(color, str):
        # Check if it's a hex color (starts with # or is all hex digits)
        if color.startswith('#') or re.match(r'^[0-9a-fA-F]{3}$|^[0-9a-fA-F]{6}$', color):
            return _hex_to_rgb(color)

        # Otherwise, treat as a colormap name
        try:
            cmap_array = _get_cmap_array(color)
            # Use the color at index 255 (brightest value in the colormap)
            return tuple(cmap_array[255])  # type: ignore
        except ValueError as e:
            raise ValueError(
                f"Invalid mask color '{color}': not a valid hex color or colormap name. "
                f"Hex colors should be like '#FF00FF' or 'f0f'. "
                f'Available colormaps can be found in mergechannels.COLORMAPS.'
            ) from e

    # Must be a sequence of RGB values
    try:
        rgb = tuple(color)  # type: ignore
        if len(rgb) != 3:
            raise ValueError(f'Invalid mask color: expected 3 RGB values, got {len(rgb)}')
        r, g, b = rgb
        # Validate range
        for val, name in [(r, 'R'), (g, 'G'), (b, 'B')]:
            if not isinstance(val, (int, np.integer)):
                raise ValueError(
                    f'Invalid mask color: {name} value must be an integer, got {type(val).__name__}'
                )
            if not 0 <= val <= 255:
                raise ValueError(f'Invalid mask color: {name} value {val} out of range [0, 255]')
        return (int(r), int(g), int(b))
    except TypeError:
        raise ValueError(
            f'Invalid mask color type: expected str, tuple, or sequence, got {type(color).__name__}'
        )


def _validate_mask(
    mask: np.ndarray,
    expected_shape: tuple,
    mask_index: int | None = None,
) -> None:
    """
    Validate a mask array.

    Args:
        mask: The mask array to validate
        expected_shape: Expected shape of the mask (should match the data array shape)
        mask_index: Optional index for error messages when validating multiple masks

    Raises:
        TypeError: If mask is not a numpy array
        ValueError: If mask shape doesn't match or dtype is invalid
    """
    idx_str = f' at index {mask_index}' if mask_index is not None else ''

    if not isinstance(mask, np.ndarray):
        raise TypeError(f'Mask{idx_str} must be a numpy array, got {type(mask).__name__}')

    if mask.shape != expected_shape:
        raise ValueError(
            f'Mask{idx_str} shape {mask.shape} does not match array shape {expected_shape}'
        )

    if mask.dtype not in (np.bool_, np.int32):
        raise ValueError(f'Mask{idx_str} dtype must be bool or int32, got {mask.dtype}')


def _parse_mask_arguments(
    masks: Sequence[np.ndarray] | np.ndarray | None,
    mask_colors: Sequence[MaskColor] | MaskColor | None,
    mask_alphas: Sequence[float] | float | None,
    expected_shape: tuple,
) -> Tuple[list[np.ndarray] | None, list[Tuple[int, int, int]] | None, list[float] | None]:
    """
    Parse and validate mask arguments, handling single values and sequences.

    Args:
        masks: Single mask array, sequence of masks, or None
        mask_colors: Single color, sequence of colors, or None
        mask_alphas: Single alpha, sequence of alphas, or None
        expected_shape: Expected shape for all masks (should match data array shape)

    Returns:
        Tuple of (masks_list, colors_list, alphas_list) or (None, None, None) if no masks

    Raises:
        ValueError: If arguments are inconsistent or invalid
    """
    if masks is None:
        return None, None, None

    # Normalize masks to a list
    if isinstance(masks, np.ndarray):
        masks_list = [masks]
    else:
        masks_list = list(masks)

    if len(masks_list) == 0:
        return None, None, None

    # Validate all masks
    for i, mask in enumerate(masks_list):
        _validate_mask(mask, expected_shape, mask_index=i if len(masks_list) > 1 else None)

    n_masks = len(masks_list)

    # Parse colors
    if mask_colors is None:
        colors_list = [DEFAULT_MASK_COLOR] * n_masks
    elif isinstance(mask_colors, (str, tuple)) or (
        isinstance(mask_colors, Sequence)
        and len(mask_colors) == 3
        and isinstance(mask_colors[0], (int, np.integer))
    ):
        # Single color specification - apply to all masks
        parsed_color = _parse_mask_color(mask_colors)  # type: ignore
        colors_list = [parsed_color] * n_masks
    else:
        # Sequence of colors
        colors_list = [_parse_mask_color(c) for c in mask_colors]  # type: ignore
        if len(colors_list) != n_masks:
            raise ValueError(
                f'Number of mask colors ({len(colors_list)}) does not match '
                f'number of masks ({n_masks})'
            )

    # Parse alphas
    if mask_alphas is None:
        alphas_list = [DEFAULT_MASK_ALPHA] * n_masks
    elif isinstance(mask_alphas, (int, float)):
        # Single alpha - apply to all masks
        alpha = float(mask_alphas)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Mask alpha {alpha} out of range [0.0, 1.0]')
        alphas_list = [alpha] * n_masks
    else:
        # Sequence of alphas
        alphas_list = []
        for i, a in enumerate(mask_alphas):
            alpha = float(a)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f'Mask alpha at index {i} ({alpha}) out of range [0.0, 1.0]')
            alphas_list.append(alpha)
        if len(alphas_list) != n_masks:
            raise ValueError(
                f'Number of mask alphas ({len(alphas_list)}) does not match '
                f'number of masks ({n_masks})'
            )

    return masks_list, colors_list, alphas_list


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
    masks: Sequence[np.ndarray] | np.ndarray | None = None,
    mask_colors: Sequence[MaskColor] | MaskColor | None = None,
    mask_alphas: Sequence[float] | float | None = None,
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
    masks : Sequence[np.ndarray] | np.ndarray | None, optional
        Mask array(s) to overlay on the result. Each mask must have the same shape as the input
        array and dtype of bool or int32. For bool masks, True pixels are overlaid. For int32
        masks, any non-zero value is overlaid.
    mask_colors : Sequence[MaskColor] | MaskColor | None, optional
        Color(s) for the mask overlay. Can be:
        - A colormap name (uses the color at index 255)
        - A hex string ('#FF00FF', 'f0f')
        - An RGB tuple (R, G, B) with values 0-255
        If a single color is provided, it applies to all masks.
        Default is purple (128, 0, 128).
    mask_alphas : Sequence[float] | float | None, optional
        Alpha value(s) for mask blending (0.0-1.0). If a single value is provided, it applies
        to all masks. Default is 0.5.
    parallel : bool, optional
        Whether to use a Rayon threadpool on the Rust side for parallel processing. Default is True.

    Returns
    -------
    np.ndarray
        RGB array with shape (..., 3) and dtype uint8.

    Raises
    ------
    ValueError
        If the colormap name is not found, color format is invalid, or mask arguments are invalid.
    TypeError
        If masks are not numpy arrays.

    Examples
    --------
    >>> import mergechannels as mc
    >>> import numpy as np
    >>> arr = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    >>> rgb = mc.apply_color_map(arr, 'betterBlue', saturation_limits=(0, 255))
    >>> rgb.shape
    (512, 512, 3)

    With a mask overlay:

    >>> mask = arr > 200  # Highlight bright pixels
    >>> rgb = mc.apply_color_map(
    ...     arr, 'Grays', saturation_limits=(0, 255),
    ...     masks=[mask], mask_colors=['#FF0000'], mask_alphas=[0.5]
    ... )
    """
    if saturation_limits is None:
        if percentiles is None:
            percentiles = (1.1, 99.9)
        low, high = np.percentile(arr, percentiles)
        saturation_limits = (low, high)

    cmap_name, cmap_values = _parse_cmap_arguments(color)

    # Parse mask arguments
    masks_list, colors_list, alphas_list = _parse_mask_arguments(
        masks, mask_colors, mask_alphas, expected_shape=arr.shape
    )

    return dispatch_single_channel(
        array_reference=arr,
        cmap_name=cmap_name,
        cmap_values=cmap_values,
        limits=saturation_limits,
        parallel=parallel,
        mask_arrays=masks_list,
        mask_colors=colors_list,
        mask_alphas=alphas_list,
    )


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
    masks : Sequence[np.ndarray] | np.ndarray | None, optional
        Mask array(s) to overlay on the blended result. Each mask must have the same shape as the
        input arrays and dtype of bool or int32. For bool masks, True pixels are overlaid. For int32
        masks, any non-zero value is overlaid.
    mask_colors : Sequence[MaskColor] | MaskColor | None, optional
        Color(s) for the mask overlay. Can be:
        - A colormap name (uses the color at index 255)
        - A hex string ('#FF00FF', 'f0f')
        - An RGB tuple (R, G, B) with values 0-255
        If a single color is provided, it applies to all masks.
        Default is purple (128, 0, 128).
    mask_alphas : Sequence[float] | float | None, optional
        Alpha value(s) for mask blending (0.0-1.0). If a single value is provided, it applies
        to all masks. Default is 0.5.
    parallel : bool, optional
        Whether to use a Rayon threadpool on the Rust side for parallel processing. Default is True.

    Returns
    -------
    np.ndarray
        Blended RGB array with shape (..., 3) and dtype uint8.

    Raises
    ------
    ValueError
        If a colormap name is not found, color format is invalid, or mask arguments are invalid.
    TypeError
        If masks are not numpy arrays.

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

    With a mask overlay:

    >>> mask = ch1 > 200  # Highlight bright pixels from channel 1
    >>> rgb = mc.merge(
    ...     [ch1, ch2],
    ...     ['betterBlue', 'betterOrange'],
    ...     saturation_limits=[(0, 255), (0, 255)],
    ...     masks=[mask], mask_colors=[(255, 0, 0)], mask_alphas=[0.5]
    ... )
    """
    cmap_names, cmap_values = zip(*[_parse_cmap_arguments(color) for color in colors])
    if saturation_limits is None:
        if percentiles is None:
            percentiles = [(1.1, 99.9)] * len(arrs)
        saturation_limits = tuple(
            np.percentile(arr, ch_percentiles)
            for arr, ch_percentiles in zip(arrs, percentiles)  # type: ignore
        )

    # Get expected shape from first array
    expected_shape = arrs[0].shape
    for a in arrs[1:]:
        if not a.shape == expected_shape:
            raise ValueError(
                f'Expected all input arrays to have the same shape, {a.shape} != {expected_shape}'
            )

    # Parse mask arguments
    masks_list, colors_list, alphas_list = _parse_mask_arguments(
        masks, mask_colors, mask_alphas, expected_shape=expected_shape
    )

    return dispatch_multi_channel(
        array_references=arrs,
        cmap_names=cmap_names,
        cmap_values=cmap_values,
        blending=blending,
        limits=saturation_limits,  # type: ignore
        parallel=parallel,
        mask_arrays=masks_list,
        mask_colors=colors_list,
        mask_alphas=alphas_list,
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
