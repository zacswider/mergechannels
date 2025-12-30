from ._internal import (
    apply_color_map,
    get_cmap_array,
    merge,
)
from .mergechannels import (  # type: ignore
    dispatch_multi_channel,
    dispatch_single_channel,
)

COLORMAPS = _COLORMAPS


__all__ = [
    'dispatch_single_channel',
    'dispatch_multi_channel',
    'merge',
    'apply_color_map',
    'COLORMAPS',
]
