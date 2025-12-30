from ._internal import (
    apply_color_map,
    get_cmap_array,
    merge,
)
from ._luts import COLORMAPS
from .mergechannels import (
    dispatch_multi_channel,
    dispatch_single_channel,
)

__all__ = [
    'dispatch_single_channel',
    'dispatch_multi_channel',
    'merge',
    'apply_color_map',
    'get_cmap_array',
    'COLORMAPS',
]
