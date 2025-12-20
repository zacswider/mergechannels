# Import Rust functions
from typing import Literal  # noqa: F401

from ._internal import apply_color_map, merge
from ._luts import COLORMAPS as _COLORMAPS
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
