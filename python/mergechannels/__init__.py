# Import Rust functions
from .mergechannels import (  # type: ignore
	dispatch_single_channel,
	dispatch_multi_channel,
)
from ._internal import merge, apply_color_map
from ._luts import COLORMAPS as _COLORMAPS
from typing import Literal  # noqa: F401

COLORMAPS = _COLORMAPS


__all__ = [
	'dispatch_single_channel',
	'dispatch_multi_channel',
	'merge',
	'apply_color_map',
	'COLORMAPS',
]
