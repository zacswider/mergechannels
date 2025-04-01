# Import Rust functions
from .mergechannels import (  # type: ignore
	apply_color_map,
	apply_colors_and_merge_2c,
	apply_colors_and_merge_3c,
	apply_colors_and_merge_4c,
)  

from ._internal import merge

__all__ = [
	"apply_color_map",
	"apply_colors_and_merge_2c",
	"apply_colors_and_merge_3c",
	"apply_colors_and_merge_4c",
	"merge",
]
