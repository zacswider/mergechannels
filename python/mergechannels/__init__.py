# Import Rust functions
from mergechannels import (
	apply_color_map,  # type: ignore
	apply_colors_and_merge_2c, # type: ignore
)  
# Import Python functions
from ._internal import merge

__all__ = [
	"apply_color_map",
	"apply_colors_and_merge_2c",
	"merge",
]
