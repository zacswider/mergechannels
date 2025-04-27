# Import Rust functions
from .mergechannels import (  # type: ignore
	apply_color_map,
	apply_colors_and_merge_nc,
	test_dynamic_arrays,
)

from ._internal import merge

__all__ = [
	'apply_color_map',
	'apply_colors_and_merge_nc',
	'test_dynamic_arrays',
	'merge',
]
