from typing import Sequence

import numpy as np

from mergechannels import (
	apply_color_map,
	apply_colors_and_merge_nc,
)
from ._luts import COLORMAPS
from ._blending import BLENDING_OPTIONS

def merge(
	arrs: Sequence[np.ndarray],
	colors: Sequence[COLORMAPS],
	blending: BLENDING_OPTIONS = 'max',
) -> np.ndarray:
	'''
	apply cmaps to arrays and blend the colors
	'''
	# region validation
	n_arrs = len(arrs)
	n_colors = len(colors)
	if not n_arrs == n_colors and n_arrs > 0:
		raise ValueError(
			'Expected an equal number of arrays to colorize and colormap names to apply. '
			f'Got {n_arrs} arrays and {n_colors} colors'
		)
	arr_shapes = [arr.shape for arr in arrs]
	if not len(set(arr_shapes)) == 1:
		raise ValueError(
			f'Expected every array to have the same shape, got {arr_shapes}'
		)
	if not len(arr_shapes[0]) == 2:
		raise ValueError(
			f'Expected every array to be 2D, got {arr_shapes[0]}'
		)
	arr_dtypes = [arr.dtype for arr in arrs]
	if not len(set(arr_dtypes)) == 1:
		raise ValueError(
			f'Expected every array to have the same dtype, got {arr_dtypes}'
		)
	# endregion

	if n_arrs == 1:
		return apply_color_map(arr=arrs[0], cmap_name=colors[0])
	else:
		return apply_colors_and_merge_nc(arrs, colors, blending)
