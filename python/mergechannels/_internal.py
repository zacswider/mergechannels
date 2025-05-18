from typing import Sequence

import numpy as np

from mergechannels import (
	dispatch_single_channel,
	dispatch_multi_channel,
)
from ._luts import COLORMAPS
from ._blending import BLENDING_OPTIONS

def merge(
	arrs: Sequence[np.ndarray],
	colors: Sequence[COLORMAPS],
	blending: BLENDING_OPTIONS = 'max',
    saturation_limits: tuple[float, float] = (0.011, 0.999),
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
	if len(arr_shapes[0]) not in (2, 3):
		raise ValueError(
			f'Expected every array to be 2D or 3D, got {arr_shapes[0]}'
		)
	arr_dtypes = [arr.dtype for arr in arrs]
	if not len(set(arr_dtypes)) == 1:
		raise ValueError(
			f'Expected every array to have the same dtype, got {arr_dtypes}'
		)
	# endregion
	if n_arrs == 1:
		if arrs[0].dtype == 'uint8':
			limits = (0, 255)
		else:
			low, high = np.percentile(arrs[0], np.array(saturation_limits) * 100)
			limits = (low, high)
		return dispatch_single_channel(
			array_reference=arrs[0],
			cmap_name=colors[0],
			limits=limits,
		)
	else:
		if all(arr.dtype == 'uint8' for arr in arrs):
			limits = (0, 255)
		else:
			limits = tuple(np.percentile(arr, np.array(saturation_limits) * 100) for arr in arrs)
		return dispatch_multi_channel(
			array_references=arrs,
			cmap_names=colors,
			blending=blending,
			limits=limits,  # type: ignore
		)
