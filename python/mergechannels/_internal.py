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
	colors: Sequence[COLORMAPS] = (),
	saturation_limits: Sequence[tuple[float, float]] = (),
	blending: BLENDING_OPTIONS = 'max',
) -> np.ndarray:
	'''
	apply cmaps to arrays and blend the colors
	'''
	n_arrs = len(arrs)
	n_colors = len(colors)
	if n_arrs > 0 and n_colors == 0:
		arr_shapes = [arr.shape for arr in arrs]
		if not all([len(arr_shape) == 3 for arr_shape in arr_shapes]):
			raise ValueError(
				'Expected a sequence of pre-colorized arrays, '
				f'got {n_arrs} arrays of shapes {arr_shapes}'
			)
		# call merge rgb arrs here
		return np.zeros((3,3))
	if not n_arrs == n_colors:
		raise ValueError(
			'Expected an equal number of arrays to colorize and colormap names to apply. '
			f'Got {n_arrs} arrays and {n_colors} colors'
		)
	arr_shapes = [arr.shape for arr in arrs]
	if not len(set(arr_shapes)) == 1:
		raise ValueError(
			f'Expected every array to have the same shape, got {arr_shapes}'
		)
	# call apply and merge rgb arrs here
	match n_arrs:
		case 1:
			if len(saturation_limits) == 0:
				lowhigh = np.asarray([0.2, 99.8])
			else:
				lowhigh = np.asarray(saturation_limits)
			low, high = np.percentile(arrs[0], lowhigh)
			return apply_color_map(arr=arrs[0], cmap_name=colors[0], saturation_limits=(low, high))
		case _:
			if len(arrs) > len(saturation_limits):
				lowhigh = np.asarray([0.002, 0.998]) * 100
				saturation_limits = []
				for arr in arrs:
					low, high = np.percentile(arr, lowhigh)
					saturation_limits.append((low, high))
			return apply_colors_and_merge_nc(arrs, colors, blending, saturation_limits)










