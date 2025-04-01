from typing import Sequence

import numpy as np

from mergechannels import (
	apply_color_map,
	apply_colors_and_merge_2c,
	apply_colors_and_merge_3c,
	apply_colors_and_merge_4c,
)
from ._luts import COLORMAPS
from ._blending import BLENDING_OPTIONS

def merge(
	arrs: Sequence[np.ndarray],
	colors: Sequence[COLORMAPS] = (),
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
			return apply_color_map(arr=arrs[0], cmap_name=colors[0])
		case 2:
			return apply_colors_and_merge_2c(
				arr1=arrs[0],
				arr2=arrs[1],
				cmap1_name=colors[0],
				cmap2_name=colors[1],
				blending=blending,
			)
		case 3:
			return apply_colors_and_merge_3c(
				arr1=arrs[0],
				arr2=arrs[1],
				arr3=arrs[2],
				cmap1_name=colors[0],
				cmap2_name=colors[1],
				cmap3_name=colors[2],
				blending=blending,
			)
		case 4:
			return apply_colors_and_merge_4c(
				arr1=arrs[0],
				arr2=arrs[1],
				arr3=arrs[2],
				arr4=arrs[3],
				cmap1_name=colors[0],
				cmap2_name=colors[1],
				cmap3_name=colors[2],
				cmap4_name=colors[3],
				blending=blending,
			)
		case _:
			raise ValueError(f'Blending is only supporter for up to 4 channels, got {n_arrs} channels')










