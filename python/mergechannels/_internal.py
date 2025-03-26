from typing import Sequence, Literal

import numpy as np

def merge(
	arrs: Sequence[np.ndarray],
	colors: Sequence[str] = (),
	blending: Literal['sum', 'max', 'min'] = 'max',
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
	if not n_arrs == n_colors:
		raise ValueError(
			'Expected an equal number of arrays to colorize and colormap names to apply. '
			f'Got {n_arrs} arrays and {n_colors} colors'
		)
	# call apply and merge rgb arrs here
	return np.zeros((3,3))
