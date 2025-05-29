from typing import (
	Sequence,
	Union,
)

import numpy as np

from mergechannels import (
	dispatch_single_channel,
	dispatch_multi_channel,
)
from ._luts import COLORMAPS
from ._blending import BLENDING_OPTIONS

def apply_color_map(
	arr: np.ndarray,
	color: COLORMAPS,
	percentiles: Union[tuple[float, float], None] = None,
	saturation_limits: Union[tuple[float, float], None] = None,
) -> np.ndarray:
	'''
	apply a color map to an array
	'''
	if saturation_limits is None:
		if percentiles is None:
			percentiles = (1.1, 99.9)
		low, high = np.percentile(arr, percentiles)
		saturation_limits = (low, high)

	return dispatch_single_channel(
		array_reference=arr,
		cmap_name=color,
		limits=saturation_limits,
	)


def merge(
	arrs: Sequence[np.ndarray],
	colors: Sequence[COLORMAPS],
	blending: BLENDING_OPTIONS = 'max',
	percentiles: Union[Sequence[tuple[float, float]], None] = None,
	saturation_limits: Union[Sequence[tuple[float, float]], None] = None,
) -> np.ndarray:
	'''
	apply cmaps to arrays and blend the colors
	'''
	if saturation_limits is None:
		if percentiles is None:
			percentiles = [(1.1, 99.9)] * len(arrs)
		saturation_limits = tuple(
			np.percentile(arr, ch_percentiles)
			for arr, ch_percentiles in zip(arrs, percentiles)  # type: ignore
		)
	return dispatch_multi_channel(
		array_references=arrs,
		cmap_names=colors,
		blending=blending,
		limits=saturation_limits,  # type: ignore
	)
