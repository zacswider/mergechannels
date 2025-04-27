
import numpy as np


# def merge(
#     arrs: Sequence[np.ndarray],
#     colors: Sequence[COLORMAPS] = (),
#     blending: Literal[BLENDING_OPTIONS] = 'max',
#     saturation_limits: tuple[float, float] = (0.2, 99.8),
# ) -> np.ndarray:
#     ...
#
# def apply_color_map(
#     cmap_name: COLORMAPS,
#     saturation_limits: tuple[float, float] = (0.2, 99.8),
# ) -> np.ndarray: ...
#
# def apply_colors_and_merge_nc(
#     arrs: Sequence[np.ndarray],
#     colors: Sequence[COLORMAPS] = (),
#     blending: Literal[BLENDING_OPTIONS] = 'max',
#     saturation_limits: Sequence[tuple[float, float]] = (),
# ) -> np.ndarray: ...

def dispatch_single_channel(
    arr: np.ndarray,
) -> None:
    ...
