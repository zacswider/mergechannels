"""Benchmarks comparing colormap application with and without mask overlay.

Three scenarios are tested:
1. No mask: baseline colormap application
2. All-False mask: measures overhead of mask checking without blending
3. All-True mask: worst-case where every pixel requires alpha blending
"""

import mergechannels as mc
import numpy as np
import pytest

# Mask overlay parameters
MASK_COLOR = (255, 0, 0)  # Red
MASK_ALPHA = 0.5


# =============================================================================
# All-True mask fixtures (worst-case: every pixel blended)
# =============================================================================


@pytest.fixture
def small_mask_true_2d(small_array_u8) -> np.ndarray:
    """All-True mask matching small_array_u8 shape."""
    return np.ones(small_array_u8.shape, dtype=bool)


@pytest.fixture
def medium_mask_true_2d(medium_array_u8) -> np.ndarray:
    """All-True mask matching medium_array_u8 shape."""
    return np.ones(medium_array_u8.shape, dtype=bool)


@pytest.fixture
def large_mask_true_2d(large_array_u8) -> np.ndarray:
    """All-True mask matching large_array_u8 shape."""
    return np.ones(large_array_u8.shape, dtype=bool)


@pytest.fixture
def xlarge_mask_true_2d(xlarge_array_u8) -> np.ndarray:
    """All-True mask matching xlarge_array_u8 shape."""
    return np.ones(xlarge_array_u8.shape, dtype=bool)


@pytest.fixture
def small_3d_mask_true(small_3d_array_u8) -> np.ndarray:
    """All-True mask matching small_3d_array_u8 shape."""
    return np.ones(small_3d_array_u8.shape, dtype=bool)


@pytest.fixture
def medium_3d_mask_true(medium_3d_array_u8) -> np.ndarray:
    """All-True mask matching medium_3d_array_u8 shape."""
    return np.ones(medium_3d_array_u8.shape, dtype=bool)


@pytest.fixture
def large_3d_mask_true(large_3d_array_u8) -> np.ndarray:
    """All-True mask matching large_3d_array_u8 shape."""
    return np.ones(large_3d_array_u8.shape, dtype=bool)


# =============================================================================
# All-False mask fixtures (control: check overhead without blending)
# =============================================================================


@pytest.fixture
def small_mask_false_2d(small_array_u8) -> np.ndarray:
    """All-False mask matching small_array_u8 shape."""
    return np.zeros(small_array_u8.shape, dtype=bool)


@pytest.fixture
def medium_mask_false_2d(medium_array_u8) -> np.ndarray:
    """All-False mask matching medium_array_u8 shape."""
    return np.zeros(medium_array_u8.shape, dtype=bool)


@pytest.fixture
def large_mask_false_2d(large_array_u8) -> np.ndarray:
    """All-False mask matching large_array_u8 shape."""
    return np.zeros(large_array_u8.shape, dtype=bool)


@pytest.fixture
def xlarge_mask_false_2d(xlarge_array_u8) -> np.ndarray:
    """All-False mask matching xlarge_array_u8 shape."""
    return np.zeros(xlarge_array_u8.shape, dtype=bool)


@pytest.fixture
def small_3d_mask_false(small_3d_array_u8) -> np.ndarray:
    """All-False mask matching small_3d_array_u8 shape."""
    return np.zeros(small_3d_array_u8.shape, dtype=bool)


@pytest.fixture
def medium_3d_mask_false(medium_3d_array_u8) -> np.ndarray:
    """All-False mask matching medium_3d_array_u8 shape."""
    return np.zeros(medium_3d_array_u8.shape, dtype=bool)


@pytest.fixture
def large_3d_mask_false(large_3d_array_u8) -> np.ndarray:
    """All-False mask matching large_3d_array_u8 shape."""
    return np.zeros(large_3d_array_u8.shape, dtype=bool)


# =============================================================================
# 2D u8 benchmarks - no mask vs all-False mask vs all-True mask
# =============================================================================


@pytest.mark.benchmark(group='small 2d u8 colormap vs mask overlay')
def test_bench_small_u8_no_mask(benchmark, small_array_u8) -> None:
    """Benchmark colorization without mask for small u8 array."""
    benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )


@pytest.mark.benchmark(group='small 2d u8 colormap vs mask overlay')
def test_bench_small_u8_mask_false(benchmark, small_array_u8, small_mask_false_2d) -> None:
    """Benchmark colorization with all-False mask for small u8 array (check overhead only)."""
    benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[small_mask_false_2d],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='small 2d u8 colormap vs mask overlay')
def test_bench_small_u8_mask_true(benchmark, small_array_u8, small_mask_true_2d) -> None:
    """Benchmark colorization with all-True mask for small u8 array (full blending)."""
    benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[small_mask_true_2d],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='medium 2d u8 colormap vs mask overlay')
def test_bench_medium_u8_no_mask(benchmark, medium_array_u8) -> None:
    """Benchmark colorization without mask for medium u8 array."""
    benchmark(
        mc.apply_color_map,
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )


@pytest.mark.benchmark(group='medium 2d u8 colormap vs mask overlay')
def test_bench_medium_u8_mask_false(benchmark, medium_array_u8, medium_mask_false_2d) -> None:
    """Benchmark colorization with all-False mask for medium u8 array (check overhead only)."""
    benchmark(
        mc.apply_color_map,
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[medium_mask_false_2d],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='medium 2d u8 colormap vs mask overlay')
def test_bench_medium_u8_mask_true(benchmark, medium_array_u8, medium_mask_true_2d) -> None:
    """Benchmark colorization with all-True mask for medium u8 array (full blending)."""
    benchmark(
        mc.apply_color_map,
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[medium_mask_true_2d],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='large 2d u8 colormap vs mask overlay')
def test_bench_large_u8_no_mask(benchmark, large_array_u8) -> None:
    """Benchmark colorization without mask for large u8 array."""
    benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )


@pytest.mark.benchmark(group='large 2d u8 colormap vs mask overlay')
def test_bench_large_u8_mask_false(benchmark, large_array_u8, large_mask_false_2d) -> None:
    """Benchmark colorization with all-False mask for large u8 array (check overhead only)."""
    benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[large_mask_false_2d],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='large 2d u8 colormap vs mask overlay')
def test_bench_large_u8_mask_true(benchmark, large_array_u8, large_mask_true_2d) -> None:
    """Benchmark colorization with all-True mask for large u8 array (full blending)."""
    benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[large_mask_true_2d],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='xlarge 2d u8 colormap vs mask overlay')
def test_bench_xlarge_u8_no_mask(benchmark, xlarge_array_u8) -> None:
    """Benchmark colorization without mask for xlarge u8 array."""
    benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )


@pytest.mark.benchmark(group='xlarge 2d u8 colormap vs mask overlay')
def test_bench_xlarge_u8_mask_false(benchmark, xlarge_array_u8, xlarge_mask_false_2d) -> None:
    """Benchmark colorization with all-False mask for xlarge u8 array (check overhead only)."""
    benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[xlarge_mask_false_2d],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='xlarge 2d u8 colormap vs mask overlay')
def test_bench_xlarge_u8_mask_true(benchmark, xlarge_array_u8, xlarge_mask_true_2d) -> None:
    """Benchmark colorization with all-True mask for xlarge u8 array (full blending)."""
    benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[xlarge_mask_true_2d],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


# =============================================================================
# 3D u8 benchmarks - no mask vs all-False mask vs all-True mask
# =============================================================================


@pytest.mark.benchmark(group='small 3d u8 colormap vs mask overlay')
def test_bench_small_3d_u8_no_mask(benchmark, small_3d_array_u8) -> None:
    """Benchmark colorization without mask for small 3d u8 stack."""
    benchmark(
        mc.apply_color_map,
        arr=small_3d_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )


@pytest.mark.benchmark(group='small 3d u8 colormap vs mask overlay')
def test_bench_small_3d_u8_mask_false(benchmark, small_3d_array_u8, small_3d_mask_false) -> None:
    """Benchmark colorization with all-False mask for small 3d u8 stack (check overhead only)."""
    benchmark(
        mc.apply_color_map,
        arr=small_3d_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[small_3d_mask_false],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='small 3d u8 colormap vs mask overlay')
def test_bench_small_3d_u8_mask_true(benchmark, small_3d_array_u8, small_3d_mask_true) -> None:
    """Benchmark colorization with all-True mask for small 3d u8 stack (full blending)."""
    benchmark(
        mc.apply_color_map,
        arr=small_3d_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[small_3d_mask_true],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='medium 3d u8 colormap vs mask overlay')
def test_bench_medium_3d_u8_no_mask(benchmark, medium_3d_array_u8) -> None:
    """Benchmark colorization without mask for medium 3d u8 stack."""
    benchmark(
        mc.apply_color_map,
        arr=medium_3d_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )


@pytest.mark.benchmark(group='medium 3d u8 colormap vs mask overlay')
def test_bench_medium_3d_u8_mask_false(benchmark, medium_3d_array_u8, medium_3d_mask_false) -> None:
    """Benchmark colorization with all-False mask for medium 3d u8 stack (check overhead only)."""
    benchmark(
        mc.apply_color_map,
        arr=medium_3d_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[medium_3d_mask_false],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='medium 3d u8 colormap vs mask overlay')
def test_bench_medium_3d_u8_mask_true(benchmark, medium_3d_array_u8, medium_3d_mask_true) -> None:
    """Benchmark colorization with all-True mask for medium 3d u8 stack (full blending)."""
    benchmark(
        mc.apply_color_map,
        arr=medium_3d_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[medium_3d_mask_true],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='large 3d u8 colormap vs mask overlay')
def test_bench_large_3d_u8_no_mask(benchmark, large_3d_array_u8) -> None:
    """Benchmark colorization without mask for large 3d u8 stack."""
    benchmark(
        mc.apply_color_map,
        arr=large_3d_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )


@pytest.mark.benchmark(group='large 3d u8 colormap vs mask overlay')
def test_bench_large_3d_u8_mask_false(benchmark, large_3d_array_u8, large_3d_mask_false) -> None:
    """Benchmark colorization with all-False mask for large 3d u8 stack (check overhead only)."""
    benchmark(
        mc.apply_color_map,
        arr=large_3d_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[large_3d_mask_false],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )


@pytest.mark.benchmark(group='large 3d u8 colormap vs mask overlay')
def test_bench_large_3d_u8_mask_true(benchmark, large_3d_array_u8, large_3d_mask_true) -> None:
    """Benchmark colorization with all-True mask for large 3d u8 stack (full blending)."""
    benchmark(
        mc.apply_color_map,
        arr=large_3d_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        masks=[large_3d_mask_true],
        mask_colors=[MASK_COLOR],
        mask_alphas=[MASK_ALPHA],
        parallel=True,
    )
