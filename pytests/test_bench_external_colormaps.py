import mergechannels as mc
import numpy as np
import pytest


@pytest.mark.benchmark(group='external colormaps')
def test_bench_small_u8_internal_cmaps(benchmark, small_array_u8) -> None:
    """benchmark merging two u8 arrays with internal cmaps"""
    small_array_u8_copy = np.copy(small_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[small_array_u8, small_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=[(0, 255), (0, 255)],
        blending='max',
    )
    assert colorized.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='external colormaps')
def test_bench_small_u8_internal_and_matplotlib_cmaps(
    benchmark, small_array_u8, matplotlib_viridis_cmap
) -> None:
    """benchmark merging two u8 arrays with an internal cmap and a matplotlib cmap"""
    small_array_u8_copy = np.copy(small_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[small_array_u8, small_array_u8_copy],
        colors=['Red', matplotlib_viridis_cmap],
        saturation_limits=[(0, 255), (0, 255)],
        blending='max',
    )
    assert colorized.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='external colormaps')
def test_bench_small_u8_internal_and_cmap_cmaps(
    benchmark, small_array_u8, cmap_mako_colormap
) -> None:
    """benchmark merging two u8 arrays with an internal cmap and a cmap cmap"""
    small_array_u8_copy = np.copy(small_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[small_array_u8, small_array_u8_copy],
        colors=['Red', cmap_mako_colormap],
        saturation_limits=[(0, 255), (0, 255)],
        blending='max',
    )
    assert colorized.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='external colormaps')
def test_bench_small_u8_matplotlib_and_cmap_cmaps(
    benchmark, small_array_u8, matplotlib_viridis_cmap, cmap_mako_colormap
) -> None:
    """benchmark merging two u8 arrays with a matplotlib cmap and a cmap cmap"""
    small_array_u8_copy = np.copy(small_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[small_array_u8, small_array_u8_copy],
        colors=[matplotlib_viridis_cmap, cmap_mako_colormap],
        saturation_limits=[(0, 255), (0, 255)],
        blending='max',
    )
    assert colorized.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='external colormaps large')
def test_bench_large_u8_internal_cmaps(benchmark, large_array_u8) -> None:
    """benchmark merging two large u8 arrays with internal cmaps"""
    large_array_u8_copy = np.copy(large_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[large_array_u8, large_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=[(0, 255), (0, 255)],
        blending='max',
    )
    assert colorized.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='external colormaps large')
def test_bench_large_u8_internal_and_matplotlib_cmaps(
    benchmark, large_array_u8, matplotlib_viridis_cmap
) -> None:
    """benchmark merging two large u8 arrays with an internal cmap and a matplotlib cmap"""
    large_array_u8_copy = np.copy(large_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[large_array_u8, large_array_u8_copy],
        colors=['Red', matplotlib_viridis_cmap],
        saturation_limits=[(0, 255), (0, 255)],
        blending='max',
    )
    assert colorized.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='external colormaps large')
def test_bench_large_u8_internal_and_cmap_cmaps(
    benchmark, large_array_u8, cmap_mako_colormap
) -> None:
    """benchmark merging two large u8 arrays with an internal cmap and a cmap cmap"""
    large_array_u8_copy = np.copy(large_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[large_array_u8, large_array_u8_copy],
        colors=['Red', cmap_mako_colormap],
        saturation_limits=[(0, 255), (0, 255)],
        blending='max',
    )
    assert colorized.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='external colormaps large')
def test_bench_large_u8_matplotlib_and_cmap_cmaps(
    benchmark, large_array_u8, matplotlib_viridis_cmap, cmap_mako_colormap
) -> None:
    """benchmark merging two large u8 arrays with a matplotlib cmap and a cmap cmap"""
    large_array_u8_copy = np.copy(large_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[large_array_u8, large_array_u8_copy],
        colors=[matplotlib_viridis_cmap, cmap_mako_colormap],
        saturation_limits=[(0, 255), (0, 255)],
        blending='max',
    )
    assert colorized.shape[:-1] == large_array_u8.shape
