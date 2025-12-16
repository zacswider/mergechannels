import cmap
import matplotlib.pyplot as plt
import mergechannels as mc
import numpy as np
import pytest
from matplotlib.colors import Colormap


@pytest.fixture
def matplotlib_viridis_cmap() -> Colormap:
    """
    Get the viridis colormap from matplotlib
    """
    return plt.get_cmap('viridis')


@pytest.fixture
def cmap_mako_colormap() -> cmap.Colormap:
    """
    Get the seaborn mako colormap from cmap
    """
    return cmap.Colormap('seaborn:mako')


@pytest.fixture
def mpl_greens_array_lut() -> np.ndarray:
    """
    Return the array version of the matplotlib greens colormap
    """
    greens = plt.get_cmap('Greens_r')
    return (greens(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


@pytest.fixture
def mpl_reds_array_lut() -> np.ndarray:
    """
    Return the array version of the matplotlib greens colormap
    """
    reds = plt.get_cmap('Reds_r')
    return (reds(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def np_merge(
    arrs: list[np.ndarray],
    cmaps: list[np.ndarray],
) -> np.ndarray:
    """
    merge some number of arrays using numpy operations
    """
    colored = []
    for a, c in zip(arrs, cmaps):
        colored.append(np.take(c, a, axis=0))
    return np.maximum(*colored)


def mc_merge(
    arrs: list[np.ndarray],
    cmaps: list[np.ndarray],
    parallel: bool = False,
) -> np.ndarray:
    """
    merge some number of arrays using mergechannels operations
    """
    return mc.merge(arrs=arrs, colors=cmaps, parallel=parallel)  # type: ignore


@pytest.fixture
def small_array_u8() -> np.ndarray:
    """Create a small u8 array for benchmarking"""
    return np.random.randn(256, 256).astype('uint8')


@pytest.fixture
def small_array_u16() -> np.ndarray:
    """Create a small u16 array for benchmarking"""
    return np.random.randn(256, 256).astype('uint16')


@pytest.fixture
def medium_array_u8() -> np.ndarray:
    """Create a medium u8 array for benchmarking"""
    return np.random.randn(512, 512).astype('uint8')


@pytest.fixture
def medium_array_u16() -> np.ndarray:
    """Create a medium u16 array for benchmarking"""
    return np.random.randn(512, 512).astype('uint16')


@pytest.fixture
def large_array_u8() -> np.ndarray:
    """Create a large u8 array for benchmarking"""
    return np.random.randn(1024, 1024).astype('uint8')


@pytest.fixture
def large_array_u16() -> np.ndarray:
    """Create a large u16 array for benchmarking"""
    return np.random.randn(1024, 1024).astype('uint16')


@pytest.fixture
def xlarge_array_u8() -> np.ndarray:
    """Create a large u8 array for benchmarking"""
    return np.random.randn(2048, 2048).astype('uint8')


@pytest.fixture
def xlarge_array_u16() -> np.ndarray:
    """Create a large u16 array for benchmarking"""
    return np.random.randn(2048, 2048).astype('uint16')


@pytest.mark.benchmark(group='single channel u8 small')
def test_bench_small_u8_no_autoscale(benchmark, small_array_u8) -> None:
    """Benchmark options for a small u8 array"""
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=small_array_u8, color='Grays', saturation_limits=(0, 255)
    )
    assert colorized_no_autoscale.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='single channel u8 small')
def test_bench_small_u8_yes_autoscale(benchmark, small_array_u8) -> None:
    """Benchmark options for a small u8 array"""
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=small_array_u8, color='Grays', saturation_limits=None
    )
    assert colorized_no_autoscale.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='single channel u8 large')
def test_bench_large_u8_no_autoscale(benchmark, large_array_u8) -> None:
    """Benchmark options for a large u8 array"""
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=large_array_u8, color='Grays', saturation_limits=(0, 255)
    )
    assert colorized_no_autoscale.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='single channel u8 large')
def test_bench_large_u8_yes_autoscale(benchmark, large_array_u8) -> None:
    """Benchmark options for a large u8 array"""
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=large_array_u8, color='Grays', saturation_limits=None
    )
    assert colorized_no_autoscale.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='single channel u16 small')
def test_bench_small_u16_no_autoscale(benchmark, small_array_u16) -> None:
    """Benchmark options for a small u8 array"""
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=small_array_u16, color='Grays', saturation_limits=(0, 2**16)
    )
    assert colorized_no_autoscale.shape[:-1] == small_array_u16.shape


@pytest.mark.benchmark(group='single channel u16 small')
def test_bench_small_u16_yes_autoscale(benchmark, small_array_u16) -> None:
    """Benchmark options for a small u8 array"""
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=small_array_u16, color='Grays', saturation_limits=None
    )
    assert colorized_no_autoscale.shape[:-1] == small_array_u16.shape


@pytest.mark.benchmark(group='single channel u16 large')
def test_bench_large_u16_no_autoscale(benchmark, large_array_u16) -> None:
    """Benchmark options for a large u16 array"""
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=large_array_u16, color='Grays', saturation_limits=(0, 2**16)
    )
    assert colorized_no_autoscale.shape[:-1] == large_array_u16.shape


@pytest.mark.benchmark(group='single channel u16 large')
def test_bench_large_u16_yes_autoscale(benchmark, large_array_u16) -> None:
    """Benchmark options for a large u16 array"""
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=large_array_u16, color='Grays', saturation_limits=None
    )
    assert colorized_no_autoscale.shape[:-1] == large_array_u16.shape


@pytest.mark.benchmark(group='blending approach')
def test_bench_small_u8_max_blending(benchmark, small_array_u8) -> None:
    """benchmark max blending for 2 small u8 arrays"""
    small_array_u8_copy = np.copy(small_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[small_array_u8, small_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=None,
        blending='max',
    )
    assert colorized.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='blending approach')
def test_bench_small_u8_sum_blending(benchmark, small_array_u8) -> None:
    """benchmark sum blending for 2 small u8 arrays"""
    small_array_u8_copy = np.copy(small_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[small_array_u8, small_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=None,
        blending='sum',
    )
    assert colorized.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='blending approach')
def test_bench_small_u8_mean_blending(benchmark, small_array_u8) -> None:
    """benchmark mean blending for 2 small u8 arrays"""
    small_array_u8_copy = np.copy(small_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[small_array_u8, small_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=None,
        blending='mean',
    )
    assert colorized.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='blending approach')
def test_bench_small_u8_min_blending(benchmark, small_array_u8) -> None:
    """benchmark min blending for 2 small u8 arrays"""
    small_array_u8_copy = np.copy(small_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[small_array_u8, small_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=None,
        blending='min',
    )
    assert colorized.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='blending approach large')
def test_bench_large_u8_max_blending(benchmark, large_array_u8) -> None:
    """benchmark max blending for 2 large u8 arrays"""
    large_array_u8_copy = np.copy(large_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[large_array_u8, large_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=None,
        blending='max',
    )
    assert colorized.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='blending approach large')
def test_bench_large_u8_sum_blending(benchmark, large_array_u8) -> None:
    """benchmark sum blending for 2 large u8 arrays"""
    large_array_u8_copy = np.copy(large_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[large_array_u8, large_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=None,
        blending='sum',
    )
    assert colorized.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='blending approach large')
def test_bench_large_u8_mean_blending(benchmark, large_array_u8) -> None:
    """benchmark mean blending for 2 large u8 arrays"""
    large_array_u8_copy = np.copy(large_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[large_array_u8, large_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=None,
        blending='mean',
    )
    assert colorized.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='blending approach large')
def test_bench_large_u8_min_blending(benchmark, large_array_u8) -> None:
    """benchmark min blending for 2 large u8 arrays"""
    large_array_u8_copy = np.copy(large_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[large_array_u8, large_array_u8_copy],
        colors=['Red', 'Green'],
        saturation_limits=None,
        blending='min',
    )
    assert colorized.shape[:-1] == large_array_u8.shape


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


@pytest.mark.benchmark(group='mergechannels vs numpy single array')
def test_apply_cmap_u8_matplotlib_small(benchmark, small_array_u8, matplotlib_viridis_cmap) -> None:
    """
    benchmark time to apply a single colormap to a large u8 array with matplotlib
    NOTE: this uses the underlying mechanism np.take to avoid some of the other matplotlib
    overhead in an attempt to be more fair
    """
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = benchmark(
        np.take,
        lut,
        small_array_u8,
        axis=0,
    )
    rgb_mc = mc.apply_color_map(small_array_u8, matplotlib_viridis_cmap)
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='mergechannels vs numpy single array')
def test_apply_cmap_u8_mergechannels_small(
    benchmark,
    small_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    """benchmark time to apply a single colormap to a large u8 array with mergechannels"""
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = np.take(lut, small_array_u8, axis=0)
    rgb_mc = benchmark(
        mc.apply_color_map,
        small_array_u8,
        matplotlib_viridis_cmap,
    )
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='mergechannels vs numpy single array')
def test_apply_cmap_u8_matplotlib_moderate(
    benchmark,
    large_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    """
    benchmark time to apply a single colormap to a large u8 array with matplotlib
    NOTE: this uses the underlying mechanism np.take to avoid some of the other matplotlib
    overhead in an attempt to be more fair
    """
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = benchmark(
        np.take,
        lut,
        large_array_u8,
        axis=0,
    )
    rgb_mc = mc.apply_color_map(large_array_u8, matplotlib_viridis_cmap)
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='mergechannels vs numpy single array')
def test_apply_cmap_u8_mergechannels_moderate(
    benchmark,
    large_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    """benchmark time to apply a single colormap to a large u8 array with mergechannels"""
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = np.take(lut, large_array_u8, axis=0)
    rgb_mc = benchmark(
        mc.apply_color_map,
        large_array_u8,
        matplotlib_viridis_cmap,
    )
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='mergechannels vs numpy single array')
def test_apply_cmap_u8_matplotlib_large(
    benchmark,
    xlarge_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    """
    benchmark time to apply a single colormap to a large u8 array with matplotlib
    NOTE: this uses the underlying mechanism np.take to avoid some of the other matplotlib
    overhead in an attempt to be more fair
    """
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = benchmark(
        np.take,
        lut,
        xlarge_array_u8,
        axis=0,
    )
    rgb_mc = mc.apply_color_map(xlarge_array_u8, matplotlib_viridis_cmap)
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='mergechannels vs numpy single array')
def test_apply_cmap_u8_mergechannels_large(
    benchmark,
    xlarge_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    """benchmark time to apply a single colormap to a large u8 array with mergechannels"""
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = np.take(lut, xlarge_array_u8, axis=0)
    rgb_mc = benchmark(
        mc.apply_color_map,
        xlarge_array_u8,
        matplotlib_viridis_cmap,
    )
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays')
def test_merge_u8_matplotlib_small(
    benchmark,
    small_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with numpy operations
    """
    small_array_u8_copy = np.copy(small_array_u8)
    np_merged = benchmark(
        np_merge,
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = mc_merge(
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays')
def test_merge_u8_mergechannels_small(
    benchmark,
    small_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with mergechannels operations
    """
    small_array_u8_copy = np.copy(small_array_u8)
    np_merged = np_merge(
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = benchmark(
        mc_merge,
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays')
def test_merge_u8_matplotlib_medium(
    benchmark,
    medium_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with numpy operations
    """
    medium_array_u8_copy = np.copy(medium_array_u8)
    np_merged = benchmark(
        np_merge,
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = mc_merge(
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays')
def test_merge_u8_mergechannels_medium(
    benchmark,
    medium_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with mergechannels operations
    """
    medium_array_u8_copy = np.copy(medium_array_u8)
    np_merged = np_merge(
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = benchmark(
        mc_merge,
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays')
def test_merge_u8_matplotlib_large(
    benchmark,
    large_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with numpy operations
    """
    large_array_u8_copy = np.copy(large_array_u8)
    np_merged = benchmark(
        np_merge,
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = mc_merge(
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays')
def test_merge_u8_mergechannels_large(
    benchmark,
    large_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with mergechannels operations
    """
    large_array_u8_copy = np.copy(large_array_u8)
    np_merged = np_merge(
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = benchmark(
        mc_merge,
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays')
def test_merge_u8_matplotlib_xlarge(
    benchmark,
    xlarge_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with numpy operations
    """
    xlarge_array_u8_copy = np.copy(xlarge_array_u8)
    np_merged = benchmark(
        np_merge,
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = mc_merge(
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays')
def test_merge_u8_mergechannels_xlarge(
    benchmark,
    xlarge_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with mergechannels operations
    """
    xlarge_array_u8_copy = np.copy(xlarge_array_u8)
    np_merged = np_merge(
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = benchmark(
        mc_merge,
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='serial vs parallel small fast path')
def test_bench_small_u8_serial_fast(benchmark, small_array_u8) -> None:
    """Benchmark serial colorization with fast path"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    assert rgb.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel small fast path')
def test_bench_small_u8_parallel_fast(benchmark, small_array_u8) -> None:
    """Benchmark parallel colorization with fast path"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert rgb.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel large fast path')
def test_bench_large_u8_serial_fast(benchmark, large_array_u8) -> None:
    """Benchmark serial colorization with fast path on large array"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    assert rgb.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel large fast path')
def test_bench_large_u8_parallel_fast(benchmark, large_array_u8) -> None:
    """Benchmark parallel colorization with fast path on large array"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert rgb.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel xlarge fast path')
def test_bench_xlarge_u8_serial_fast(benchmark, xlarge_array_u8) -> None:
    """Benchmark serial colorization with fast path on xlarge array"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    assert rgb.shape[:-1] == xlarge_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel xlarge fast path')
def test_bench_xlarge_u8_parallel_fast(benchmark, xlarge_array_u8) -> None:
    """Benchmark parallel colorization with fast path on xlarge array"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert rgb.shape[:-1] == xlarge_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel small slow path')
def test_bench_small_u8_serial_slow(benchmark, small_array_u8) -> None:
    """Benchmark serial colorization with slow path (normalization)"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    assert rgb.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel small slow path')
def test_bench_small_u8_parallel_slow(benchmark, small_array_u8) -> None:
    """Benchmark parallel colorization with slow path (normalization)"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    assert rgb.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel large slow path')
def test_bench_large_u8_serial_slow(benchmark, large_array_u8) -> None:
    """Benchmark serial colorization with slow path on large array"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    assert rgb.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel large slow path')
def test_bench_large_u8_parallel_slow(benchmark, large_array_u8) -> None:
    """Benchmark parallel colorization with slow path on large array"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    assert rgb.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel xlarge slow path')
def test_bench_xlarge_u8_serial_slow(benchmark, xlarge_array_u8) -> None:
    """Benchmark serial colorization with slow path on xlarge array"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    assert rgb.shape[:-1] == xlarge_array_u8.shape


@pytest.mark.benchmark(group='serial vs parallel xlarge slow path')
def test_bench_xlarge_u8_parallel_slow(benchmark, xlarge_array_u8) -> None:
    """Benchmark parallel colorization with slow path on xlarge array"""
    rgb = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    assert rgb.shape[:-1] == xlarge_array_u8.shape


def test_parallel_correctness_fast_path(small_array_u8) -> None:
    """Test that parallel and serial produce identical results for fast path"""
    rgb_serial = mc.apply_color_map(
        arr=small_array_u8, color='Grays', saturation_limits=(0, 255), parallel=False
    )
    rgb_parallel = mc.apply_color_map(
        arr=small_array_u8, color='Grays', saturation_limits=(0, 255), parallel=True
    )
    np.testing.assert_array_equal(rgb_serial, rgb_parallel)


def test_parallel_correctness_slow_path(small_array_u8) -> None:
    """Test that parallel and serial produce identical results for slow path"""
    rgb_serial = mc.apply_color_map(
        arr=small_array_u8, color='Grays', saturation_limits=(10, 200), parallel=False
    )
    rgb_parallel = mc.apply_color_map(
        arr=small_array_u8, color='Grays', saturation_limits=(10, 200), parallel=True
    )
    np.testing.assert_array_equal(rgb_serial, rgb_parallel)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays parallel')
def test_merge_u8_matplotlib_small_parallel(
    benchmark,
    small_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with numpy operations
    """
    small_array_u8_copy = np.copy(small_array_u8)
    np_merged = benchmark(
        np_merge,
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = mc_merge(
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
        parallel=True,
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays parallel')
def test_merge_u8_mergechannels_small_parallel(
    benchmark,
    small_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with mergechannels operations
    """
    small_array_u8_copy = np.copy(small_array_u8)
    np_merged = np_merge(
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = benchmark(
        mc_merge,
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
        parallel=True,
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays parallel')
def test_merge_u8_matplotlib_medium_parallel(
    benchmark,
    medium_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with numpy operations
    """
    medium_array_u8_copy = np.copy(medium_array_u8)
    np_merged = benchmark(
        np_merge,
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = mc_merge(
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
        parallel=True,
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays parallel')
def test_merge_u8_mergechannels_medium_parallel(
    benchmark,
    medium_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with mergechannels operations
    """
    medium_array_u8_copy = np.copy(medium_array_u8)
    np_merged = np_merge(
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = benchmark(
        mc_merge,
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
        parallel=True,
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays parallel')
def test_merge_u8_matplotlib_large_parallel(
    benchmark,
    large_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with numpy operations
    """
    large_array_u8_copy = np.copy(large_array_u8)
    np_merged = benchmark(
        np_merge,
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = mc_merge(
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
        parallel=True,
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays parallel')
def test_merge_u8_mergechannels_large_parallel(
    benchmark,
    large_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with mergechannels operations
    """
    large_array_u8_copy = np.copy(large_array_u8)
    np_merged = np_merge(
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = benchmark(
        mc_merge,
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
        parallel=True,
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays parallel')
def test_merge_u8_matplotlib_xlarge_parallel(
    benchmark,
    xlarge_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with numpy operations
    """
    xlarge_array_u8_copy = np.copy(xlarge_array_u8)
    np_merged = benchmark(
        np_merge,
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = mc_merge(
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
        parallel=True,
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='mergechannels vs numpy merge arrays parallel')
def test_merge_u8_mergechannels_xlarge_parallel(
    benchmark,
    xlarge_array_u8,
    mpl_greens_array_lut,
    mpl_reds_array_lut,
) -> None:
    """
    benchmark time to merge two images with mergechannels operations
    """
    xlarge_array_u8_copy = np.copy(xlarge_array_u8)
    np_merged = np_merge(
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    mc_merged = benchmark(
        mc_merge,
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
        parallel=True,
    )
    np.testing.assert_allclose(np_merged, mc_merged)
