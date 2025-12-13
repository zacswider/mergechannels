import numpy as np
import cmap
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import pytest
import mergechannels as mc


@pytest.fixture
def matplotlib_viridis_cmap() -> Colormap:
    '''
    Get the viridis colormap from matplotlib
    '''
    return plt.get_cmap('viridis')


@pytest.fixture
def cmap_mako_colormap() -> cmap.Colormap:
    '''
    Get the seaborn mako colormap from cmap
    '''
    return cmap.Colormap('seaborn:mako')


@pytest.fixture
def small_array_u8() -> np.ndarray:
    '''Create a small u8 array for benchmarking'''
    return np.random.randn(256, 256).astype('uint8')


@pytest.fixture
def small_array_u16() -> np.ndarray:
    '''Create a small u16 array for benchmarking'''
    return np.random.randn(256, 256).astype('uint16')


@pytest.fixture
def large_array_u8() -> np.ndarray:
    '''Create a large u8 array for benchmarking'''
    return np.random.randn(1024, 1024).astype('uint8')


@pytest.fixture
def large_array_u16() -> np.ndarray:
    '''Create a large u16 array for benchmarking'''
    return np.random.randn(1024, 1024).astype('uint16')

@pytest.fixture
def xlarge_array_u8() -> np.ndarray:
    '''Create a large u8 array for benchmarking'''
    return np.random.randn(2048, 2048).astype('uint8')


@pytest.fixture
def xlarge_array_u16() -> np.ndarray:
    '''Create a large u16 array for benchmarking'''
    return np.random.randn(2048, 2048).astype('uint16')


@pytest.mark.benchmark(group='single channel u8 small')
def test_bench_small_u8_no_autoscale(benchmark, small_array_u8) -> None:
    '''Benchmark options for a small u8 array'''
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=small_array_u8, color='Grays', saturation_limits=(0, 255)
    )
    assert colorized_no_autoscale.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='single channel u8 small')
def test_bench_small_u8_yes_autoscale(benchmark, small_array_u8) -> None:
    '''Benchmark options for a small u8 array'''
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=small_array_u8, color='Grays', saturation_limits=None
    )
    assert colorized_no_autoscale.shape[:-1] == small_array_u8.shape


@pytest.mark.benchmark(group='single channel u8 large')
def test_bench_large_u8_no_autoscale(benchmark, large_array_u8) -> None:
    '''Benchmark options for a large u8 array'''
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=large_array_u8, color='Grays', saturation_limits=(0, 255)
    )
    assert colorized_no_autoscale.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='single channel u8 large')
def test_bench_large_u8_yes_autoscale(benchmark, large_array_u8) -> None:
    '''Benchmark options for a large u8 array'''
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=large_array_u8, color='Grays', saturation_limits=None
    )
    assert colorized_no_autoscale.shape[:-1] == large_array_u8.shape


@pytest.mark.benchmark(group='single channel u16 small')
def test_bench_small_u16_no_autoscale(benchmark, small_array_u16) -> None:
    '''Benchmark options for a small u8 array'''
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=small_array_u16, color='Grays', saturation_limits=(0, 2**16)
    )
    assert colorized_no_autoscale.shape[:-1] == small_array_u16.shape


@pytest.mark.benchmark(group='single channel u16 small')
def test_bench_small_u16_yes_autoscale(benchmark, small_array_u16) -> None:
    '''Benchmark options for a small u8 array'''
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=small_array_u16, color='Grays', saturation_limits=None
    )
    assert colorized_no_autoscale.shape[:-1] == small_array_u16.shape


@pytest.mark.benchmark(group='single channel u16 large')
def test_bench_large_u16_no_autoscale(benchmark, large_array_u16) -> None:
    '''Benchmark options for a large u16 array'''
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=large_array_u16, color='Grays', saturation_limits=(0, 2**16)
    )
    assert colorized_no_autoscale.shape[:-1] == large_array_u16.shape


@pytest.mark.benchmark(group='single channel u16 large')
def test_bench_large_u16_yes_autoscale(benchmark, large_array_u16) -> None:
    '''Benchmark options for a large u16 array'''
    colorized_no_autoscale = benchmark(
        mc.apply_color_map, arr=large_array_u16, color='Grays', saturation_limits=None
    )
    assert colorized_no_autoscale.shape[:-1] == large_array_u16.shape


@pytest.mark.benchmark(group='blending approach')
def test_bench_small_u8_max_blending(benchmark, small_array_u8) -> None:
    '''benchmark max blending for 2 small u8 arrays'''
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
    '''benchmark sum blending for 2 small u8 arrays'''
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
    '''benchmark mean blending for 2 small u8 arrays'''
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
    '''benchmark min blending for 2 small u8 arrays'''
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
    '''benchmark max blending for 2 large u8 arrays'''
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
    '''benchmark sum blending for 2 large u8 arrays'''
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
    '''benchmark mean blending for 2 large u8 arrays'''
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
    '''benchmark min blending for 2 large u8 arrays'''
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
    '''benchmark merging two u8 arrays with internal cmaps'''
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
    '''benchmark merging two u8 arrays with an internal cmap and a matplotlib cmap'''
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
    '''benchmark merging two u8 arrays with an internal cmap and a cmap cmap'''
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
    '''benchmark merging two u8 arrays with a matplotlib cmap and a cmap cmap'''
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
    '''benchmark merging two large u8 arrays with internal cmaps'''
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
    '''benchmark merging two large u8 arrays with an internal cmap and a matplotlib cmap'''
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
    '''benchmark merging two large u8 arrays with an internal cmap and a cmap cmap'''
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
    '''benchmark merging two large u8 arrays with a matplotlib cmap and a cmap cmap'''
    large_array_u8_copy = np.copy(large_array_u8)
    colorized = benchmark(
        mc.merge,
        arrs=[large_array_u8, large_array_u8_copy],
        colors=[matplotlib_viridis_cmap, cmap_mako_colormap],
        saturation_limits=[(0, 255), (0, 255)],
        blending='max',
    )
    assert colorized.shape[:-1] == large_array_u8.shape


def test_apply_with_matplotlib_cmap(matplotlib_viridis_cmap: Colormap):
    '''
    Test that the color map is applied correctly with a matplotlib colormap
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    rgb = mc.apply_color_map(x, matplotlib_viridis_cmap, saturation_limits=(0, 255))
    assert np.allclose(rgb, np.array([[[68, 2, 85]]]))
    x = x * 255
    rgb = mc.apply_color_map(x, matplotlib_viridis_cmap, saturation_limits=(0, 255))
    assert np.allclose(rgb, np.array([[[253, 231, 36]]]))


def test_apply_with_cmap_colormap(cmap_mako_colormap: cmap.Colormap):
    '''
    Test that the color map is applied correctly with a cmap colormap
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    rgb = mc.apply_color_map(x, cmap_mako_colormap, saturation_limits=(0, 255))
    assert np.allclose(rgb, np.array([[[12, 4, 6]]]))
    x = x * 255
    rgb = mc.apply_color_map(x, cmap_mako_colormap, saturation_limits=(0, 255))
    assert np.allclose(rgb, np.array([[[222, 244, 228]]]))


def test_merge_with_both(
    matplotlib_viridis_cmap: Colormap, cmap_mako_colormap: cmap.Colormap
) -> None:
    '''
    Test that the colors are merged correctly with both matplotlib and cmap colormaps
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb = mc.merge(
        [x, y],
        [matplotlib_viridis_cmap, cmap_mako_colormap],
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb, np.array([[[68, 4, 85]]]))
    x = x * 255
    y = y * 255
    rgb = mc.merge(
        [x, y],
        [matplotlib_viridis_cmap, cmap_mako_colormap],
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb, np.array([[[253, 244, 228]]]))


def test_apply_color_map():
    '''
    Test that the color map is applied correctly
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    rgb = mc.dispatch_single_channel(x, 'betterBlue', None, (0, 255))
    assert rgb.shape == (1, 1, 3)
    assert rgb.dtype == np.uint8
    assert np.allclose(rgb, np.array([[[0, 1, 2]]]))
    x = np.ones((1, 1), dtype=np.uint8) * 255
    rgb = mc.dispatch_single_channel(x, 'betterBlue', None, (0, 255))
    assert np.allclose(rgb, np.array([[[0, 188, 254]]]))
    rgb2 = mc.apply_color_map(x, 'betterBlue', saturation_limits=(0, 255))
    assert np.allclose(rgb, rgb2)


def test_apply_colors_and_merge_low_sum():
    '''
    Test that the colors are merged correctly with sum blending and low values
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_sum = mc.dispatch_multi_channel(
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'sum', [(0, 255), (0, 255)]
    )
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(rgb_sum, np.array([[[1, 2, 2]]]))
    rgb_sum2 = mc.merge(
        [x, y],
        ['betterBlue', 'betterOrange'],
        blending='sum',
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb_sum, rgb_sum2)


def test_apply_colors_and_merge_high_sum():
    '''
    Test that the colors are merged correctly with sum blending and high values
    '''
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_sum = mc.dispatch_multi_channel(
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'sum', [(0, 255), (0, 255)]
    )
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(rgb_sum, np.array([[[255, 255, 254]]]))
    rgb_sum2 = mc.merge(
        [x, y],
        ['betterBlue', 'betterOrange'],
        blending='sum',
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb_sum, rgb_sum2)


def test_apply_colors_and_merge_low_max():
    '''
    Test that the colors are merged correctly with max blending and low values
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_max = mc.dispatch_multi_channel(
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'max', [(0, 255), (0, 255)]
    )
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(rgb_max, np.array([[[1, 1, 2]]]))
    rgb_max2 = mc.merge(
        [x, y],
        ['betterBlue', 'betterOrange'],
        blending='max',
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb_max, rgb_max2)


def test_apply_colors_and_merge_high_max():
    '''
    Test that the colors are merged correctly with max blending and high values
    '''
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_max = mc.dispatch_multi_channel(
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'max', [(0, 255), (0, 255)]
    )
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(rgb_max, np.array([[[255, 188, 254]]]))
    rgb_max2 = mc.merge(
        [x, y],
        ['betterBlue', 'betterOrange'],
        blending='max',
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb_max, rgb_max2)


def test_apply_colors_and_merge_low_min():
    '''
    Test that the colors are merged correctly with min blending and low values
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_min = mc.dispatch_multi_channel(
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'min', [(0, 255), (0, 255)]
    )
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(rgb_min, np.array([[[0, 1, 0]]]))
    rgb_min2 = mc.merge(
        [x, y],
        ['betterBlue', 'betterOrange'],
        blending='min',
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb_min, rgb_min2)


def test_apply_colors_and_merge_high_min():
    '''
    Test that the colors are merged correctly with min blending and high values
    '''
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_min = mc.dispatch_multi_channel(
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'min', [(0, 255), (0, 255)]
    )
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(rgb_min, np.array([[[0, 149, 0]]]))
    rgb_min2 = mc.merge(
        [x, y],
        ['betterBlue', 'betterOrange'],
        blending='min',
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb_min, rgb_min2)


def test_apply_colors_and_merge_low_mean():
    '''
    Test that the colors are merged correctly with mean blending and low values
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_mean = mc.dispatch_multi_channel(
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'mean', [(0, 255), (0, 255)]
    )
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(
        # NOTE: I haven't decided if I'm going to re-normalize after blending so this test may fail
        # in the future
        rgb_mean,
        np.array([[[0, 1, 1]]]),
    )
    rgb_mean2 = mc.merge(
        [x, y],
        ['betterBlue', 'betterOrange'],
        blending='mean',
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb_mean, rgb_mean2)


def test_apply_colors_and_merge_high_mean():
    '''
    Test that the colors are merged correctly with mean blending and high values
    '''
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_mean = mc.dispatch_multi_channel(
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'mean', [(0, 255), (0, 255)]
    )
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(
        # NOTE: I haven't decided if I'm going to re-normalize after blending so this test may fail
        # in the future
        rgb_mean,
        np.array([[[127, 168, 127]]]),
    )
    rgb_mean2 = mc.merge(
        [x, y],
        ['betterBlue', 'betterOrange'],
        blending='mean',
        saturation_limits=[(0, 255), (0, 255)],
    )
    assert np.allclose(rgb_mean, rgb_mean2)

@pytest.mark.benchmark(group='mergechannels vs numpy')
def test_apply_cmap_u8_matplotlib_small(benchmark, small_array_u8, matplotlib_viridis_cmap) -> None:
    '''
    benchmark time to apply a single colormap to a large u8 array with matplotlib
    NOTE: this uses the underlying mechanism np.take to avoid some of the other matplotlib
    overhead in an attempt to be more fair
    '''
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = benchmark(
        np.take,
        lut,
        small_array_u8,
        axis=0,
    )
    rgb_mc = mc.apply_color_map(small_array_u8, matplotlib_viridis_cmap)
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)

@pytest.mark.benchmark(group='mergechannels vs numpy')
def test_apply_cmap_u8_mergechannels_small(
    benchmark,
    small_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    '''benchmark time to apply a single colormap to a large u8 array with mergechannels'''
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = np.take(lut, small_array_u8, axis=0)
    rgb_mc = benchmark(
        mc.apply_color_map,
        small_array_u8,
        matplotlib_viridis_cmap,
    )
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='mergechannels vs numpy')
def test_apply_cmap_u8_matplotlib_moderate(
    benchmark,
    large_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    '''
    benchmark time to apply a single colormap to a large u8 array with matplotlib
    NOTE: this uses the underlying mechanism np.take to avoid some of the other matplotlib
    overhead in an attempt to be more fair
    '''
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = benchmark(
        np.take,
        lut,
        large_array_u8,
        axis=0,
    )
    rgb_mc = mc.apply_color_map(large_array_u8, matplotlib_viridis_cmap)
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)

@pytest.mark.benchmark(group='mergechannels vs numpy')
def test_apply_cmap_u8_mergechannels_moderate(
    benchmark,
    large_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    '''benchmark time to apply a single colormap to a large u8 array with mergechannels'''
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = np.take(lut, large_array_u8, axis=0)
    rgb_mc = benchmark(
        mc.apply_color_map,
        large_array_u8,
        matplotlib_viridis_cmap,
    )
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)

@pytest.mark.benchmark(group='mergechannels vs numpy')
def test_apply_cmap_u8_matplotlib_large(
    benchmark,
    xlarge_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    '''
    benchmark time to apply a single colormap to a large u8 array with matplotlib
    NOTE: this uses the underlying mechanism np.take to avoid some of the other matplotlib
    overhead in an attempt to be more fair
    '''
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = benchmark(
        np.take,
        lut,
        xlarge_array_u8,
        axis=0,
    )
    rgb_mc = mc.apply_color_map(xlarge_array_u8, matplotlib_viridis_cmap)
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)

@pytest.mark.benchmark(group='mergechannels vs numpy')
def test_apply_cmap_u8_mergechannels_large(
    benchmark,
    xlarge_array_u8,
    matplotlib_viridis_cmap,
) -> None:
    '''benchmark time to apply a single colormap to a large u8 array with mergechannels'''
    lut = (matplotlib_viridis_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb_mpl = np.take(lut, xlarge_array_u8, axis=0)
    rgb_mc = benchmark(
        mc.apply_color_map,
        xlarge_array_u8,
        matplotlib_viridis_cmap,
    )
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)
