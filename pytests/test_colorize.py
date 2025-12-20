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


def test_apply_with_matplotlib_cmap(matplotlib_viridis_cmap: Colormap):
    """
    Test that the color map is applied correctly with a matplotlib colormap
    """
    x = np.ones((1, 1), dtype=np.uint8)
    rgb = mc.apply_color_map(x, matplotlib_viridis_cmap, saturation_limits=(0, 255))
    assert np.allclose(rgb, np.array([[[68, 2, 85]]]))
    x = x * 255
    rgb = mc.apply_color_map(x, matplotlib_viridis_cmap, saturation_limits=(0, 255))
    assert np.allclose(rgb, np.array([[[253, 231, 36]]]))


def test_apply_with_cmap_colormap(cmap_mako_colormap: cmap.Colormap):
    """
    Test that the color map is applied correctly with a cmap colormap
    """
    x = np.ones((1, 1), dtype=np.uint8)
    rgb = mc.apply_color_map(x, cmap_mako_colormap, saturation_limits=(0, 255))
    assert np.allclose(rgb, np.array([[[12, 4, 6]]]))
    x = x * 255
    rgb = mc.apply_color_map(x, cmap_mako_colormap, saturation_limits=(0, 255))
    assert np.allclose(rgb, np.array([[[222, 244, 228]]]))


def test_merge_with_both(
    matplotlib_viridis_cmap: Colormap, cmap_mako_colormap: cmap.Colormap
) -> None:
    """
    Test that the colors are merged correctly with both matplotlib and cmap colormaps
    """
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
    """
    Test that the color map is applied correctly
    """
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
    """
    Test that the colors are merged correctly with sum blending and low values
    """
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
    """
    Test that the colors are merged correctly with sum blending and high values
    """
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
    """
    Test that the colors are merged correctly with max blending and low values
    """
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
    """
    Test that the colors are merged correctly with max blending and high values
    """
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
    """
    Test that the colors are merged correctly with min blending and low values
    """
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
    """
    Test that the colors are merged correctly with min blending and high values
    """
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
    """
    Test that the colors are merged correctly with mean blending and low values
    """
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
    """
    Test that the colors are merged correctly with mean blending and high values
    """
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
