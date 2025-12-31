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
    rgb = mc.dispatch_single_channel(x, 'betterBlue', None, (0, 255), parallel=False)
    assert rgb.shape == (1, 1, 3)
    assert rgb.dtype == np.uint8
    assert np.allclose(rgb, np.array([[[0, 1, 2]]]))
    x = np.ones((1, 1), dtype=np.uint8) * 255
    rgb = mc.dispatch_single_channel(x, 'betterBlue', None, (0, 255), parallel=False)
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
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'sum', [(0, 255), (0, 255)], False
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
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'sum', [(0, 255), (0, 255)], False
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
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'max', [(0, 255), (0, 255)], False
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
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'max', [(0, 255), (0, 255)], False
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
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'min', [(0, 255), (0, 255)], False
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
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'min', [(0, 255), (0, 255)], False
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
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'mean', [(0, 255), (0, 255)], False
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
        [x, y], ['betterBlue', 'betterOrange'], [None, None], 'mean', [(0, 255), (0, 255)], False
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


def test_u16_2d_serial_vs_parallel():
    """
    Test that serial and parallel colorization produce identical results for u16 2D arrays
    """
    arr = np.random.randint(0, 2**16, size=(512, 512), dtype=np.uint16)
    serial_result = mc.apply_color_map(
        arr=arr,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=arr,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_u16_2d_serial_vs_parallel_custom_limits():
    """
    Test that serial and parallel colorization produce identical results for u16 2D arrays with
    custom limits
    """
    arr = np.random.randint(0, 2**16, size=(512, 512), dtype=np.uint16)
    serial_result = mc.apply_color_map(
        arr=arr,
        color='betterBlue',
        saturation_limits=(1000, 50000),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=arr,
        color='betterBlue',
        saturation_limits=(1000, 50000),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_u16_3d_serial_vs_parallel():
    """
    Test that serial and parallel colorization produce identical results for u16 3D arrays
    """
    arr = np.random.randint(0, 2**16, size=(10, 256, 256), dtype=np.uint16)
    serial_result = mc.apply_color_map(
        arr=arr,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=arr,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_u16_3d_serial_vs_parallel_custom_limits():
    """
    Test that serial and parallel colorization produce identical results for u16 3D arrays with
    custom limits
    """
    arr = np.random.randint(0, 2**16, size=(10, 256, 256), dtype=np.uint16)
    serial_result = mc.apply_color_map(
        arr=arr,
        color='betterBlue',
        saturation_limits=(1000, 50000),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=arr,
        color='betterBlue',
        saturation_limits=(1000, 50000),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_2d_u8_serial_vs_parallel_no_autoscale():
    """
    Test that serial and parallel merge produce identical results for 2D u8 arrays without
    autoscaling
    """
    arr1 = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    arr2 = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_2d_u8_serial_vs_parallel_with_autoscale():
    """
    Test that serial and parallel merge produce identical results for 2D u8 arrays with autoscaling
    """
    arr1 = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    arr2 = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_2d_u8_serial_vs_parallel_three_channels():
    """
    Test that serial and parallel merge produce identical results for 2D u8 arrays with three
    channels
    """
    arr1 = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    arr2 = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    arr3 = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    serial_result = mc.merge(
        arrs=[arr1, arr2, arr3],
        colors=['betterBlue', 'betterOrange', 'betterGreen'],
        saturation_limits=[(0, 255), (0, 255), (0, 255)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2, arr3],
        colors=['betterBlue', 'betterOrange', 'betterGreen'],
        saturation_limits=[(0, 255), (0, 255), (0, 255)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_3d_u8_serial_vs_parallel_no_autoscale():
    """
    Test that serial and parallel merge produce identical results for 3D u8 arrays without
    autoscaling
    """
    arr1 = np.random.randint(0, 256, size=(50, 256, 256), dtype=np.uint8)
    arr2 = np.random.randint(0, 256, size=(50, 256, 256), dtype=np.uint8)
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_3d_u8_serial_vs_parallel_with_autoscale():
    """
    Test that serial and parallel merge produce identical results for 3D u8 arrays with autoscaling
    """
    arr1 = np.random.randint(0, 256, size=(50, 256, 256), dtype=np.uint8)
    arr2 = np.random.randint(0, 256, size=(50, 256, 256), dtype=np.uint8)
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_3d_u8_serial_vs_parallel_three_channels():
    """
    Test that serial and parallel merge produce identical results for 3D u8 arrays with three
    channels
    """
    arr1 = np.random.randint(0, 256, size=(50, 256, 256), dtype=np.uint8)
    arr2 = np.random.randint(0, 256, size=(50, 256, 256), dtype=np.uint8)
    arr3 = np.random.randint(0, 256, size=(50, 256, 256), dtype=np.uint8)
    serial_result = mc.merge(
        arrs=[arr1, arr2, arr3],
        colors=['betterBlue', 'betterOrange', 'betterGreen'],
        saturation_limits=[(0, 255), (0, 255), (0, 255)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2, arr3],
        colors=['betterBlue', 'betterOrange', 'betterGreen'],
        saturation_limits=[(0, 255), (0, 255), (0, 255)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_2d_u16_serial_vs_parallel():
    """
    Test that serial and parallel merge produce identical results for 2D u16 arrays
    """
    arr1 = np.random.randint(0, 65536, size=(512, 512), dtype=np.uint16)
    arr2 = np.random.randint(0, 65536, size=(512, 512), dtype=np.uint16)
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_2d_u16_serial_vs_parallel_three_channels():
    """
    Test that serial and parallel merge produce identical results for 2D u16 arrays with three
    channels
    """
    arr1 = np.random.randint(0, 65536, size=(512, 512), dtype=np.uint16)
    arr2 = np.random.randint(0, 65536, size=(512, 512), dtype=np.uint16)
    arr3 = np.random.randint(0, 65536, size=(512, 512), dtype=np.uint16)
    serial_result = mc.merge(
        arrs=[arr1, arr2, arr3],
        colors=['betterBlue', 'betterOrange', 'betterGreen'],
        saturation_limits=[(100, 40000), (100, 40000), (100, 40000)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2, arr3],
        colors=['betterBlue', 'betterOrange', 'betterGreen'],
        saturation_limits=[(100, 40000), (100, 40000), (100, 40000)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_3d_u16_serial_vs_parallel():
    """
    Test that serial and parallel merge produce identical results for 3D u16 arrays
    """
    arr1 = np.random.randint(0, 65536, size=(50, 256, 256), dtype=np.uint16)
    arr2 = np.random.randint(0, 65536, size=(50, 256, 256), dtype=np.uint16)
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_merge_3d_u16_serial_vs_parallel_three_channels():
    """
    Test that serial and parallel merge produce identical results for 3D u16 arrays with three
    channels
    """
    arr1 = np.random.randint(0, 65536, size=(50, 256, 256), dtype=np.uint16)
    arr2 = np.random.randint(0, 65536, size=(50, 256, 256), dtype=np.uint16)
    arr3 = np.random.randint(0, 65536, size=(50, 256, 256), dtype=np.uint16)
    serial_result = mc.merge(
        arrs=[arr1, arr2, arr3],
        colors=['betterBlue', 'betterOrange', 'betterGreen'],
        saturation_limits=[(100, 40000), (100, 40000), (100, 40000)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2, arr3],
        colors=['betterBlue', 'betterOrange', 'betterGreen'],
        saturation_limits=[(100, 40000), (100, 40000), (100, 40000)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


def test_get_cmap_array_shape_and_dtype():
    """
    Test that get_cmap_array returns correct shape and dtype
    """
    cmap_arr = mc.get_cmap_array('betterBlue')
    assert cmap_arr.shape == (256, 3)
    assert cmap_arr.dtype == np.uint8


def test_get_cmap_array_values():
    """
    Test that get_cmap_array returns expected values for a known colormap
    """
    cmap_arr = mc.get_cmap_array('betterBlue')
    # First value should be black/near-black
    assert np.allclose(cmap_arr[0], [0, 0, 0])
    # Last value should match known betterBlue end color
    assert np.allclose(cmap_arr[255], [0, 188, 254])


def test_get_cmap_array_invalid_name():
    """
    Test that get_cmap_array raises ValueError for invalid colormap name
    """
    with pytest.raises(ValueError) as exc_info:
        mc.get_cmap_array('invalid_colormap_name')
    assert 'invalid_colormap_name' in str(exc_info.value)
    assert 'COLORMAPS' in str(exc_info.value)


def test_get_cmap_array_all_colormaps():
    """
    Test that get_cmap_array works for all available colormaps
    """
    from typing import get_args

    from mergechannels._luts import COLORMAPS as COLORMAPS_TYPE

    all_cmap_names = get_args(COLORMAPS_TYPE)
    for name in all_cmap_names:
        cmap_arr = mc.get_cmap_array(name)
        assert cmap_arr.shape == (256, 3), f'Failed for colormap: {name}'
        assert cmap_arr.dtype == np.uint8, f'Failed for colormap: {name}'


def test_get_cmap_array_consistency_with_apply():
    """
    Test that get_cmap_array returns values consistent with apply_color_map
    """
    # Create a simple test array with values 0-255
    test_arr = np.arange(256, dtype=np.uint8).reshape(1, 256)

    # Get the colormap array
    cmap_arr = mc.get_cmap_array('betterOrange')

    # Apply colormap to the test array
    result = mc.apply_color_map(test_arr, 'betterOrange', saturation_limits=(0, 255))

    # The result should match the colormap array (accounting for shape)
    assert np.array_equal(result[0], cmap_arr)


def test_get_mpl_cmap_returns_listed_colormap():
    """
    Test that get_mpl_cmap returns a matplotlib ListedColormap
    """
    from matplotlib.colors import ListedColormap

    cmap = mc.get_mpl_cmap('betterBlue')
    assert isinstance(cmap, ListedColormap)
    assert cmap.name == 'betterBlue'
    assert cmap.N == 256


def test_get_mpl_cmap_colors_match_array():
    """
    Test that get_mpl_cmap returns colors matching get_cmap_array
    """
    cmap = mc.get_mpl_cmap('betterOrange')
    arr = mc.get_cmap_array('betterOrange')

    # Convert matplotlib colors (0-1 float) to uint8 (0-255)
    mpl_colors = (cmap.colors * 255).astype(np.uint8)
    assert np.array_equal(arr, mpl_colors)


def test_get_mpl_cmap_invalid_name():
    """
    Test that get_mpl_cmap raises ValueError for invalid colormap name
    """
    with pytest.raises(ValueError) as exc_info:
        mc.get_mpl_cmap('invalid_colormap_name')
    assert 'invalid_colormap_name' in str(exc_info.value)


def test_colormapping_matches():
    """
    Test that the cmap returned by get_mpl_cmap colormaps an image identically to mergechannels
    """
    mpl_cmap = mc.get_mpl_cmap('betterBlue')
    data = np.arange(256).reshape(16, 16).astype(np.uint8)
    # matplotlib expects normalized (0-1) input, so we divide by 255
    mpl_res = mpl_cmap(data / 255.0)
    assert mpl_res.shape == (16, 16, 4)  # RGBA output
    assert mpl_res.max() == 1.0
    # matplotlib creates floating point arrays, we create uint8 arrays
    mpl_res_rgb = (mpl_res[:, :, :3] * 255).astype('uint8')
    # use explicit saturation limits to avoid autoscaling differences
    mc_res_rgb = mc.apply_color_map(data, 'betterBlue', saturation_limits=(0, 255))
    np.testing.assert_array_equal(mc_res_rgb, mpl_res_rgb)
