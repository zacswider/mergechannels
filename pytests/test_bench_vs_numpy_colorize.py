import mergechannels as mc
import numpy as np
import pytest


@pytest.mark.benchmark(group='apply colormap small with autoscale')
def test_apply_cmap_u8_matplotlib_small_autoscale(
    benchmark, small_array_u8, matplotlib_viridis_cmap
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
        small_array_u8,
        axis=0,
    )
    rgb_mc = mc.apply_color_map(small_array_u8, matplotlib_viridis_cmap)
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='apply colormap small no autoscale')
def test_apply_cmap_u8_matplotlib_small_no_autoscale(
    benchmark, small_array_u8, matplotlib_viridis_cmap
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
        small_array_u8,
        axis=0,
    )
    rgb_mc = mc.apply_color_map(small_array_u8, matplotlib_viridis_cmap)
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='apply colormap small with autoscale')
def test_apply_cmap_u8_mergechannels_autoscale_small(
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


@pytest.mark.benchmark(group='apply colormap moderate with autoscale')
def test_apply_cmap_u8_matplotlib_moderate_autoscale(
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


@pytest.mark.benchmark(group='apply colormap moderate no autoscale')
def test_apply_cmap_u8_matplotlib_moderate_no_autoscale(
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


@pytest.mark.benchmark(group='apply colormap moderate with autoscale')
def test_apply_cmap_u8_mergechannels_autoscale_moderate(
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


@pytest.mark.benchmark(group='apply colormap large with autoscale')
def test_apply_cmap_u8_matplotlib_large_autoscale(
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


@pytest.mark.benchmark(group='apply colormap large no autoscale')
def test_apply_cmap_u8_matplotlib_large_no_autoscale(
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


@pytest.mark.benchmark(group='apply colormap large with autoscale')
def test_apply_cmap_u8_mergechannels_autoscale_large(
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


@pytest.mark.benchmark(group='apply colormap small no autoscale')
def test_apply_cmap_u8_mergechannels_no_autoscale_small(
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
        saturation_limits=(0, 255),
    )
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='apply colormap moderate no autoscale')
def test_apply_cmap_u8_mergechannels_no_autoscale_moderate(
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
        saturation_limits=(0, 255),
    )
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)


@pytest.mark.benchmark(group='apply colormap large no autoscale')
def test_apply_cmap_u8_mergechannels_no_autoscale_large(
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
        saturation_limits=(0, 255),
    )
    np.testing.assert_array_equal(rgb_mpl, rgb_mc)
