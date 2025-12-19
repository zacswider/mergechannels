import mergechannels as mc
import numpy as np
import pytest


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


def mc_merge_u8_with_cmaps_no_autoscale(
    arrs: list[np.ndarray],
    cmaps: list[np.ndarray],
) -> np.ndarray:
    """
    merge some number of arrays using mergechannels operations
    """
    return mc.merge(
        arrs=arrs,
        colors=cmaps,  # type: ignore
        blending='max',
        saturation_limits=[(0.0, 255)] * len(arrs),
    )


def mc_merge_u8_with_cmaps_autoscale(
    arrs: list[np.ndarray],
    cmaps: list[np.ndarray],
) -> np.ndarray:
    """
    merge some number of arrays using mergechannels operations
    """
    return mc.merge(
        arrs=arrs,
        colors=cmaps,  # type: ignore
        blending='max',
    )


@pytest.mark.benchmark(group='merge colormaps small with autoscale')
def test_merge_u8_matplotlib_small_autoscale(
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
    mc_merged = mc_merge_u8_with_cmaps_autoscale(
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps small with autoscale')
def test_merge_u8_mergechannels_small_autoscale(
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
        mc_merge_u8_with_cmaps_autoscale,
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps medium with autoscale')
def test_merge_u8_matplotlib_medium_autoscale(
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
    mc_merged = mc_merge_u8_with_cmaps_autoscale(
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps medium with autoscale')
def test_merge_u8_mergechannels_medium_autoscale(
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
        mc_merge_u8_with_cmaps_autoscale,
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps large with autoscale')
def test_merge_u8_matplotlib_large_autoscale(
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
    mc_merged = mc_merge_u8_with_cmaps_autoscale(
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps large with autoscale')
def test_merge_u8_mergechannels_large_autoscale(
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
        mc_merge_u8_with_cmaps_autoscale,
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps xlarge with autoscale')
def test_merge_u8_matplotlib_xlarge_autoscale(
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
    mc_merged = mc_merge_u8_with_cmaps_autoscale(
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps xlarge with autoscale')
def test_merge_u8_mergechannels_xlarge_autoscale(
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
        mc_merge_u8_with_cmaps_autoscale,
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps small no autoscale')
def test_merge_u8_matplotlib_small_no_autoscale(
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
    mc_merged = mc_merge_u8_with_cmaps_no_autoscale(
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps small no autoscale')
def test_merge_u8_mergechannels_small_no_autoscale(
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
        mc_merge_u8_with_cmaps_no_autoscale,
        arrs=[small_array_u8, small_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps medium no autoscale')
def test_merge_u8_matplotlib_medium_no_autoscale(
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
    mc_merged = mc_merge_u8_with_cmaps_no_autoscale(
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps medium no autoscale')
def test_merge_u8_mergechannels_medium_no_autoscale(
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
        mc_merge_u8_with_cmaps_no_autoscale,
        arrs=[medium_array_u8, medium_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps large no autoscale')
def test_merge_u8_matplotlib_large_no_autoscale(
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
    mc_merged = mc_merge_u8_with_cmaps_no_autoscale(
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps large no autoscale')
def test_merge_u8_mergechannels_large_no_autoscale(
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
        mc_merge_u8_with_cmaps_no_autoscale,
        arrs=[large_array_u8, large_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps xlarge no autoscale')
def test_merge_u8_matplotlib_xlarge_no_autoscale(
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
    mc_merged = mc_merge_u8_with_cmaps_no_autoscale(
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)


@pytest.mark.benchmark(group='merge colormaps xlarge no autoscale')
def test_merge_u8_mergechannels_xlarge_no_autoscale(
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
        mc_merge_u8_with_cmaps_no_autoscale,
        arrs=[xlarge_array_u8, xlarge_array_u8_copy],
        cmaps=[mpl_greens_array_lut, mpl_reds_array_lut],
    )
    np.testing.assert_allclose(np_merged, mc_merged)
