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


@pytest.fixture
def small_array_u8() -> np.ndarray:
    """Create a small u8 array for benchmarking"""
    return np.random.randn(256, 256).astype('uint8')


@pytest.fixture
def small_3d_array_u8() -> np.ndarray:
    """Create a small 3d u8 array for benchmarking"""
    return np.random.randn(50, 256, 256).astype('uint8')


@pytest.fixture
def small_array_u16() -> np.ndarray:
    """Create a small u16 array for benchmarking"""
    return np.random.randn(256, 256).astype('uint16')


@pytest.fixture
def medium_array_u8() -> np.ndarray:
    """Create a medium u8 array for benchmarking"""
    return np.random.randn(512, 512).astype('uint8')


@pytest.fixture
def medium_3d_array_u8() -> np.ndarray:
    """Create a medium 3d u8 array for benchmarking"""
    return np.random.randn(50, 512, 512).astype('uint8')


@pytest.fixture
def medium_array_u16() -> np.ndarray:
    """Create a medium u16 array for benchmarking"""
    return np.random.randn(512, 512).astype('uint16')


@pytest.fixture
def large_array_u8() -> np.ndarray:
    """Create a large u8 array for benchmarking"""
    return np.random.randn(1024, 1024).astype('uint8')


@pytest.fixture
def large_3d_array_u8() -> np.ndarray:
    """Create a large 3d u8 array for benchmarking"""
    return np.random.randn(50, 1024, 1024).astype('uint8')


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


@pytest.fixture
def small_3d_array_u16() -> np.ndarray:
    """Create a small 3d u16 array for benchmarking"""
    return np.random.randn(50, 256, 256).astype('uint16')


@pytest.fixture
def medium_3d_array_u16() -> np.ndarray:
    """Create a medium 3d u16 array for benchmarking"""
    return np.random.randn(50, 512, 512).astype('uint16')


@pytest.fixture
def large_3d_array_u16() -> np.ndarray:
    """Create a large 3d u16 array for benchmarking"""
    return np.random.randn(50, 1024, 1024).astype('uint16')
