"""
Benchmark comparison between original colorize implementation and v2 refactored version.

Run with: pytest pytests/test_bench_v1_vs_v2.py --benchmark-only --benchmark-group-by=group -v
"""

import numpy as np
import pytest
from mergechannels.mergechannels import (
    dispatch_multi_channel,
    dispatch_multi_channel_v2,
    dispatch_single_channel,
    dispatch_single_channel_v2,
)

# ============================================================================
# Fixtures for creating test arrays
# ============================================================================


@pytest.fixture
def small_2d_u8():
    """Create a small 2D u8 array (256x256)."""
    return np.random.randint(0, 256, (256, 256), dtype=np.uint8)


@pytest.fixture
def large_2d_u8():
    """Create a large 2D u8 array (1024x1024)."""
    return np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)


@pytest.fixture
def small_2d_u16():
    """Create a small 2D u16 array (256x256)."""
    return np.random.randint(0, 65536, (256, 256), dtype=np.uint16)


@pytest.fixture
def large_2d_u16():
    """Create a large 2D u16 array (1024x1024)."""
    return np.random.randint(0, 65536, (1024, 1024), dtype=np.uint16)


@pytest.fixture
def small_3d_u8():
    """Create a small 3D u8 array (10x256x256)."""
    return np.random.randint(0, 256, (10, 256, 256), dtype=np.uint8)


@pytest.fixture
def medium_3d_u8():
    """Create a medium 3D u8 array (10x512x512)."""
    return np.random.randint(0, 256, (10, 512, 512), dtype=np.uint8)


# ============================================================================
# Single Channel 2D u8 Benchmarks - Fast Path (0-255 limits)
# ============================================================================


@pytest.mark.benchmark(group='colorize_2d_u8_fast_256')
def test_colorize_2d_u8_fast_256_v1(benchmark, small_2d_u8):
    """Original implementation - 2D u8 fast path (256x256)"""
    result = benchmark(
        dispatch_single_channel,
        array_reference=small_2d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 255.0],
        parallel=True,
    )
    assert result.shape == (256, 256, 3)


@pytest.mark.benchmark(group='colorize_2d_u8_fast_256')
def test_colorize_2d_u8_fast_256_v2(benchmark, small_2d_u8):
    """V2 implementation - 2D u8 fast path (256x256)"""
    result = benchmark(
        dispatch_single_channel_v2,
        array_reference=small_2d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 255.0],
        parallel=True,
    )
    assert result.shape == (256, 256, 3)


@pytest.mark.benchmark(group='colorize_2d_u8_fast_1024')
def test_colorize_2d_u8_fast_1024_v1(benchmark, large_2d_u8):
    """Original implementation - 2D u8 fast path (1024x1024)"""
    result = benchmark(
        dispatch_single_channel,
        array_reference=large_2d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 255.0],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


@pytest.mark.benchmark(group='colorize_2d_u8_fast_1024')
def test_colorize_2d_u8_fast_1024_v2(benchmark, large_2d_u8):
    """V2 implementation - 2D u8 fast path (1024x1024)"""
    result = benchmark(
        dispatch_single_channel_v2,
        array_reference=large_2d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 255.0],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


# ============================================================================
# Single Channel 2D u8 Benchmarks - Normalize Path
# ============================================================================


@pytest.mark.benchmark(group='colorize_2d_u8_norm_1024')
def test_colorize_2d_u8_norm_1024_v1(benchmark, large_2d_u8):
    """Original implementation - 2D u8 normalize path (1024x1024)"""
    result = benchmark(
        dispatch_single_channel,
        array_reference=large_2d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[10.0, 245.0],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


@pytest.mark.benchmark(group='colorize_2d_u8_norm_1024')
def test_colorize_2d_u8_norm_1024_v2(benchmark, large_2d_u8):
    """V2 implementation - 2D u8 normalize path (1024x1024)"""
    result = benchmark(
        dispatch_single_channel_v2,
        array_reference=large_2d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[10.0, 245.0],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


# ============================================================================
# Single Channel 2D u16 Benchmarks
# ============================================================================


@pytest.mark.benchmark(group='colorize_2d_u16_1024')
def test_colorize_2d_u16_1024_v1(benchmark, large_2d_u16):
    """Original implementation - 2D u16 (1024x1024)"""
    result = benchmark(
        dispatch_single_channel,
        array_reference=large_2d_u16,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 65535.0],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


@pytest.mark.benchmark(group='colorize_2d_u16_1024')
def test_colorize_2d_u16_1024_v2(benchmark, large_2d_u16):
    """V2 implementation - 2D u16 (1024x1024)"""
    result = benchmark(
        dispatch_single_channel_v2,
        array_reference=large_2d_u16,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 65535.0],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


# ============================================================================
# Single Channel 3D u8 Benchmarks - Fast Path
# ============================================================================


@pytest.mark.benchmark(group='colorize_3d_u8_fast_10x256')
def test_colorize_3d_u8_fast_10x256_v1(benchmark, small_3d_u8):
    """Original implementation - 3D u8 fast path (10x256x256)"""
    result = benchmark(
        dispatch_single_channel,
        array_reference=small_3d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 255.0],
        parallel=True,
    )
    assert result.shape == (10, 256, 256, 3)


@pytest.mark.benchmark(group='colorize_3d_u8_fast_10x256')
def test_colorize_3d_u8_fast_10x256_v2(benchmark, small_3d_u8):
    """V2 implementation - 3D u8 fast path (10x256x256)"""
    result = benchmark(
        dispatch_single_channel_v2,
        array_reference=small_3d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 255.0],
        parallel=True,
    )
    assert result.shape == (10, 256, 256, 3)


@pytest.mark.benchmark(group='colorize_3d_u8_fast_10x512')
def test_colorize_3d_u8_fast_10x512_v1(benchmark, medium_3d_u8):
    """Original implementation - 3D u8 fast path (10x512x512)"""
    result = benchmark(
        dispatch_single_channel,
        array_reference=medium_3d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 255.0],
        parallel=True,
    )
    assert result.shape == (10, 512, 512, 3)


@pytest.mark.benchmark(group='colorize_3d_u8_fast_10x512')
def test_colorize_3d_u8_fast_10x512_v2(benchmark, medium_3d_u8):
    """V2 implementation - 3D u8 fast path (10x512x512)"""
    result = benchmark(
        dispatch_single_channel_v2,
        array_reference=medium_3d_u8,
        cmap_name='Grays',
        cmap_values=None,
        limits=[0.0, 255.0],
        parallel=True,
    )
    assert result.shape == (10, 512, 512, 3)


# ============================================================================
# Merge 2D u8 Benchmarks - Fast Path
# ============================================================================


@pytest.mark.benchmark(group='merge_2d_u8_fast_1024')
def test_merge_2d_u8_fast_1024_v1(benchmark, large_2d_u8):
    """Original implementation - merge 2D u8 fast path (1024x1024)"""
    arr1 = large_2d_u8
    arr2 = large_2d_u8.copy()
    result = benchmark(
        dispatch_multi_channel,
        array_references=[arr1, arr2],
        cmap_names=['Grays', 'I Red'],
        cmap_values=[None, None],
        blending='max',
        limits=[[0.0, 255.0], [0.0, 255.0]],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


@pytest.mark.benchmark(group='merge_2d_u8_fast_1024')
def test_merge_2d_u8_fast_1024_v2(benchmark, large_2d_u8):
    """V2 implementation - merge 2D u8 fast path (1024x1024)"""
    arr1 = large_2d_u8
    arr2 = large_2d_u8.copy()
    result = benchmark(
        dispatch_multi_channel_v2,
        array_references=[arr1, arr2],
        cmap_names=['Grays', 'I Red'],
        cmap_values=[None, None],
        blending='max',
        limits=[[0.0, 255.0], [0.0, 255.0]],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


# ============================================================================
# Merge 2D u8 Benchmarks - Normalize Path
# ============================================================================


@pytest.mark.benchmark(group='merge_2d_u8_norm_1024')
def test_merge_2d_u8_norm_1024_v1(benchmark, large_2d_u8):
    """Original implementation - merge 2D u8 normalize path (1024x1024)"""
    arr1 = large_2d_u8
    arr2 = large_2d_u8.copy()
    result = benchmark(
        dispatch_multi_channel,
        array_references=[arr1, arr2],
        cmap_names=['Grays', 'I Red'],
        cmap_values=[None, None],
        blending='max',
        limits=[[10.0, 245.0], [10.0, 245.0]],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


@pytest.mark.benchmark(group='merge_2d_u8_norm_1024')
def test_merge_2d_u8_norm_1024_v2(benchmark, large_2d_u8):
    """V2 implementation - merge 2D u8 normalize path (1024x1024)"""
    arr1 = large_2d_u8
    arr2 = large_2d_u8.copy()
    result = benchmark(
        dispatch_multi_channel_v2,
        array_references=[arr1, arr2],
        cmap_names=['Grays', 'I Red'],
        cmap_values=[None, None],
        blending='max',
        limits=[[10.0, 245.0], [10.0, 245.0]],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


# ============================================================================
# Merge 2D u16 Benchmarks
# ============================================================================


@pytest.mark.benchmark(group='merge_2d_u16_1024')
def test_merge_2d_u16_1024_v1(benchmark, large_2d_u16):
    """Original implementation - merge 2D u16 (1024x1024)"""
    arr1 = large_2d_u16
    arr2 = large_2d_u16.copy()
    result = benchmark(
        dispatch_multi_channel,
        array_references=[arr1, arr2],
        cmap_names=['Grays', 'I Red'],
        cmap_values=[None, None],
        blending='max',
        limits=[[0.0, 65535.0], [0.0, 65535.0]],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


@pytest.mark.benchmark(group='merge_2d_u16_1024')
def test_merge_2d_u16_1024_v2(benchmark, large_2d_u16):
    """V2 implementation - merge 2D u16 (1024x1024)"""
    arr1 = large_2d_u16
    arr2 = large_2d_u16.copy()
    result = benchmark(
        dispatch_multi_channel_v2,
        array_references=[arr1, arr2],
        cmap_names=['Grays', 'I Red'],
        cmap_values=[None, None],
        blending='max',
        limits=[[0.0, 65535.0], [0.0, 65535.0]],
        parallel=True,
    )
    assert result.shape == (1024, 1024, 3)


# ============================================================================
# Merge 3D u8 Benchmarks - Fast Path
# ============================================================================


@pytest.mark.benchmark(group='merge_3d_u8_fast_10x256')
def test_merge_3d_u8_fast_10x256_v1(benchmark, small_3d_u8):
    """Original implementation - merge 3D u8 fast path (10x256x256)"""
    arr1 = small_3d_u8
    arr2 = small_3d_u8.copy()
    result = benchmark(
        dispatch_multi_channel,
        array_references=[arr1, arr2],
        cmap_names=['Grays', 'I Red'],
        cmap_values=[None, None],
        blending='max',
        limits=[[0.0, 255.0], [0.0, 255.0]],
        parallel=True,
    )
    assert result.shape == (10, 256, 256, 3)


@pytest.mark.benchmark(group='merge_3d_u8_fast_10x256')
def test_merge_3d_u8_fast_10x256_v2(benchmark, small_3d_u8):
    """V2 implementation - merge 3D u8 fast path (10x256x256)"""
    arr1 = small_3d_u8
    arr2 = small_3d_u8.copy()
    result = benchmark(
        dispatch_multi_channel_v2,
        array_references=[arr1, arr2],
        cmap_names=['Grays', 'I Red'],
        cmap_values=[None, None],
        blending='max',
        limits=[[0.0, 255.0], [0.0, 255.0]],
        parallel=True,
    )
    assert result.shape == (10, 256, 256, 3)
