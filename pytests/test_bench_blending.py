import mergechannels as mc
import numpy as np
import pytest


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
