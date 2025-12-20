import mergechannels as mc
import pytest


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
