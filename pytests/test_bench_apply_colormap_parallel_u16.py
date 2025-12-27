import mergechannels as mc
import numpy as np
import pytest


@pytest.mark.benchmark(group='small linear vs parallel')
def test_bench_small_u16_serial(benchmark, small_array_u16) -> None:
    """Benchmark serial colorization for small u16 array"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=small_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=small_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='small linear vs parallel')
def test_bench_small_u16_parallel(benchmark, small_array_u16) -> None:
    """Benchmark parallel colorization for small u16 array"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=small_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=small_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium linear vs parallel')
def test_bench_medium_u16_serial(benchmark, medium_array_u16) -> None:
    """Benchmark serial colorization for medium u16 array"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=medium_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=medium_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium linear vs parallel')
def test_bench_medium_u16_parallel(benchmark, medium_array_u16) -> None:
    """Benchmark parallel colorization for medium u16 array"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=medium_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=medium_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large linear vs parallel')
def test_bench_large_u16_serial(benchmark, large_array_u16) -> None:
    """Benchmark serial colorization for large u16 array"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=large_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=large_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large linear vs parallel')
def test_bench_large_u16_parallel(benchmark, large_array_u16) -> None:
    """Benchmark parallel colorization for large u16 array"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=large_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=large_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='xlarge linear vs parallel')
def test_bench_xlarge_u16_serial(benchmark, xlarge_array_u16) -> None:
    """Benchmark serial colorization for xlarge u16 array"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=xlarge_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='xlarge linear vs parallel')
def test_bench_xlarge_u16_parallel(benchmark, xlarge_array_u16) -> None:
    """Benchmark parallel colorization for xlarge u16 array"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=xlarge_array_u16,
        color='Grays',
        saturation_limits=(0, 2**16),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)
