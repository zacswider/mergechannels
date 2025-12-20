import mergechannels as mc
import numpy as np
import pytest


@pytest.mark.benchmark(group='small linear vs parallel no autoscale')
def test_bench_small_u8_serial_no_autoscale(benchmark, small_array_u8) -> None:
    """Benchmark serial colorization for small u8 array without autoscaling"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='small linear vs parallel no autoscale')
def test_bench_small_u8_parallel_no_autoscale(benchmark, small_array_u8) -> None:
    """Benchmark parallel colorization for small u8 array without autoscaling"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='small linear vs parallel with autoscale')
def test_bench_small_u8_serial_autoscale(benchmark, small_array_u8) -> None:
    """Benchmark serial colorization for small u8 array with autoscaling"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='small linear vs parallel with autoscale')
def test_bench_small_u8_parallel_autoscale(benchmark, small_array_u8) -> None:
    """Benchmark parallel colorization for small u8 array with autoscaling"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=small_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium linear vs parallel no autoscale')
def test_bench_medium_u8_serial_no_autoscale(benchmark, medium_array_u8) -> None:
    """Benchmark serial colorization for medium u8 array without autoscaling"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium linear vs parallel no autoscale')
def test_bench_medium_u8_parallel_no_autoscale(benchmark, medium_array_u8) -> None:
    """Benchmark parallel colorization for medium u8 array without autoscaling"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium linear vs parallel with autoscale')
def test_bench_medium_u8_serial_autoscale(benchmark, medium_array_u8) -> None:
    """Benchmark serial colorization for medium u8 array with autoscaling"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium linear vs parallel with autoscale')
def test_bench_medium_u8_parallel_autoscale(benchmark, medium_array_u8) -> None:
    """Benchmark parallel colorization for medium u8 array with autoscaling"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=medium_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large linear vs parallel no autoscale')
def test_bench_large_u8_serial_no_autoscale(benchmark, large_array_u8) -> None:
    """Benchmark serial colorization for large u8 array without autoscaling"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large linear vs parallel no autoscale')
def test_bench_large_u8_parallel_no_autoscale(benchmark, large_array_u8) -> None:
    """Benchmark parallel colorization for large u8 array without autoscaling"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large linear vs parallel with autoscale')
def test_bench_large_u8_serial_autoscale(benchmark, large_array_u8) -> None:
    """Benchmark serial colorization for large u8 array with autoscaling"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large linear vs parallel with autoscale')
def test_bench_large_u8_parallel_autoscale(benchmark, large_array_u8) -> None:
    """Benchmark parallel colorization for large u8 array with autoscaling"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=large_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='xlarge linear vs parallel no autoscale')
def test_bench_xlarge_u8_serial_no_autoscale(benchmark, xlarge_array_u8) -> None:
    """Benchmark serial colorization for xlarge u8 array without autoscaling"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='xlarge linear vs parallel no autoscale')
def test_bench_xlarge_u8_parallel_no_autoscale(benchmark, xlarge_array_u8) -> None:
    """Benchmark parallel colorization for xlarge u8 array without autoscaling"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='xlarge linear vs parallel with autoscale')
def test_bench_xlarge_u8_serial_autoscale(benchmark, xlarge_array_u8) -> None:
    """Benchmark serial colorization for xlarge u8 array with autoscaling"""
    serial_result = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    parallel_result = mc.apply_color_map(
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='xlarge linear vs parallel with autoscale')
def test_bench_xlarge_u8_parallel_autoscale(benchmark, xlarge_array_u8) -> None:
    """Benchmark parallel colorization for xlarge u8 array with autoscaling"""
    parallel_result = benchmark(
        mc.apply_color_map,
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=True,
    )
    serial_result = mc.apply_color_map(
        arr=xlarge_array_u8,
        color='Grays',
        saturation_limits=(10, 200),
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)
