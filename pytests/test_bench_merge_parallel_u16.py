import mergechannels as mc
import numpy as np
import pytest


@pytest.mark.benchmark(group='small 2d u16 merge linear vs parallel')
def test_bench_small_merge_u16_serial(benchmark, small_array_u16) -> None:
    """
    benchmark merging two small u16 arrays using serial approach
    """
    arr1 = small_array_u16
    arr2 = np.random.randn(256, 256).astype('uint16')
    serial_result = benchmark(
        mc.merge,
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


@pytest.mark.benchmark(group='small 2d u16 merge linear vs parallel')
def test_bench_small_merge_u16_parallel(benchmark, small_array_u16) -> None:
    """
    benchmark merging two small u16 arrays using parallel approach
    """
    arr1 = small_array_u16
    arr2 = np.random.randn(256, 256).astype('uint16')
    parallel_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=True,
    )
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium 2d u16 merge linear vs parallel')
def test_bench_medium_merge_u16_serial(benchmark, medium_array_u16) -> None:
    """
    benchmark merging two medium u16 arrays using serial approach
    """
    arr1 = medium_array_u16
    arr2 = np.random.randn(512, 512).astype('uint16')
    serial_result = benchmark(
        mc.merge,
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


@pytest.mark.benchmark(group='medium 2d u16 merge linear vs parallel')
def test_bench_medium_merge_u16_parallel(benchmark, medium_array_u16) -> None:
    """
    benchmark merging two medium u16 arrays using parallel approach
    """
    arr1 = medium_array_u16
    arr2 = np.random.randn(512, 512).astype('uint16')
    parallel_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=True,
    )
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large 2d u16 merge linear vs parallel')
def test_bench_large_merge_u16_serial(benchmark, large_array_u16) -> None:
    """
    benchmark merging two large u16 arrays using serial approach
    """
    arr1 = large_array_u16
    arr2 = np.random.randn(1024, 1024).astype('uint16')
    serial_result = benchmark(
        mc.merge,
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


@pytest.mark.benchmark(group='large 2d u16 merge linear vs parallel')
def test_bench_large_merge_u16_parallel(benchmark, large_array_u16) -> None:
    """
    benchmark merging two large u16 arrays using parallel approach
    """
    arr1 = large_array_u16
    arr2 = np.random.randn(1024, 1024).astype('uint16')
    parallel_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=True,
    )
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(100, 40000), (100, 40000)],
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)
