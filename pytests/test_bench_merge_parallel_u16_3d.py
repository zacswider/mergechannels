import mergechannels as mc
import numpy as np
import pytest


@pytest.mark.benchmark(group='small 3d u16 merge linear vs parallel')
def test_bench_small_3d_merge_u16_serial(benchmark, small_3d_array_u16) -> None:
    """
    benchmark merging two small 3d u16 arrays using serial approach
    """
    arr1 = small_3d_array_u16
    arr2 = np.random.randn(50, 256, 256).astype('uint16')
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


@pytest.mark.benchmark(group='small 3d u16 merge linear vs parallel')
def test_bench_small_3d_merge_u16_parallel(benchmark, small_3d_array_u16) -> None:
    """
    benchmark merging two small 3d u16 arrays using parallel approach
    """
    arr1 = small_3d_array_u16
    arr2 = np.random.randn(50, 256, 256).astype('uint16')
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


@pytest.mark.benchmark(group='medium 3d u16 merge linear vs parallel')
def test_bench_medium_3d_merge_u16_serial(benchmark, medium_3d_array_u16) -> None:
    """
    benchmark merging two medium 3d u16 arrays using serial approach
    """
    arr1 = medium_3d_array_u16
    arr2 = np.random.randn(50, 512, 512).astype('uint16')
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


@pytest.mark.benchmark(group='medium 3d u16 merge linear vs parallel')
def test_bench_medium_3d_merge_u16_parallel(benchmark, medium_3d_array_u16) -> None:
    """
    benchmark merging two medium 3d u16 arrays using parallel approach
    """
    arr1 = medium_3d_array_u16
    arr2 = np.random.randn(50, 512, 512).astype('uint16')
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


@pytest.mark.benchmark(group='large 3d u16 merge linear vs parallel')
def test_bench_large_3d_merge_u16_serial(benchmark, large_3d_array_u16) -> None:
    """
    benchmark merging two large 3d u16 arrays using serial approach
    """
    arr1 = large_3d_array_u16
    arr2 = np.random.randn(50, 1024, 1024).astype('uint16')
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


@pytest.mark.benchmark(group='large 3d u16 merge linear vs parallel')
def test_bench_large_3d_merge_u16_parallel(benchmark, large_3d_array_u16) -> None:
    """
    benchmark merging two large 3d u16 arrays using parallel approach
    """
    arr1 = large_3d_array_u16
    arr2 = np.random.randn(50, 1024, 1024).astype('uint16')
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
