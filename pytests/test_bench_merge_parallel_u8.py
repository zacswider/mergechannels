import mergechannels as mc
import numpy as np
import pytest


@pytest.mark.benchmark(group='small merge linear vs parallel no autoscale')
def test_bench_small_merge_u8_serial_no_autoscale(benchmark, small_array_u8) -> None:
    """
    benchmark merging two small 2d u8 arrays using serial approach
    """
    arr1 = small_array_u8
    arr2 = np.random.randn(256, 256).astype('uint8')
    serial_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='small merge linear vs parallel no autoscale')
def test_bench_small_merge_u8_parallel_no_autoscale(benchmark, small_array_u8) -> None:
    """
    benchmark merging two small 2d u8 arrays using parallel approach
    """
    arr1 = small_array_u8
    arr2 = np.random.randn(256, 256).astype('uint8')
    parallel_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=True,
    )
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='small merge linear vs parallel with autoscale')
def test_bench_small_merge_u8_serial_autoscale(benchmark, small_array_u8) -> None:
    """
    benchmark merging two small 2d u8 arrays using serial approach
    """
    arr1 = small_array_u8
    arr2 = np.random.randn(256, 256).astype('uint8')
    serial_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='small merge linear vs parallel with autoscale')
def test_bench_small_merge_u8_parallel_autoscale(benchmark, small_array_u8) -> None:
    """
    benchmark merging two small 2d u8 arrays using parallel approach
    """
    arr1 = small_array_u8
    arr2 = np.random.randn(256, 256).astype('uint8')
    parallel_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=True,
    )
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium merge linear vs parallel no autoscale')
def test_bench_medium_merge_u8_serial_no_autoscale(benchmark, medium_array_u8) -> None:
    """
    benchmark merging two medium 2d u8 arrays using serial approach
    """
    arr1 = medium_array_u8
    arr2 = np.random.randn(512, 512).astype('uint8')
    serial_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium merge linear vs parallel no autoscale')
def test_bench_medium_merge_u8_parallel_no_autoscale(benchmark, medium_array_u8) -> None:
    """
    benchmark merging two medium 2d u8 arrays using parallel approach
    """
    arr1 = medium_array_u8
    arr2 = np.random.randn(512, 512).astype('uint8')
    parallel_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=True,
    )
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium merge linear vs parallel with autoscale')
def test_bench_medium_merge_u8_serial_autoscale(benchmark, medium_array_u8) -> None:
    """
    benchmark merging two medium 2d u8 arrays using serial approach
    """
    arr1 = medium_array_u8
    arr2 = np.random.randn(512, 512).astype('uint8')
    serial_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='medium merge linear vs parallel with autoscale')
def test_bench_medium_merge_u8_parallel_autoscale(benchmark, medium_array_u8) -> None:
    """
    benchmark merging two medium 2d u8 arrays using parallel approach
    """
    arr1 = medium_array_u8
    arr2 = np.random.randn(512, 512).astype('uint8')
    parallel_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=True,
    )
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large merge linear vs parallel no autoscale')
def test_bench_large_merge_u8_serial_no_autoscale(benchmark, large_array_u8) -> None:
    """
    benchmark merging two large 2d u8 arrays using serial approach
    """
    arr1 = large_array_u8
    arr2 = np.random.randn(1024, 1024).astype('uint8')
    serial_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large merge linear vs parallel no autoscale')
def test_bench_large_merge_u8_parallel_no_autoscale(benchmark, large_array_u8) -> None:
    """
    benchmark merging two large 2d u8 arrays using parallel approach
    """
    arr1 = large_array_u8
    arr2 = np.random.randn(1024, 1024).astype('uint8')
    parallel_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=True,
    )
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(0, 255), (0, 255)],
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large merge linear vs parallel with autoscale')
def test_bench_large_merge_u8_serial_autoscale(benchmark, large_array_u8) -> None:
    """
    benchmark merging two large 2d u8 arrays using serial approach
    """
    arr1 = large_array_u8
    arr2 = np.random.randn(1024, 1024).astype('uint8')
    serial_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=False,
    )
    parallel_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=True,
    )
    assert np.array_equal(serial_result, parallel_result)


@pytest.mark.benchmark(group='large merge linear vs parallel with autoscale')
def test_bench_large_merge_u8_parallel_autoscale(benchmark, large_array_u8) -> None:
    """
    benchmark merging two large 2d u8 arrays using parallel approach
    """
    arr1 = large_array_u8
    arr2 = np.random.randn(1024, 1024).astype('uint8')
    parallel_result = benchmark(
        mc.merge,
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=True,
    )
    serial_result = mc.merge(
        arrs=[arr1, arr2],
        colors=['betterBlue', 'betterOrange'],
        saturation_limits=[(10, 200), (10, 200)],
        parallel=False,
    )
    assert np.array_equal(serial_result, parallel_result)
