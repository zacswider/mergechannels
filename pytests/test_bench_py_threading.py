import time
from concurrent.futures import ThreadPoolExecutor

import mergechannels as mc
import numpy as np
import pytest


@pytest.fixture
def images_100x512x512_u8() -> list[np.ndarray]:
    """Create 100 512x512 uint8 images for benchmarking"""
    return [np.random.randint(0, 256, (512, 512), dtype='uint8') for _ in range(100)]


@pytest.fixture
def images_100x512x512_u16() -> list[np.ndarray]:
    """Create 100 512x512 uint16 images for benchmarking"""
    return [np.random.randint(0, 65536, (512, 512), dtype='uint16') for _ in range(100)]


@pytest.fixture
def images_10x175x85_u8() -> list[np.ndarray]:
    """Create 10 175x85 uint8 images for benchmarking (a realistic workload for web app)"""
    return [np.random.randint(0, 256, (175, 85), dtype='uint8') for _ in range(10)]


def apply_colormap_serial(images: list[np.ndarray], colormap: str = 'Grays') -> list[np.ndarray]:
    """Apply colormap to images in series"""
    results = []
    for img in images:
        results.append(mc.apply_color_map(arr=img, color=colormap, saturation_limits=None))
    return results


def apply_colormap_threadpool(
    images: list[np.ndarray], colormap: str = 'Grays', max_workers: int = 4
) -> list[np.ndarray]:
    """Apply colormap to images using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(mc.apply_color_map, arr=img, color=colormap, saturation_limits=None)
            for img in images
        ]
        results = [future.result() for future in futures]
    return results


@pytest.mark.benchmark(group='threading u8 serial vs parallel web app workload')
def test_bench_serial_u8_webapp_workload(benchmark, images_10x175x85_u8) -> None:
    """Benchmark serial colormap application on 10 175x85 images (a realistic web app workload)"""
    results = benchmark(apply_colormap_serial, images_10x175x85_u8)
    assert len(results) == 10
    assert all(r.shape[:2] == (175, 85) for r in results)


@pytest.mark.benchmark(group='threading u8 serial vs parallel web app workload')
def test_bench_threadpool_u8_webapp_workload(benchmark, images_10x175x85_u8) -> None:
    """
    Benchmark ThreadPoolExecutor (10 workers) colormap application on 10 175x85 images
    (a realistic web app workload)
    """
    results = benchmark(apply_colormap_threadpool, images_10x175x85_u8, max_workers=10)
    assert len(results) == 10
    assert all(r.shape[:2] == (175, 85) for r in results)


@pytest.mark.benchmark(group='threading u8 serial vs parallel')
def test_bench_serial_u8(benchmark, images_100x512x512_u8) -> None:
    """Benchmark serial colormap application on 100 512x512 uint8 images"""
    results = benchmark(apply_colormap_serial, images_100x512x512_u8)
    assert len(results) == 100
    assert all(r.shape[:2] == (512, 512) for r in results)


@pytest.mark.benchmark(group='threading u8 serial vs parallel')
def test_bench_threadpool_2workers_u8(benchmark, images_100x512x512_u8) -> None:
    """Benchmark ThreadPoolExecutor (2 workers) colormap application on 100 512x512 uint8 images"""
    results = benchmark(apply_colormap_threadpool, images_100x512x512_u8, max_workers=2)
    assert len(results) == 100
    assert all(r.shape[:2] == (512, 512) for r in results)


@pytest.mark.benchmark(group='threading u8 serial vs parallel')
def test_bench_threadpool_4workers_u8(benchmark, images_100x512x512_u8) -> None:
    """Benchmark ThreadPoolExecutor (4 workers) colormap application on 100 512x512 uint8 images"""
    results = benchmark(apply_colormap_threadpool, images_100x512x512_u8, max_workers=4)
    assert len(results) == 100
    assert all(r.shape[:2] == (512, 512) for r in results)


@pytest.mark.benchmark(group='threading u8 serial vs parallel')
def test_bench_threadpool_8workers_u8(benchmark, images_100x512x512_u8) -> None:
    """Benchmark ThreadPoolExecutor (8 workers) colormap application on 100 512x512 uint8 images"""
    results = benchmark(apply_colormap_threadpool, images_100x512x512_u8, max_workers=8)
    assert len(results) == 100
    assert all(r.shape[:2] == (512, 512) for r in results)


@pytest.mark.benchmark(group='threading u16 serial vs parallel')
def test_bench_serial_u16(benchmark, images_100x512x512_u16) -> None:
    """Benchmark serial colormap application on 100 512x512 uint16 images"""
    results = benchmark(apply_colormap_serial, images_100x512x512_u16)
    assert len(results) == 100
    assert all(r.shape[:2] == (512, 512) for r in results)


@pytest.mark.benchmark(group='threading u16 serial vs parallel')
def test_bench_threadpool_2workers_u16(benchmark, images_100x512x512_u16) -> None:
    """Benchmark ThreadPoolExecutor (2 workers) colormap application on 100 512x512 uint16 images"""
    results = benchmark(apply_colormap_threadpool, images_100x512x512_u16, max_workers=2)
    assert len(results) == 100
    assert all(r.shape[:2] == (512, 512) for r in results)


@pytest.mark.benchmark(group='threading u16 serial vs parallel')
def test_bench_threadpool_4workers_u16(benchmark, images_100x512x512_u16) -> None:
    """Benchmark ThreadPoolExecutor (4 workers) colormap application on 100 512x512 uint16 images"""
    results = benchmark(apply_colormap_threadpool, images_100x512x512_u16, max_workers=4)
    assert len(results) == 100
    assert all(r.shape[:2] == (512, 512) for r in results)


@pytest.mark.benchmark(group='threading u16 serial vs parallel')
def test_bench_threadpool_8workers_u16(benchmark, images_100x512x512_u16) -> None:
    """Benchmark ThreadPoolExecutor (8 workers) colormap application on 100 512x512 uint16 images"""
    results = benchmark(apply_colormap_threadpool, images_100x512x512_u16, max_workers=8)
    assert len(results) == 100
    assert all(r.shape[:2] == (512, 512) for r in results)


# Non-benchmark timing tests for manual inspection
def test_timing_comparison_u8(images_100x512x512_u8) -> None:
    """Manual timing comparison for serial vs threadpool execution (uint8)"""
    # Serial
    start = time.perf_counter()
    serial_results = apply_colormap_serial(images_100x512x512_u8)
    serial_time = time.perf_counter() - start

    # ThreadPoolExecutor with 4 workers
    start = time.perf_counter()
    threadpool_results = apply_colormap_threadpool(images_100x512x512_u8, max_workers=4)
    threadpool_time = time.perf_counter() - start

    print('\nUInt8 Timing Results:')
    print(f'Serial: {serial_time:.4f}s')
    print(f'ThreadPoolExecutor (4 workers): {threadpool_time:.4f}s')
    print(f'Speedup: {serial_time / threadpool_time:.2f}x')

    assert len(serial_results) == len(threadpool_results) == 100


def test_timing_comparison_u16(images_100x512x512_u16) -> None:
    """Manual timing comparison for serial vs threadpool execution (uint16)"""
    # Serial
    start = time.perf_counter()
    serial_results = apply_colormap_serial(images_100x512x512_u16)
    serial_time = time.perf_counter() - start

    # ThreadPoolExecutor with 4 workers
    start = time.perf_counter()
    threadpool_results = apply_colormap_threadpool(images_100x512x512_u16, max_workers=4)
    threadpool_time = time.perf_counter() - start

    print('\nUInt16 Timing Results:')
    print(f'Serial: {serial_time:.4f}s')
    print(f'ThreadPoolExecutor (4 workers): {threadpool_time:.4f}s')
    print(f'Speedup: {serial_time / threadpool_time:.2f}x')

    assert len(serial_results) == len(threadpool_results) == 100
