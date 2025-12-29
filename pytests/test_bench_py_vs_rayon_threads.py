from concurrent.futures import ThreadPoolExecutor

import mergechannels as mc
import numpy as np
import pytest


def colorize_single_image(args):
    """Worker function to colorize a single image in a Python thread."""
    arr, colormap, saturation_limits = args
    return mc.apply_color_map(arr, colormap, saturation_limits=saturation_limits, parallel=False)


def python_threads_colorize(images, colormaps, max_workers=10, saturation_limits=(0, 255)):
    """Colorize multiple images using Python ThreadPoolExecutor.

    Args:
        images: List of numpy arrays to colorize
        colormaps: List of colormap names (same length as images)
        max_workers: Number of threads to use (default: 10)
        saturation_limits: Tuple of (min, max) or None for auto-scaling

    Returns:
        List of colorized images
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                colorize_single_image, zip(images, colormaps, [saturation_limits] * len(images))
            )
        )
    return results


@pytest.mark.benchmark(group='web app workload: rust parallel vs python threads')
def test_bench_webapp_workload_rust_parallel(benchmark) -> None:
    """
    Benchmark colorizing 10 175x85 u8 images using Rust's parallel flag (realistic web app workload)
    """
    # Create 10 random 175x85 u8 images
    np.random.seed(42)
    images = np.random.randint(0, 256, size=(10, 175, 85), dtype=np.uint8)

    results = benchmark(
        mc.apply_color_map,
        arr=images,
        color='Grays',
        saturation_limits=None,
        parallel=True,
    )
    assert len(results) == 10
    assert all(r.shape == (175, 85, 3) for r in results)
    assert all(r.dtype == np.uint8 for r in results)


@pytest.mark.benchmark(group='web app workload: rust parallel vs python threads')
def test_bench_webapp_workload_python_threads(benchmark) -> None:
    """
    Benchmark colorizing 10 175x85 u8 images using 10 Python threads (realistic web app workload)
    """
    # Create 10 random 175x85 u8 images
    np.random.seed(42)
    images = [np.random.randint(0, 256, size=(175, 85), dtype=np.uint8) for _ in range(10)]
    colormaps = ['Grays'] * 10

    results = benchmark(python_threads_colorize, images, colormaps, saturation_limits=None)
    assert len(results) == 10
    assert all(r.shape == (175, 85, 3) for r in results)
    assert all(r.dtype == np.uint8 for r in results)


@pytest.mark.benchmark(group='10 images: rust parallel vs python threads')
def test_bench_10_images_rust_parallel(benchmark) -> None:
    """Benchmark colorizing 10 512x512 u8 images using Rust's parallel flag"""
    # Create 10 random 512x512 u8 images
    np.random.seed(42)
    images = np.random.randint(0, 256, size=(10, 512, 512), dtype=np.uint8)

    results = benchmark(
        mc.apply_color_map,
        arr=images,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert len(results) == 10
    assert all(r.shape == (512, 512, 3) for r in results)
    assert all(r.dtype == np.uint8 for r in results)


@pytest.mark.benchmark(group='10 images: rust parallel vs python threads')
def test_bench_10_images_python_threads(benchmark) -> None:
    """Benchmark colorizing 10 512x512 u8 images using 10 Python threads"""
    # Create 10 random 512x512 u8 images
    np.random.seed(42)
    images = [np.random.randint(0, 256, size=(512, 512), dtype=np.uint8) for _ in range(10)]
    colormaps = ['Grays'] * 10

    results = benchmark(python_threads_colorize, images, colormaps)
    assert len(results) == 10
    assert all(r.shape == (512, 512, 3) for r in results)
    assert all(r.dtype == np.uint8 for r in results)


@pytest.mark.benchmark(group='100 images: rust parallel vs python threads')
def test_bench_100_images_rust_parallel(benchmark) -> None:
    """Benchmark colorizing 100 512x512 u8 images using Rust's parallel flag"""
    # Create 100 random 512x512 u8 images
    np.random.seed(42)
    images = np.random.randint(0, 256, size=(100, 512, 512), dtype=np.uint8)

    results = benchmark(
        mc.apply_color_map,
        arr=images,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert len(results) == 100
    assert all(r.shape == (512, 512, 3) for r in results)
    assert all(r.dtype == np.uint8 for r in results)


@pytest.mark.benchmark(group='100 images: rust parallel vs python threads')
def test_bench_100_images_python_threads(benchmark) -> None:
    """Benchmark colorizing 100 512x512 u8 images using 10 Python threads"""
    # Create 100 random 512x512 u8 images
    np.random.seed(42)
    images = [np.random.randint(0, 256, size=(512, 512), dtype=np.uint8) for _ in range(100)]
    colormaps = ['Grays'] * 100

    results = benchmark(python_threads_colorize, images, colormaps)
    assert len(results) == 100
    assert all(r.shape == (512, 512, 3) for r in results)
    assert all(r.dtype == np.uint8 for r in results)


@pytest.mark.benchmark(group='500 images: rust parallel vs python threads')
def test_bench_500_images_rust_parallel(benchmark) -> None:
    """Benchmark colorizing 500 512x512 u8 images using Rust's parallel flag"""
    # Create 500 random 512x512 u8 images
    np.random.seed(42)
    images = np.random.randint(0, 256, size=(500, 512, 512), dtype=np.uint8)

    results = benchmark(
        mc.apply_color_map,
        arr=images,
        color='Grays',
        saturation_limits=(0, 255),
        parallel=True,
    )
    assert len(results) == 500
    assert all(r.shape == (512, 512, 3) for r in results)
    assert all(r.dtype == np.uint8 for r in results)


@pytest.mark.benchmark(group='500 images: rust parallel vs python threads')
def test_bench_500_images_python_threads(benchmark) -> None:
    """Benchmark colorizing 500 512x512 u8 images using 10 Python threads"""
    # Create 500 random 512x512 u8 images
    np.random.seed(42)
    images = [np.random.randint(0, 256, size=(512, 512), dtype=np.uint8) for _ in range(500)]
    colormaps = ['Grays'] * 500

    results = benchmark(python_threads_colorize, images, colormaps)
    assert len(results) == 500
    assert all(r.shape == (512, 512, 3) for r in results)
    assert all(r.dtype == np.uint8 for r in results)
