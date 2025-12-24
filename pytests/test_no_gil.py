import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor

import mergechannels as mc
import numpy as np


def test_gil_status():
    """Verify we're running in free-threaded mode"""
    print(f'Python: {sys.version}')

    # Check if free-threading is available
    if hasattr(sys, '_is_gil_enabled'):
        gil_enabled = sys._is_gil_enabled()  # type: ignore
        print(f'GIL enabled: {gil_enabled}')
    else:
        print('GIL status: Not available (Python < 3.13)')
        gil_enabled = True

    if hasattr(sysconfig, 'get_config_var'):
        free_threaded = sysconfig.get_config_var('Py_GIL_DISABLED')
        print(f'Free-threaded build: {free_threaded}')
    else:
        free_threaded = 0

    # If we're in free-threaded Python, GIL should be disabled
    if free_threaded == 1:
        msg = (
            '⚠️  WARNING: Running free-threaded Python but GIL is enabled!\n',
            '    This means the module may have re-enabled the GIL.',
        )
        assert not gil_enabled, msg
    else:
        print('ℹ️  Running with GIL (standard Python)')

    return gil_enabled, free_threaded


def colorize_worker(worker_id: int, iterations: int = 100):
    """Worker function that calls the Rust extension"""
    results = []
    for i in range(iterations):
        # Create test array
        arr = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

        # Call Rust function via Python wrapper
        result = mc.apply_color_map(arr, color='mpl-viridis', saturation_limits=(0, 255))

        # Verify result shape
        assert result.shape == (512, 512, 3), f'Unexpected shape: {result.shape}'
        results.append(result.shape)

    return (worker_id, len(results))


def test_parallel_single_channel():
    """Test that multiple threads can call single-channel colorization simultaneously"""
    print('\n' + '=' * 60)
    print('TEST: Parallel single-channel colorization')
    print('=' * 60)

    num_threads = 4
    iterations_per_thread = 50

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(colorize_worker, i, iterations_per_thread) for i in range(num_threads)
        ]
        results = [f.result() for f in futures]

    elapsed = time.time() - start_time

    print('\nResults:')
    print(f'  Threads: {num_threads}')
    print(f'  Iterations per thread: {iterations_per_thread}')
    print(f'  Total operations: {num_threads * iterations_per_thread}')
    print(f'  Elapsed time: {elapsed:.2f}s')
    print(f'  Operations/second: {(num_threads * iterations_per_thread) / elapsed:.1f}')

    # Verify all threads completed successfully
    assert len(results) == num_threads
    for worker_id, count in results:
        assert count == iterations_per_thread

    print('✓ All threads completed successfully')
    return elapsed


def merge_worker(worker_id: int, iterations: int = 50):
    """Worker function that calls multi-channel merge"""
    results = []
    for i in range(iterations):
        # Create test arrays
        arr1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        arr2 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        # Call multi-channel merge
        result = mc.merge(
            [arr1, arr2],
            colors=['mpl-viridis', 'mpl-plasma'],
            blending='max',
            saturation_limits=[(0, 255), (0, 255)],
        )

        assert result.shape == (256, 256, 3), f'Unexpected shape: {result.shape}'
        results.append(result.shape)

    return (worker_id, len(results))


def test_parallel_multi_channel():
    """Test that multiple threads can call multi-channel merge simultaneously"""
    print('\n' + '=' * 60)
    print('TEST: Parallel multi-channel merge')
    print('=' * 60)

    num_threads = 4
    iterations_per_thread = 25

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(merge_worker, i, iterations_per_thread) for i in range(num_threads)
        ]
        results = [f.result() for f in futures]

    elapsed = time.time() - start_time

    print('\nResults:')
    print(f'  Threads: {num_threads}')
    print(f'  Iterations per thread: {iterations_per_thread}')
    print(f'  Total operations: {num_threads * iterations_per_thread}')
    print(f'  Elapsed time: {elapsed:.2f}s')
    print(f'  Operations/second: {(num_threads * iterations_per_thread) / elapsed:.1f}')

    # Verify all threads completed successfully
    assert len(results) == num_threads
    for worker_id, count in results:
        assert count == iterations_per_thread

    print('✓ All threads completed successfully')
    return elapsed


def test_concurrent_cmap_access():
    """Test that colormap loading is thread-safe"""
    print('\n' + '=' * 60)
    print('TEST: Concurrent colormap access')
    print('=' * 60)

    # Test various colormaps
    cmaps = [
        'mpl-viridis',
        'mpl-plasma',
        'mpl-inferno',
        'mpl-magma',
        'betterBlue',
        'betterGreen',
        'betterRed',
        'betterYellow',
        '16_colors',
        '5_ramps',
        '6_shades',
    ]

    def load_and_use_cmap(cmap_name):
        arr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        return mc.apply_color_map(arr, color=cmap_name, saturation_limits=(0, 255))

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit 100 random colormap requests
        futures = [executor.submit(load_and_use_cmap, cmaps[i % len(cmaps)]) for i in range(100)]
        results = [f.result() for f in futures]

    elapsed = time.time() - start_time

    assert len(results) == 100
    print(f'\n✓ Successfully processed {len(results)} concurrent colormap operations')
    print(f'  Elapsed time: {elapsed:.2f}s')
    print(f'  Operations/second: {100 / elapsed:.1f}')

    return elapsed


def test_direct_rust_functions():
    """Test calling Rust functions directly (lower-level API)"""
    print('\n' + '=' * 60)
    print('TEST: Direct Rust function calls')
    print('=' * 60)

    def dispatch_worker(worker_id: int, iterations: int = 50):
        results = []
        for i in range(iterations):
            arr = np.random.randint(0, 65535, (128, 128), dtype=np.uint16)
            result = mc.dispatch_single_channel(
                array_reference=arr, cmap_name='mpl-viridis', cmap_values=None, limits=(0, 65535)
            )
            results.append(result.shape)
        return (worker_id, len(results))

    num_threads = 4
    iterations = 50

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(dispatch_worker, i, iterations) for i in range(num_threads)]
        results = [f.result() for f in futures]

    assert len(results) == num_threads
    print(f'✓ Successfully called Rust functions directly from {num_threads} threads')


def benchmark_sequential_vs_parallel():
    """Compare sequential vs parallel execution (only useful in free-threaded Python)"""
    print('\n' + '=' * 60)
    print('BENCHMARK: Sequential vs Parallel Execution')
    print('=' * 60)

    def task():
        arr = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        return mc.apply_color_map(arr, color='mpl-viridis', saturation_limits=(0, 255))

    num_tasks = 20

    # Sequential
    print('\nRunning sequential...')
    start = time.time()
    for _ in range(num_tasks):
        task()
    sequential_time = time.time() - start

    # Parallel
    print('Running parallel...')
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(task) for _ in range(num_tasks)]
        [f.result() for f in futures]
    parallel_time = time.time() - start

    speedup = sequential_time / parallel_time

    print('\nResults:')
    print(f'  Sequential time: {sequential_time:.2f}s')
    print(f'  Parallel time:   {parallel_time:.2f}s')
    print(f'  Speedup:         {speedup:.2f}x')

    if speedup > 1.5:
        print('  ✓ Significant speedup from parallelism!')
    elif speedup > 1.1:
        print('  ✓ Modest speedup from parallelism')
    else:
        print('  ⚠️  Limited speedup (GIL may be limiting parallelism)')


def main():
    """Run all tests"""
    print('\n' + '=' * 60)
    print('MERGECHANNELS FREE-THREADED PYTHON TEST SUITE')
    print('=' * 60)

    # Check GIL status
    gil_enabled, free_threaded = test_gil_status()

    # Run all tests
    try:
        test_parallel_single_channel()
        test_parallel_multi_channel()
        test_concurrent_cmap_access()
        test_direct_rust_functions()

        # Only run benchmark if we're in free-threaded mode
        if free_threaded == 1 and not gil_enabled:
            benchmark_sequential_vs_parallel()

        print('\n' + '=' * 60)
        print('✅ ALL TESTS PASSED!')
        print('=' * 60)

        if free_threaded == 1 and not gil_enabled:
            print('\n🎉 mergechannels is fully compatible with free-threaded Python!')
        elif free_threaded == 1 and gil_enabled:
            print('\n⚠️  Tests passed but GIL is enabled in free-threaded Python')
            print('   The module may have re-enabled the GIL')
        else:
            print('\nℹ️  Tests passed in standard (GIL-enabled) Python')
            print('   Install free-threaded Python 3.13+ to test GIL-free execution')

    except Exception as e:
        print(f'\n❌ TEST FAILED: {e}')
        raise


if __name__ == '__main__':
    main()
