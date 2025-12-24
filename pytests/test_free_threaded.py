"""
Tests for Python 3.14+ free-threaded (no-GIL) support.

These tests verify that mergechannels can:
1. Be imported without re-enabling the GIL automatically
2. Be called concurrently from multiple threads without deadlocks or crashes

Expected to fail until free-threading support is implemented.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Barrier

import mergechannels as mc
import numpy as np
import pytest


# Skip entire module if not in free-threaded environment
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 13) or not hasattr(sys, '_is_gil_enabled') or sys._is_gil_enabled(),
    reason='Tests require Python 3.13+ with GIL disabled (free-threaded build)',
)


def test_gil_disabled_on_import():
    """
    Test that importing mergechannels does not re-enable the GIL.

    In a free-threaded Python build, the GIL should remain disabled after
    importing mergechannels. If this test fails, it means the extension
    is automatically re-enabling the GIL on import.
    """
    assert hasattr(sys, '_is_gil_enabled'), 'sys._is_gil_enabled() not available'
    assert not sys._is_gil_enabled(), 'GIL should remain disabled after import'


def test_concurrent_apply_color_map_basic(large_array_u8):
    """
    Test that apply_color_map can be called concurrently from multiple threads.

    This test spawns 10 threads that simultaneously call apply_color_map on
    different arrays with different colormaps. Each thread should complete
    without deadlocks or crashes.
    """
    num_threads = 10
    colormaps = [
        'betterBlue',
        'betterRed',
        'betterGreen',
        'betterOrange',
        'betterYellow',
        'betterCyan',
        'Grays',
        'mpl-viridis',
        'mpl-plasma',
        'mpl-inferno',
    ]

    # Create deterministic arrays for each thread
    np.random.seed(42)
    arrays = [
        np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)
        for _ in range(num_threads)
    ]

    def apply_colormap_worker(thread_id):
        """Worker function that applies a colormap to an array"""
        arr = arrays[thread_id]
        cmap = colormaps[thread_id]
        result = mc.apply_color_map(arr, cmap, saturation_limits=(0, 255))

        # Verify result shape and dtype
        assert result.shape == (*arr.shape, 3), f'Thread {thread_id}: unexpected shape'
        assert result.dtype == np.uint8, f'Thread {thread_id}: unexpected dtype'

        return thread_id, result

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(apply_colormap_worker, i) for i in range(num_threads)]

        # Collect results
        results = {}
        for future in as_completed(futures):
            thread_id, result = future.result()
            results[thread_id] = result

    # Verify all threads completed
    assert len(results) == num_threads, 'Not all threads completed'


def test_concurrent_apply_color_map_same_array(large_array_u8):
    """
    Test that multiple threads can apply different colormaps to the same array.

    This tests thread safety when multiple threads read from the same input
    array simultaneously.
    """
    num_threads = 10
    colormaps = [
        'betterBlue',
        'betterRed',
        'betterGreen',
        'betterOrange',
        'betterYellow',
        'betterCyan',
        'Grays',
        'mpl-viridis',
        'mpl-plasma',
        'mpl-inferno',
    ]

    # Use the same input array for all threads
    np.random.seed(42)
    shared_array = np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)

    def apply_colormap_worker(thread_id):
        """Worker function that applies a colormap to the shared array"""
        cmap = colormaps[thread_id]
        result = mc.apply_color_map(shared_array, cmap, saturation_limits=(0, 255))
        return thread_id, result

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(apply_colormap_worker, i) for i in range(num_threads)]
        results = {future.result()[0]: future.result()[1] for future in as_completed(futures)}

    # Verify all threads completed and produced different results
    assert len(results) == num_threads

    # Verify results are deterministic by comparing with sequential execution
    for i in range(num_threads):
        expected = mc.apply_color_map(shared_array, colormaps[i], saturation_limits=(0, 255))
        assert np.array_equal(results[i], expected), f'Thread {i} produced incorrect result'


def test_concurrent_merge(large_array_u8):
    """
    Test that merge can be called concurrently from multiple threads.

    This tests the multi-channel merging function under concurrent load.
    """
    num_threads = 10

    # Create deterministic arrays for each thread
    np.random.seed(42)
    array_sets = [
        [np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8) for _ in range(2)]
        for _ in range(num_threads)
    ]

    colormaps = [
        ['betterBlue', 'betterRed'],
        ['betterGreen', 'betterOrange'],
        ['betterYellow', 'betterCyan'],
        ['mpl-viridis', 'mpl-plasma'],
        ['mpl-inferno', 'mpl-magma'],
        ['Grays', 'betterBlue'],
        ['betterRed', 'betterGreen'],
        ['betterOrange', 'betterYellow'],
        ['betterCyan', 'mpl-viridis'],
        ['mpl-plasma', 'mpl-inferno'],
    ]

    def merge_worker(thread_id):
        """Worker function that merges channels"""
        arrs = array_sets[thread_id]
        cmaps = colormaps[thread_id]
        result = mc.merge(arrs, cmaps, blending='max', saturation_limits=[(0, 255), (0, 255)])

        # Verify result shape and dtype
        assert result.shape == (*arrs[0].shape, 3), f'Thread {thread_id}: unexpected shape'
        assert result.dtype == np.uint8, f'Thread {thread_id}: unexpected dtype'

        return thread_id, result

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(merge_worker, i) for i in range(num_threads)]
        results = {future.result()[0]: future.result()[1] for future in as_completed(futures)}

    # Verify all threads completed
    assert len(results) == num_threads


def test_concurrent_mixed_operations(large_array_u8):
    """
    Test that different operations can be performed concurrently.

    This test mixes apply_color_map and merge operations across threads
    to test cross-function thread safety.
    """
    num_threads = 10

    # Create deterministic arrays
    np.random.seed(42)
    arrays = [
        np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)
        for _ in range(num_threads)
    ]

    def worker(thread_id):
        """Worker that alternates between apply_color_map and merge"""
        arr = arrays[thread_id]

        if thread_id % 2 == 0:
            # Even threads: apply_color_map
            result = mc.apply_color_map(arr, 'betterBlue', saturation_limits=(0, 255))
        else:
            # Odd threads: merge
            arr2 = arrays[(thread_id + 1) % num_threads]
            result = mc.merge(
                [arr, arr2],
                ['betterRed', 'betterGreen'],
                blending='max',
                saturation_limits=[(0, 255), (0, 255)],
            )

        return thread_id, result

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = {future.result()[0]: future.result()[1] for future in as_completed(futures)}

    # Verify all threads completed
    assert len(results) == num_threads


def test_concurrent_with_barrier(large_array_u8):
    """
    Test concurrent execution with synchronized start using a barrier.

    This test uses a barrier to ensure all threads start executing
    simultaneously, maximizing the chance of exposing race conditions.
    """
    num_threads = 10
    barrier = Barrier(num_threads)
    results = {}
    exceptions = {}

    # Create deterministic arrays
    np.random.seed(42)
    arrays = [
        np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)
        for _ in range(num_threads)
    ]

    def worker(thread_id):
        """Worker that waits at barrier before executing"""
        try:
            # Wait for all threads to be ready
            barrier.wait()

            # Execute immediately after barrier
            arr = arrays[thread_id]
            result = mc.apply_color_map(arr, 'betterBlue', saturation_limits=(0, 255))
            results[thread_id] = result

        except Exception as e:
            exceptions[thread_id] = e

    # Start threads
    threads = [Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    # Check for exceptions
    if exceptions:
        raise AssertionError(f'Threads raised exceptions: {exceptions}')

    # Verify all threads completed
    assert len(results) == num_threads, 'Not all threads completed successfully'


def test_concurrent_stress_test(large_array_u8):
    """
    Stress test with multiple iterations per thread.

    This test performs repeated operations in each thread to increase
    the likelihood of exposing race conditions and deadlocks.
    """
    num_threads = 10
    iterations_per_thread = 5

    # Create deterministic arrays
    np.random.seed(42)
    base_arrays = [
        np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)
        for _ in range(num_threads)
    ]

    def stress_worker(thread_id):
        """Worker that performs multiple operations"""
        results = []
        arr = base_arrays[thread_id]

        for iteration in range(iterations_per_thread):
            # Alternate between different operations
            if iteration % 3 == 0:
                result = mc.apply_color_map(arr, 'betterBlue', saturation_limits=(0, 255))
            elif iteration % 3 == 1:
                result = mc.apply_color_map(arr, 'betterRed', saturation_limits=(0, 255))
            else:
                arr2 = base_arrays[(thread_id + 1) % num_threads]
                result = mc.merge(
                    [arr, arr2],
                    ['betterGreen', 'betterOrange'],
                    blending='max',
                    saturation_limits=[(0, 255), (0, 255)],
                )

            results.append(result)

        return thread_id, results

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
        results = {future.result()[0]: future.result()[1] for future in as_completed(futures)}

    # Verify all threads completed all iterations
    assert len(results) == num_threads
    for thread_id, thread_results in results.items():
        assert len(thread_results) == iterations_per_thread, (
            f'Thread {thread_id} completed {len(thread_results)} of {iterations_per_thread} iterations'
        )


def test_concurrent_data_integrity(large_array_u8):
    """
    Test that concurrent execution produces correct results.

    This test compares results from concurrent execution with results
    from sequential execution to verify data integrity.
    """
    num_threads = 10

    # Create deterministic arrays
    np.random.seed(42)
    arrays = [
        np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)
        for _ in range(num_threads)
    ]

    colormaps = [
        'betterBlue',
        'betterRed',
        'betterGreen',
        'betterOrange',
        'betterYellow',
        'betterCyan',
        'Grays',
        'mpl-viridis',
        'mpl-plasma',
        'mpl-inferno',
    ]

    # Compute expected results sequentially
    expected_results = {}
    for i in range(num_threads):
        expected_results[i] = mc.apply_color_map(
            arrays[i], colormaps[i], saturation_limits=(0, 255)
        )

    def worker(thread_id):
        """Worker function"""
        result = mc.apply_color_map(
            arrays[thread_id], colormaps[thread_id], saturation_limits=(0, 255)
        )
        return thread_id, result

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        concurrent_results = {
            future.result()[0]: future.result()[1] for future in as_completed(futures)
        }

    # Compare results
    for thread_id in range(num_threads):
        assert np.array_equal(concurrent_results[thread_id], expected_results[thread_id]), (
            f'Thread {thread_id} produced incorrect result (data corruption or race condition)'
        )


def test_dispatch_single_channel_concurrent(large_array_u8):
    """
    Test that dispatch_single_channel can be called concurrently.

    This tests the lower-level dispatch_single_channel function.
    """
    num_threads = 10

    # Create deterministic arrays
    np.random.seed(42)
    arrays = [
        np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)
        for _ in range(num_threads)
    ]

    def worker(thread_id):
        """Worker using dispatch_single_channel"""
        result = mc.dispatch_single_channel(arrays[thread_id], 'betterBlue', None, (0, 255))
        return thread_id, result

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = {future.result()[0]: future.result()[1] for future in as_completed(futures)}

    # Verify all threads completed
    assert len(results) == num_threads


def test_dispatch_multi_channel_concurrent(large_array_u8):
    """
    Test that dispatch_multi_channel can be called concurrently.

    This tests the lower-level dispatch_multi_channel function.
    """
    num_threads = 10

    # Create deterministic arrays
    np.random.seed(42)
    array_sets = [
        [np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8) for _ in range(2)]
        for _ in range(num_threads)
    ]

    def worker(thread_id):
        """Worker using dispatch_multi_channel"""
        result = mc.dispatch_multi_channel(
            array_sets[thread_id],
            ['betterBlue', 'betterRed'],
            [None, None],
            'max',
            [(0, 255), (0, 255)],
        )
        return thread_id, result

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = {future.result()[0]: future.result()[1] for future in as_completed(futures)}

    # Verify all threads completed
    assert len(results) == num_threads
