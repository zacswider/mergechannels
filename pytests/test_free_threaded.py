import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import mergechannels as mc
import numpy as np
import pytest

# Skip entire module if not in free-threaded environment
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 13) or not hasattr(sys, '_is_gil_enabled') or sys._is_gil_enabled(),  # type: ignore[attr-defined]
    reason='Tests require Python 3.13+ with GIL disabled (free-threaded build)',
)

# Test constants
NUM_THREADS = 10
STRESS_ITERATIONS = 5
COLORMAPS = [
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


def run_concurrent(worker_fn, num_threads=NUM_THREADS):
    """
    Execute a worker function concurrently across multiple threads.

    Args:
        worker_fn: Callable that takes thread_id and returns (thread_id, result)
        num_threads: Number of concurrent threads to spawn

    Returns:
        dict: Mapping of thread_id -> result for all completed threads
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_fn, i) for i in range(num_threads)]
        results = {}
        for future in as_completed(futures):
            thread_id, result = future.result()
            results[thread_id] = result
    return results


def test_gil_disabled_on_import():
    """
    Test that importing mergechannels does not re-enable the GIL.

    In a free-threaded Python build, the GIL should remain disabled after
    importing mergechannels. If this test fails, it means the extension
    is automatically re-enabling the GIL on import.
    """
    assert hasattr(sys, '_is_gil_enabled'), 'sys._is_gil_enabled() not available'
    assert not sys._is_gil_enabled(), 'GIL should remain disabled after import'  # type: ignore[attr-defined]


def test_concurrent_apply_color_map(large_array_u8):
    """
    Test that apply_color_map can be called concurrently from multiple threads.

    Each thread applies a different colormap to a different array.
    """
    # Create deterministic arrays for each thread
    np.random.seed(42)
    arrays = [
        np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)
        for _ in range(NUM_THREADS)
    ]

    def worker(thread_id):
        arr = arrays[thread_id]
        cmap = COLORMAPS[thread_id]
        result = mc.apply_color_map(arr, cmap, saturation_limits=(0, 255))

        # Verify result shape and dtype
        assert result.shape == (*arr.shape, 3), f'Thread {thread_id}: unexpected shape'
        assert result.dtype == np.uint8, f'Thread {thread_id}: unexpected dtype'

        return thread_id, result

    results = run_concurrent(worker)
    assert len(results) == NUM_THREADS, 'Not all threads completed'


def test_concurrent_apply_color_map_shared_input(large_array_u8):
    """
    Test that multiple threads can apply different colormaps to the same array.

    This tests thread safety when multiple threads read from the same input
    array simultaneously. Results should match sequential execution.
    """
    np.random.seed(42)
    shared_array = np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)

    def worker(thread_id):
        cmap = COLORMAPS[thread_id]
        result = mc.apply_color_map(shared_array, cmap, saturation_limits=(0, 255))
        return thread_id, result

    results = run_concurrent(worker)
    assert len(results) == NUM_THREADS

    # Verify results match sequential execution
    for i in range(NUM_THREADS):
        expected = mc.apply_color_map(shared_array, COLORMAPS[i], saturation_limits=(0, 255))
        assert np.array_equal(results[i], expected), f'Thread {i} produced incorrect result'


def test_concurrent_merge(large_array_u8):
    """
    Test that merge can be called concurrently from multiple threads.

    Each thread merges two arrays with different colormaps.
    """
    # Create deterministic array pairs for each thread
    np.random.seed(42)
    array_pairs = [
        [
            np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8),
            np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8),
        ]
        for _ in range(NUM_THREADS)
    ]

    colormap_pairs = [
        [COLORMAPS[i], COLORMAPS[(i + 1) % len(COLORMAPS)]] for i in range(NUM_THREADS)
    ]

    def worker(thread_id):
        arrs = array_pairs[thread_id]
        cmaps = colormap_pairs[thread_id]
        result = mc.merge(
            arrs,
            cmaps,
            blending='max',
            saturation_limits=[(0, 255), (0, 255)],
        )

        # Verify result shape and dtype
        assert result.shape == (*arrs[0].shape, 3), f'Thread {thread_id}: unexpected shape'
        assert result.dtype == np.uint8, f'Thread {thread_id}: unexpected dtype'

        return thread_id, result

    results = run_concurrent(worker)
    assert len(results) == NUM_THREADS, 'Not all threads completed'


def test_concurrent_mixed_operations(large_array_u8):
    """
    Test that apply_color_map and merge can be called concurrently.

    Even threads use apply_color_map, odd threads use merge.
    This tests cross-function thread safety.
    """
    np.random.seed(42)
    arrays = [
        np.random.randint(0, 256, size=large_array_u8.shape, dtype=np.uint8)
        for _ in range(NUM_THREADS)
    ]

    def worker(thread_id):
        arr = arrays[thread_id]

        if thread_id % 2 == 0:
            # Even threads: apply_color_map
            result = mc.apply_color_map(arr, 'betterBlue', saturation_limits=(0, 255))
        else:
            # Odd threads: merge
            arr2 = arrays[(thread_id + 1) % NUM_THREADS]
            result = mc.merge(
                [arr, arr2],
                ['betterRed', 'betterGreen'],
                blending='max',
                saturation_limits=[(0, 255), (0, 255)],
            )

        return thread_id, result

    results = run_concurrent(worker)
    assert len(results) == NUM_THREADS, 'Not all threads completed'
