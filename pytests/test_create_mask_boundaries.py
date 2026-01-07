"""Tests for create_mask_boundaries function.

Tests verify boundary detection for:
- Different data types (bool, int32, uint16)
- Uniform regions (no boundaries)
- Two adjacent regions (boundary at interface)
- Multiple labels
- Small arrays (edge cases)
- Negative values (for int32)
- Cross-dtype consistency
"""

import mergechannels as mc
import numpy as np
import pytest


class TestUniformRegions:
    """Test that uniform regions produce no boundaries."""

    def test_uniform_bool(self):
        """All True values should produce no boundaries."""
        arr = np.ones((4, 4), dtype=bool)
        result = mc.create_mask_boundaries(arr)
        assert result.dtype == bool
        assert result.shape == (4, 4)
        assert not result.any()

    def test_uniform_i32(self):
        """All same int32 value should produce no boundaries."""
        arr = np.full((4, 4), 5, dtype=np.int32)
        result = mc.create_mask_boundaries(arr)
        assert result.dtype == bool
        assert result.shape == (4, 4)
        assert not result.any()

    def test_uniform_u16(self):
        """All same uint16 value should produce no boundaries."""
        arr = np.full((4, 4), 1000, dtype=np.uint16)
        result = mc.create_mask_boundaries(arr)
        assert result.dtype == bool
        assert result.shape == (4, 4)
        assert not result.any()

    def test_uniform_u8(self):
        """All same uint8 value should produce no boundaries."""
        arr = np.full((4, 4), 100, dtype=np.uint8)
        result = mc.create_mask_boundaries(arr)
        assert result.dtype == bool
        assert result.shape == (4, 4)
        assert not result.any()


class TestTwoRegions:
    """Test boundary detection between two distinct regions."""

    def test_two_regions_bool(self):
        """Test bool array with two regions."""
        arr = np.array(
            [
                [False, False, True, True],
                [False, False, True, True],
                [False, False, True, True],
            ],
            dtype=bool,
        )
        result = mc.create_mask_boundaries(arr)
        expected = np.array(
            [
                [False, True, True, False],
                [False, True, True, False],
                [False, True, True, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(result, expected)

    def test_two_regions_i32(self):
        """Test int32 array with two labeled regions."""
        arr = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.int32,
        )
        result = mc.create_mask_boundaries(arr)
        expected = np.array(
            [
                [False, True, True, False],
                [False, True, True, False],
                [False, True, True, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(result, expected)

    def test_two_regions_u16(self):
        """Test uint16 array with two regions."""
        arr = np.array(
            [
                [100, 100, 200, 200],
                [100, 100, 200, 200],
                [100, 100, 200, 200],
            ],
            dtype=np.uint16,
        )
        result = mc.create_mask_boundaries(arr)
        expected = np.array(
            [
                [False, True, True, False],
                [False, True, True, False],
                [False, True, True, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(result, expected)

    def test_two_regions_u8(self):
        """Test uint8 array with two regions."""
        arr = np.array(
            [
                [10, 10, 20, 20],
                [10, 10, 20, 20],
                [10, 10, 20, 20],
            ],
            dtype=np.uint8,
        )
        result = mc.create_mask_boundaries(arr)
        expected = np.array(
            [
                [False, True, True, False],
                [False, True, True, False],
                [False, True, True, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(result, expected)


class TestSingleDifferentPixel:
    """Test that a single different pixel creates boundaries everywhere."""

    def test_single_pixel_bool(self):
        """Single True pixel in sea of False should make all pixels boundaries."""
        arr = np.array(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ],
            dtype=bool,
        )
        result = mc.create_mask_boundaries(arr)
        # All pixels are boundaries because they all touch the different center pixel
        assert result.all()

    def test_single_pixel_i32(self):
        """Single different label should make all pixels boundaries."""
        arr = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int32,
        )
        result = mc.create_mask_boundaries(arr)
        assert result.all()

    def test_single_pixel_u16(self):
        """Single different value should make all pixels boundaries."""
        arr = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.uint16,
        )
        result = mc.create_mask_boundaries(arr)
        assert result.all()


class TestSmallArrays:
    """Test edge cases with small arrays."""

    def test_1x1_uniform(self):
        """1x1 array should have no boundary."""
        arr = np.array([[5]], dtype=np.int32)
        result = mc.create_mask_boundaries(arr)
        assert result.shape == (1, 1)
        assert not result[0, 0]

    def test_2x2_uniform(self):
        """2x2 uniform array should have no boundaries."""
        arr = np.array([[1, 1], [1, 1]], dtype=np.int32)
        result = mc.create_mask_boundaries(arr)
        assert result.shape == (2, 2)
        assert not result.any()

    def test_2x2_mixed(self):
        """2x2 mixed array should have all boundaries."""
        arr = np.array([[1, 2], [1, 1]], dtype=np.int32)
        result = mc.create_mask_boundaries(arr)
        assert result.shape == (2, 2)
        assert result.all()


class TestMultipleLabels:
    """Test arrays with multiple distinct labels."""

    def test_three_labels_i32(self):
        """Test with three adjacent labeled regions."""
        arr = np.array(
            [
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 2, 3, 3],
            ],
            dtype=np.int32,
        )
        result = mc.create_mask_boundaries(arr)
        expected = np.array(
            [
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(result, expected)

    def test_checkerboard_pattern(self):
        """Test checkerboard pattern creates dense boundaries."""
        arr = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.int32,
        )
        result = mc.create_mask_boundaries(arr)
        # In a checkerboard, every pixel should be a boundary
        assert result.all()


class TestNegativeValues:
    """Test int32 arrays with negative label values."""

    def test_negative_labels(self):
        """Test that negative labels work correctly."""
        arr = np.array(
            [
                [-1, -1, 0, 0],
                [-1, -1, 0, 0],
                [-1, -1, 0, 0],
            ],
            dtype=np.int32,
        )
        result = mc.create_mask_boundaries(arr)
        expected = np.array(
            [
                [False, True, True, False],
                [False, True, True, False],
                [False, True, True, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(result, expected)

    def test_all_negative_uniform(self):
        """Test uniform array with negative values has no boundaries."""
        arr = np.full((3, 3), -5, dtype=np.int32)
        result = mc.create_mask_boundaries(arr)
        assert not result.any()


class TestDtypeConsistency:
    """Test that equivalent patterns produce consistent results across dtypes."""

    def test_two_regions_consistency(self):
        """Same pattern should produce identical results across dtypes."""

        # Create equivalent patterns
        arr_bool = np.array(
            [
                [False, False, True, True],
                [False, False, True, True],
                [False, False, True, True],
            ],
            dtype=bool,
        )
        arr_i32 = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.int32,
        )
        arr_u16 = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.uint16,
        )
        arr_u8 = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.uint8,
        )

        result_bool = mc.create_mask_boundaries(arr_bool)
        result_i32 = mc.create_mask_boundaries(arr_i32)
        result_u16 = mc.create_mask_boundaries(arr_u16)
        result_u8 = mc.create_mask_boundaries(arr_u8)

        # All should produce the same boundary pattern
        np.testing.assert_array_equal(result_bool, result_i32)
        np.testing.assert_array_equal(result_bool, result_u16)
        np.testing.assert_array_equal(result_bool, result_u8)

    def test_uniform_consistency(self):
        """Uniform arrays should produce no boundaries regardless of dtype."""
        arr_bool = np.ones((3, 3), dtype=bool)
        arr_u8 = np.full((3, 3), 50, dtype=np.uint8)
        arr_i32 = np.full((3, 3), 5, dtype=np.int32)
        arr_u16 = np.full((3, 3), 100, dtype=np.uint16)

        result_bool = mc.create_mask_boundaries(arr_bool)
        result_u8 = mc.create_mask_boundaries(arr_u8)
        result_i32 = mc.create_mask_boundaries(arr_i32)
        result_u16 = mc.create_mask_boundaries(arr_u16)

        assert not result_bool.any()
        assert not result_u8.any()
        assert not result_i32.any()
        assert not result_u16.any()


class TestEdgeCases:
    """Test edge and corner pixel behavior."""

    def test_edge_pixels_different(self):
        """Test that edge pixels with different neighbors are detected."""
        arr = np.array(
            [
                [1, 1, 1],
                [1, 2, 1],
                [1, 1, 1],
            ],
            dtype=np.int32,
        )
        result = mc.create_mask_boundaries(arr)
        # All pixels should be boundaries
        assert result.all()

    def test_corner_difference(self):
        """Test corner pixel with different value."""
        arr = np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.int32,
        )
        result = mc.create_mask_boundaries(arr)
        # Pixels that touch the corner should be boundaries
        assert result[0, 0]  # The different corner itself
        assert result[0, 1]  # Right neighbor
        assert result[1, 0]  # Bottom neighbor
        assert result[1, 1]  # Diagonal neighbor


class TestInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_wrong_dtype_raises_error(self):
        """Test that unsupported dtype raises error."""
        arr = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match='only supports bool, uint8, int32, and uint16'):
            mc.create_mask_boundaries(arr)

    def test_3d_array_raises_error(self):
        """Test that 3D array raises error."""
        arr = np.ones((3, 3, 3), dtype=np.int32)
        with pytest.raises(ValueError, match='unsupported number of dimensions'):
            mc.create_mask_boundaries(arr)

    def test_1d_array_raises_error(self):
        """Test that 1D array raises error."""
        arr = np.ones(10, dtype=np.int32)
        with pytest.raises(ValueError, match='unsupported number of dimensions'):
            mc.create_mask_boundaries(arr)


class TestLargerArrays:
    """Test with slightly larger arrays to ensure scalability."""

    def test_50x50_uniform(self):
        """Test that larger uniform array has no boundaries."""
        arr = np.full((50, 50), 42, dtype=np.int32)
        result = mc.create_mask_boundaries(arr)
        assert result.shape == (50, 50)
        assert not result.any()

    def test_50x50_vertical_split(self):
        """Test larger array with vertical boundary."""
        arr = np.zeros((50, 50), dtype=np.int32)
        arr[:, 25:] = 1
        result = mc.create_mask_boundaries(arr)

        # Only columns 24 and 25 should have boundaries
        assert not result[:, :24].any()
        assert result[:, 24].all()
        assert result[:, 25].all()
        assert not result[:, 26:].any()

    def test_50x50_horizontal_split(self):
        """Test larger array with horizontal boundary."""
        arr = np.zeros((50, 50), dtype=np.int32)
        arr[25:, :] = 1
        result = mc.create_mask_boundaries(arr)

        # Only rows 24 and 25 should have boundaries
        assert not result[:24, :].any()
        assert result[24, :].all()
        assert result[25, :].all()
        assert not result[26:, :].any()
