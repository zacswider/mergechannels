"""Tests for mask overlay functionality in colorize operations.

These tests verify that mask overlays work correctly for:
- u8 arrays with bool masks
- u8 arrays with i32 masks
- u16 arrays with bool masks
- u16 arrays with i32 masks

All tests use simple 3x3 arrays with constant pixel values and the default
purple mask color [128, 0, 128] with alpha=0.5 for easy expected value calculation.

Alpha blending formula: result = base * (1 - alpha) + mask * alpha
With alpha=0.5 and purple [128, 0, 128]:
    result = base * 0.5 + [128, 0, 128] * 0.5
           = base * 0.5 + [64, 0, 64]
"""

import mergechannels as mc
import numpy as np
import pytest

# Default mask overlay parameters
PURPLE = (128, 0, 128)
ALPHA = 0.5


@pytest.fixture
def arr_u8():
    """3x3 uint8 array with constant value 100."""
    return np.full((3, 3), 100, dtype=np.uint8)


@pytest.fixture
def arr_u16():
    """3x3 uint16 array with value 25700 (maps to ~100 after normalization)."""
    return np.full((3, 3), 25700, dtype=np.uint16)


@pytest.fixture
def mask_i32():
    """3x3 int32 mask with different non-zero values in a cross pattern."""
    return np.array(
        [
            [0, 1, 0],
            [2, 0, 3],
            [0, 4, 0],
        ],
        dtype=np.int32,
    )


def alpha_blend(
    base_rgb: tuple[int, int, int],
    mask_rgb: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Calculate expected alpha blended color.

    Formula: result = base * (1 - alpha) + mask * alpha
    """
    inv_alpha = 1.0 - alpha
    return (
        int(base_rgb[0] * inv_alpha + mask_rgb[0] * alpha),
        int(base_rgb[1] * inv_alpha + mask_rgb[1] * alpha),
        int(base_rgb[2] * inv_alpha + mask_rgb[2] * alpha),
    )


class TestMaskOverlayU8:
    """Test mask overlay for uint8 arrays."""

    def test_bool_mask_with_specified_color(self, arr_u8, mask_i32):
        """Test u8 array with bool mask and specified purple color."""
        # Use fixture arr_u8: 3x3 array with constant value 100 -> grayscale maps to [100, 100, 100]
        # with Grays cmap

        # Bool mask: cast i32 mask to bool (non-zero values become True)
        mask = mask_i32.astype(bool)

        # Expected: using Grays colormap, value 100 maps to [100, 100, 100]
        # Masked pixels: alpha_blend([100, 100, 100], [128, 0, 128], 0.5)
        # = [100*0.5 + 128*0.5, 100*0.5 + 0*0.5, 100*0.5 + 128*0.5]
        # = [114, 50, 114]
        base_color = (100, 100, 100)
        blended = alpha_blend(base_color, PURPLE, ALPHA)

        result = mc.apply_color_map(
            arr_u8,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask],
            mask_colors=[PURPLE],
            mask_alphas=[ALPHA],
        )

        # Pixels where mask == 0 should be unchanged
        assert result[0, 0, 0] == base_color[0]  # mask[0,0] = 0
        assert result[1, 1, 0] == base_color[0]  # mask[1,1] = 0
        assert result[2, 0, 0] == base_color[0]  # mask[2,0] = 0

        # Pixels where mask != 0 should be blended
        assert result[0, 1, 0] == blended[0]  # mask[0,1] = 1
        assert result[1, 0, 0] == blended[0]  # mask[1,0] = 2
        assert result[1, 2, 0] == blended[0]  # mask[1,2] = 3
        assert result[2, 1, 0] == blended[0]  # mask[2,1] = 4

    def test_i32_mask_with_different_values(self, arr_u8, mask_i32):
        """Test u8 array with i32 mask of different values and specified purple color."""
        # Use fixture arr_u8: 3x3 array with constant value 100

        # Use fixture mask_i32: different non-zero values should all be treated as "mask active"
        # 0 = no mask, any non-zero = mask applied
        mask = mask_i32

        base_color = (100, 100, 100)
        blended = alpha_blend(base_color, PURPLE, ALPHA)

        result = mc.apply_color_map(
            arr_u8,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask],
            mask_colors=[PURPLE],
            mask_alphas=[ALPHA],
        )

        # Pixels where mask == 0 should be unchanged
        assert result[0, 0, 0] == base_color[0]  # mask[0,0] = 0
        assert result[1, 1, 0] == base_color[0]  # mask[1,1] = 0
        assert result[2, 0, 0] == base_color[0]  # mask[2,0] = 0

        # Pixels where mask != 0 should be blended
        assert result[0, 1, 0] == blended[0]  # mask[0,1] = 1
        assert result[1, 0, 0] == blended[0]  # mask[1,0] = 2
        assert result[1, 2, 0] == blended[0]  # mask[1,2] = 3
        assert result[2, 1, 0] == blended[0]  # mask[2,1] = 4


class TestMaskOverlayU16:
    """Test mask overlay for uint16 arrays."""

    def test_bool_mask_with_specified_color(self, arr_u16, mask_i32):
        """Test u16 array with bool mask and specified purple color."""
        # Use fixture arr_u16: 3x3 array with value that maps to ~100 after normalization
        # With limits (0, 65535), value 25700 maps to index ~100
        # 25700 / 65535 * 255 ≈ 100

        # Bool mask: cast i32 mask to bool (non-zero values become True)
        mask = mask_i32.astype(bool)

        # With Grays cmap and limits (0, 65535):
        # 25700 -> index 100 -> [100, 100, 100]
        base_color = (100, 100, 100)
        blended = alpha_blend(base_color, PURPLE, ALPHA)

        result = mc.apply_color_map(
            arr_u16,
            'Grays',
            saturation_limits=(0, 65535),
            masks=[mask],
            mask_colors=[PURPLE],
            mask_alphas=[ALPHA],
        )

        # Pixels where mask == 0 should be unchanged
        assert result[0, 0, 0] == base_color[0]  # mask[0,0] = 0
        assert result[1, 1, 0] == base_color[0]  # mask[1,1] = 0
        assert result[2, 0, 0] == base_color[0]  # mask[2,0] = 0

        # Pixels where mask != 0 should be blended
        assert result[0, 1, 0] == blended[0]  # mask[0,1] = 1
        assert result[1, 0, 0] == blended[0]  # mask[1,0] = 2
        assert result[1, 2, 0] == blended[0]  # mask[1,2] = 3
        assert result[2, 1, 0] == blended[0]  # mask[2,1] = 4

    def test_i32_mask_with_different_values(self, arr_u16, mask_i32):
        """Test u16 array with i32 mask of different values and specified purple color."""
        # Use fixture arr_u16: 3x3 array with value that maps to ~100 after normalization

        # Use fixture mask_i32: different non-zero values should all be treated as "mask active"
        mask = mask_i32

        base_color = (100, 100, 100)
        blended = alpha_blend(base_color, PURPLE, ALPHA)

        result = mc.apply_color_map(
            arr_u16,
            'Grays',
            saturation_limits=(0, 65535),
            masks=[mask],
            mask_colors=[PURPLE],
            mask_alphas=[ALPHA],
        )

        # Pixels where mask == 0 should be unchanged
        assert result[0, 0, 0] == base_color[0]
        assert result[1, 1, 0] == base_color[0]
        assert result[2, 0, 0] == base_color[0]

        # Pixels where mask != 0 should be blended
        assert result[0, 1, 0] == blended[0]
        assert result[1, 0, 0] == blended[0]
        assert result[1, 2, 0] == blended[0]
        assert result[2, 1, 0] == blended[0]


class TestMaskOverlaySerialVsParallel:
    """Test that serial and parallel mask overlay produce identical results."""

    def test_u8_serial_vs_parallel(self, arr_u8, mask_i32):
        """Test that serial and parallel produce identical results for u8 arrays."""
        mask = mask_i32.astype(bool)

        serial_result = mc.apply_color_map(
            arr_u8,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask],
            mask_colors=[PURPLE],
            mask_alphas=[ALPHA],
            parallel=False,
        )

        parallel_result = mc.apply_color_map(
            arr_u8,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask],
            mask_colors=[PURPLE],
            mask_alphas=[ALPHA],
            parallel=True,
        )

        assert np.array_equal(serial_result, parallel_result)

    def test_u16_serial_vs_parallel(self, arr_u16, mask_i32):
        """Test that serial and parallel produce identical results for u16 arrays."""
        mask = mask_i32.astype(bool)

        serial_result = mc.apply_color_map(
            arr_u16,
            'Grays',
            saturation_limits=(0, 65535),
            masks=[mask],
            mask_colors=[PURPLE],
            mask_alphas=[ALPHA],
            parallel=False,
        )

        parallel_result = mc.apply_color_map(
            arr_u16,
            'Grays',
            saturation_limits=(0, 65535),
            masks=[mask],
            mask_colors=[PURPLE],
            mask_alphas=[ALPHA],
            parallel=True,
        )

        assert np.array_equal(serial_result, parallel_result)


class TestMaskOverlayNamedColors:
    """Test mask overlay with named colormap colors."""

    def test_named_color_from_colormap(self, arr_u8, mask_i32):
        """Test mask color specified as a colormap name uses index 255."""
        mask = mask_i32.astype(bool)

        # 'betterBlue' at index 255 is [0, 188, 254]
        blue_color = (0, 188, 254)
        base_color = (100, 100, 100)
        blended = alpha_blend(base_color, blue_color, ALPHA)

        result = mc.apply_color_map(
            arr_u8,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask],
            mask_colors=['betterBlue'],
            mask_alphas=[ALPHA],
        )

        # Pixels where mask != 0 should be blended with blue
        assert result[0, 1, 0] == blended[0]  # R channel
        assert result[0, 1, 1] == blended[1]  # G channel
        assert result[0, 1, 2] == blended[2]  # B channel

    def test_hex_color_full(self, arr_u8, mask_i32):
        """Test mask color specified as a full hex string."""
        mask = mask_i32.astype(bool)

        # '#FF0000' is red
        red_color = (255, 0, 0)
        base_color = (100, 100, 100)
        blended = alpha_blend(base_color, red_color, ALPHA)

        result = mc.apply_color_map(
            arr_u8,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask],
            mask_colors=['#FF0000'],
            mask_alphas=[ALPHA],
        )

        # Pixels where mask != 0 should be blended with red
        assert result[0, 1, 0] == blended[0]  # R channel
        assert result[0, 1, 1] == blended[1]  # G channel
        assert result[0, 1, 2] == blended[2]  # B channel

    def test_hex_color_shorthand(self, arr_u8, mask_i32):
        """Test mask color specified as a shorthand hex string."""
        mask = mask_i32.astype(bool)

        # 'f00' is red (shorthand for FF0000)
        red_color = (255, 0, 0)
        base_color = (100, 100, 100)
        blended = alpha_blend(base_color, red_color, ALPHA)

        result = mc.apply_color_map(
            arr_u8,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask],
            mask_colors=['f00'],
            mask_alphas=[ALPHA],
        )

        # Pixels where mask != 0 should be blended with red
        assert result[0, 1, 0] == blended[0]
        assert result[0, 1, 1] == blended[1]
        assert result[0, 1, 2] == blended[2]


class TestMaskOverlayMultipleMasks:
    """Test mask overlay with multiple masks."""

    def test_two_masks_different_colors(self, arr_u8):
        """Test two non-overlapping masks with different colors."""
        # Create two non-overlapping masks
        mask1 = np.array(
            [
                [True, False, False],
                [False, False, False],
                [False, False, False],
            ],
            dtype=bool,
        )
        mask2 = np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, True],
            ],
            dtype=bool,
        )

        red = (255, 0, 0)
        green = (0, 255, 0)
        base_color = (100, 100, 100)
        blended_red = alpha_blend(base_color, red, ALPHA)
        blended_green = alpha_blend(base_color, green, ALPHA)

        result = mc.apply_color_map(
            arr_u8,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask1, mask2],
            mask_colors=[red, green],
            mask_alphas=[ALPHA, ALPHA],
        )

        # Pixel at [0,0] should be blended with red
        assert result[0, 0, 0] == blended_red[0]
        assert result[0, 0, 1] == blended_red[1]
        assert result[0, 0, 2] == blended_red[2]

        # Pixel at [2,2] should be blended with green
        assert result[2, 2, 0] == blended_green[0]
        assert result[2, 2, 1] == blended_green[1]
        assert result[2, 2, 2] == blended_green[2]

        # Pixel at [1,1] should be unchanged (no mask)
        assert result[1, 1, 0] == base_color[0]
        assert result[1, 1, 1] == base_color[1]
        assert result[1, 1, 2] == base_color[2]
