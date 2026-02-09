"""Tests for boundaries_only parameter in apply_color_map and merge."""

import warnings

import mergechannels as mc
import numpy as np
import pytest


class TestBoundariesOnly3DWarning:
    """Test that boundaries_only=True emits a warning for 3D arrays."""

    def test_apply_color_map_3d_warning(self):
        """apply_color_map should warn when boundaries_only=True with 3D array."""
        arr = np.random.randint(0, 256, (5, 64, 64), dtype=np.uint8)
        mask = np.zeros((5, 64, 64), dtype=np.bool_)
        mask[:, 20:40, 20:40] = True

        with pytest.warns(
            UserWarning,
            match='boundaries_only=True is not supported for 3D arrays',
        ):
            mc.apply_color_map(
                arr,
                'Grays',
                saturation_limits=(0, 255),
                masks=[mask],
                boundaries_only=True,
            )

    def test_merge_3d_warning(self):
        """merge should warn when boundaries_only=True with 3D arrays."""
        arr1 = np.random.randint(0, 256, (5, 64, 64), dtype=np.uint8)
        arr2 = np.random.randint(0, 256, (5, 64, 64), dtype=np.uint8)
        mask = np.zeros((5, 64, 64), dtype=np.bool_)
        mask[:, 20:40, 20:40] = True

        with pytest.warns(
            UserWarning,
            match='boundaries_only=True is not supported for 3D arrays',
        ):
            mc.merge(
                [arr1, arr2],
                ['betterBlue', 'betterOrange'],
                saturation_limits=[(0, 255), (0, 255)],
                masks=[mask],
                boundaries_only=True,
            )

    def test_apply_color_map_2d_no_warning(self):
        """apply_color_map should NOT warn when boundaries_only=True with 2D array."""
        arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.bool_)
        mask[20:40, 20:40] = True

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            # Should not raise any warnings
            mc.apply_color_map(
                arr,
                'Grays',
                saturation_limits=(0, 255),
                masks=[mask],
                boundaries_only=True,
            )

    def test_merge_2d_no_warning(self):
        """merge should NOT warn when boundaries_only=True with 2D arrays."""
        arr1 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        arr2 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.bool_)
        mask[20:40, 20:40] = True

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            # Should not raise any warnings
            mc.merge(
                [arr1, arr2],
                ['betterBlue', 'betterOrange'],
                saturation_limits=[(0, 255), (0, 255)],
                masks=[mask],
                boundaries_only=True,
            )

    def test_no_warning_without_masks(self):
        """No warning should be emitted if boundaries_only=True but no masks provided."""
        arr = np.random.randint(0, 256, (5, 64, 64), dtype=np.uint8)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            # Should not raise any warnings since there are no masks
            mc.apply_color_map(
                arr,
                'Grays',
                saturation_limits=(0, 255),
                boundaries_only=True,
            )

    def test_apply_color_map_3d_warning_with_list(self):
        """apply_color_map should warn when any boundaries_only value is True with 3D array."""
        arr = np.random.randint(0, 256, (5, 64, 64), dtype=np.uint8)
        mask1 = np.zeros((5, 64, 64), dtype=np.bool_)
        mask1[:, 10:30, 10:30] = True
        mask2 = np.zeros((5, 64, 64), dtype=np.bool_)
        mask2[:, 30:50, 30:50] = True

        with pytest.warns(
            UserWarning,
            match='boundaries_only=True is not supported for 3D arrays',
        ):
            mc.apply_color_map(
                arr,
                'Grays',
                saturation_limits=(0, 255),
                masks=[mask1, mask2],
                boundaries_only=[False, True],  # Only second mask has boundaries
            )


class TestBoundariesOnlyPerMask:
    """Test per-mask boundaries_only functionality."""

    def test_apply_color_map_per_mask_boundaries(self):
        """apply_color_map with per-mask boundaries_only list."""
        arr = np.zeros((64, 64), dtype=np.uint8)
        arr[20:44, 20:44] = 128

        # First mask: full overlay (solid square)
        mask1 = np.zeros((64, 64), dtype=np.bool_)
        mask1[10:20, 10:20] = True

        # Second mask: boundary only (should only show edges)
        mask2 = np.zeros((64, 64), dtype=np.int32)
        mask2[30:50, 30:50] = 1

        rgb = mc.apply_color_map(
            arr,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask1, mask2],
            mask_colors=['#FF0000', '#00FF00'],
            boundaries_only=[False, True],  # First full, second boundary-only
        )

        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

        # First mask region (10:20, 10:20) should have red overlay (full)
        # Check center of first mask - should be affected by red
        assert rgb[15, 15, 0] > 0  # Red channel should be affected

        # To verify boundaries_only works, compare with full mask overlay
        rgb_full = mc.apply_color_map(
            arr,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask1, mask2],
            mask_colors=['#FF0000', '#00FF00'],
            boundaries_only=[False, False],  # Both full overlays
        )

        # Interior of second mask should differ: boundary-only has no green, full has green
        # The interior point (40,40) should have MORE green in the full overlay version
        assert rgb_full[40, 40, 1] > rgb[40, 40, 1]  # Full has more green in interior

        # Boundary should be the same in both cases
        assert rgb[30, 40, 1] == rgb_full[30, 40, 1]  # Top edge same in both

    def test_merge_per_mask_boundaries(self):
        """merge with per-mask boundaries_only list."""
        arr1 = np.zeros((64, 64), dtype=np.uint8)
        arr1[20:44, 20:44] = 128
        arr2 = np.zeros((64, 64), dtype=np.uint8)
        arr2[10:54, 10:54] = 100

        # First mask: boundary only
        mask1 = np.zeros((64, 64), dtype=np.bool_)
        mask1[5:15, 5:15] = True

        # Second mask: full overlay
        mask2 = np.zeros((64, 64), dtype=np.int32)
        mask2[50:60, 50:60] = 1

        rgb = mc.merge(
            [arr1, arr2],
            ['betterBlue', 'betterOrange'],
            saturation_limits=[(0, 255), (0, 255)],
            masks=[mask1, mask2],
            mask_colors=['#FF0000', '#00FF00'],
            boundaries_only=[True, False],  # First boundary-only, second full
        )

        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

        # Compare with full mask overlays to verify boundaries_only works
        rgb_full = mc.merge(
            [arr1, arr2],
            ['betterBlue', 'betterOrange'],
            saturation_limits=[(0, 255), (0, 255)],
            masks=[mask1, mask2],
            mask_colors=['#FF0000', '#00FF00'],
            boundaries_only=[False, False],  # Both full overlays
        )

        # First mask interior: boundary-only should have LESS red than full
        assert rgb_full[10, 10, 0] > rgb[10, 10, 0]  # Full has more red in interior

        # First mask boundary should be the same
        assert rgb[5, 10, 0] == rgb_full[5, 10, 0]  # Top edge same in both

        # Second mask interior should be the same (both have full overlay for mask2)
        assert rgb[55, 55, 1] == rgb_full[55, 55, 1]  # Center of second mask same

    def test_boundaries_only_length_mismatch_raises(self):
        """boundaries_only list with wrong length should raise ValueError."""
        arr = np.zeros((64, 64), dtype=np.uint8)
        mask1 = np.zeros((64, 64), dtype=np.bool_)
        mask1[10:20, 10:20] = True
        mask2 = np.zeros((64, 64), dtype=np.bool_)
        mask2[30:40, 30:40] = True

        with pytest.raises(ValueError, match='boundaries_only values .* does not match'):
            mc.apply_color_map(
                arr,
                'Grays',
                saturation_limits=(0, 255),
                masks=[mask1, mask2],
                boundaries_only=[True],  # Wrong length - should be 2
            )

    def test_boundaries_only_none_defaults_to_false(self):
        """boundaries_only=None should default to False for all masks."""
        arr = np.zeros((64, 64), dtype=np.uint8)
        arr[20:44, 20:44] = 128

        mask = np.zeros((64, 64), dtype=np.int32)
        mask[30:50, 30:50] = 1

        rgb = mc.apply_color_map(
            arr,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask],
            mask_colors=['#00FF00'],
            boundaries_only=None,  # Should default to False
        )

        # Interior of mask should have green (full overlay, not boundary-only)
        assert rgb[40, 40, 1] > 0  # Center should have green

    def test_boundaries_only_all_false_same_as_none(self):
        """boundaries_only=[False, False] should behave same as None."""
        arr = np.zeros((64, 64), dtype=np.uint8)
        arr[20:44, 20:44] = 128

        mask1 = np.zeros((64, 64), dtype=np.bool_)
        mask1[10:20, 10:20] = True
        mask2 = np.zeros((64, 64), dtype=np.int32)
        mask2[30:50, 30:50] = 1

        rgb_none = mc.apply_color_map(
            arr,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask1, mask2],
            mask_colors=['#FF0000', '#00FF00'],
            boundaries_only=None,
        )

        rgb_false = mc.apply_color_map(
            arr,
            'Grays',
            saturation_limits=(0, 255),
            masks=[mask1, mask2],
            mask_colors=['#FF0000', '#00FF00'],
            boundaries_only=[False, False],
        )

        np.testing.assert_array_equal(rgb_none, rgb_false)
