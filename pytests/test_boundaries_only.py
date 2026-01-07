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
