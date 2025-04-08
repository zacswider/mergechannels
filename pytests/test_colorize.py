import numpy as np
import mergechannels as mc


def test_apply_color_map():
    x = np.ones((1, 1), dtype=np.uint8)
    rgb = mc.apply_color_map(x, 'betterBlue')
    assert rgb.shape == (1,1,3)
    assert rgb.dtype == np.uint8
    assert np.allclose(
        rgb,
        np.array(
            [[[0, 1, 2]]]
        )
    )
    x = np.ones((1, 1), dtype=np.uint8) * 255
    rgb = mc.apply_color_map(x, 'betterBlue')
    assert np.allclose(
        rgb,
        np.array(
            [[[0, 188, 254]]]
        )
    )

def test_apply_colors_and_merge_low_sum():
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_sum = mc.apply_colors_and_merge_nc([x, y], ['betterBlue', 'betterOrange'], 'sum')
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(
        rgb_sum,
        np.array(
            [[[1, 2, 2]]]
        )
    )

def test_apply_colors_and_merge_high_sum():
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_sum = mc.apply_colors_and_merge_nc([x, y], ['betterBlue', 'betterOrange'], 'sum')
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(
        rgb_sum,
        np.array(
            [[[255, 255, 254]]]
        )
    )

def test_apply_colors_and_merge_low_max():
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_max = mc.apply_colors_and_merge_nc([x, y], ['betterBlue', 'betterOrange'], 'max')
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(
        rgb_max,
        np.array(
            [[[1, 1, 2]]]
        )
    )

def test_apply_colors_and_merge_high_max():
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_max = mc.apply_colors_and_merge_nc([x, y], ['betterBlue', 'betterOrange'], 'max')
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(
        rgb_max,
        np.array(
            [[[255, 188, 254]]]
        )
    )

def test_apply_colors_and_merge_low_min():
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_min = mc.apply_colors_and_merge_nc([x, y], ['betterBlue', 'betterOrange'], 'min')
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(
        rgb_min,
        np.array(
            [[[0, 1, 0]]]
        )
    )

def test_apply_colors_and_merge_high_min():
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_min = mc.apply_colors_and_merge_nc([x, y], ['betterBlue', 'betterOrange'], 'min')
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(
        rgb_min,
        np.array(
            [[[0, 149, 0]]]
        )
    )

def test_apply_colors_and_merge_low_mean():
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_mean = mc.apply_colors_and_merge_nc([x, y], ['betterBlue', 'betterOrange'], 'mean')
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(
        # NOTE: I haven't decided if I'm going to re-normalize after blending so this test may fail
        # in the future
        rgb_mean,
        np.array(
            [[[0, 1, 1]]]
        )
    )

def test_apply_colors_and_merge_high_mean():
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_mean = mc.apply_colors_and_merge_nc([x, y], ['betterBlue', 'betterOrange'], 'mean')
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(
        # NOTE: I haven't decided if I'm going to re-normalize after blending so this test may fail
        # in the future
        rgb_mean,
        np.array(
            [[[127, 168, 127]]]
        )
    )