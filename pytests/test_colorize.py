import numpy as np
import mergechannels as mc


def test_apply_color_map():
    '''
    Test that the color map is applied correctly
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    rgb = mc.dispatch_single_channel(x, 'betterBlue', (0, 255))
    assert rgb.shape == (1,1,3)
    assert rgb.dtype == np.uint8
    assert np.allclose(
        rgb,
        np.array(
            [[[0, 1, 2]]]
        )
    )
    x = np.ones((1, 1), dtype=np.uint8) * 255
    rgb = mc.dispatch_single_channel(x, 'betterBlue', (0, 255))
    assert np.allclose(
        rgb,
        np.array(
            [[[0, 188, 254]]]
        )
    )

def test_apply_colors_and_merge_low_sum():
    '''
    Test that the colors are merged correctly with sum blending and low values
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_sum = mc.dispatch_multi_channel(
        [x, y],
        ['betterBlue', 'betterOrange'],
        'sum',
        [(0, 255), (0, 255)],
    )
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(
        rgb_sum,
        np.array(
            [[[1, 2, 2]]]
        )
    )

def test_apply_colors_and_merge_high_sum():
    '''
    Test that the colors are merged correctly with sum blending and high values
    '''
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_sum = mc.dispatch_multi_channel(
        [x, y],
        ['betterBlue', 'betterOrange'],
        'sum',
        [(0, 255), (0, 255)],
    )
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(
        rgb_sum,
        np.array(
            [[[255, 255, 254]]]
        )
    )

def test_apply_colors_and_merge_low_max():
    '''
    Test that the colors are merged correctly with max blending and low values
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_max = mc.dispatch_multi_channel(
        [x, y],
        ['betterBlue', 'betterOrange'],
        'max',
        [(0, 255), (0, 255)],
    )
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(
        rgb_max,
        np.array(
            [[[1, 1, 2]]]
        )
    )

def test_apply_colors_and_merge_high_max():
    '''
    Test that the colors are merged correctly with max blending and high values
    '''
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_max = mc.dispatch_multi_channel(
        [x, y],
        ['betterBlue', 'betterOrange'],
        'max',
        [(0, 255), (0, 255)],
    )
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(
        rgb_max,
        np.array(
            [[[255, 188, 254]]]
        )
    )

def test_apply_colors_and_merge_low_min():
    '''
    Test that the colors are merged correctly with min blending and low values
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_min = mc.dispatch_multi_channel(
        [x, y],
        ['betterBlue', 'betterOrange'],
        'min',
        [(0, 255), (0, 255)],
    )
    # blue = [0, 1, 2,]
    # orange = [1, 1, 0]
    assert np.allclose(
        rgb_min,
        np.array(
            [[[0, 1, 0]]]
        )
    )

def test_apply_colors_and_merge_high_min():
    '''
    Test that the colors are merged correctly with min blending and high values
    '''
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_min = mc.dispatch_multi_channel(
        [x, y],
        ['betterBlue', 'betterOrange'],
        'min',
        [(0, 255), (0, 255)],
    )
    # blue = [0, 188, 254]
    # orange = [255, 149, 0]
    assert np.allclose(
        rgb_min,
        np.array(
            [[[0, 149, 0]]]
        )
    )

def test_apply_colors_and_merge_low_mean():
    '''
    Test that the colors are merged correctly with mean blending and low values
    '''
    x = np.ones((1, 1), dtype=np.uint8)
    y = np.ones((1, 1), dtype=np.uint8)
    rgb_mean = mc.dispatch_multi_channel(
        [x, y],
        ['betterBlue', 'betterOrange'],
        'mean',
        [(0, 255), (0, 255)],
    )
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
    '''
    Test that the colors are merged correctly with mean blending and high values
    '''
    x = np.ones((1, 1), dtype=np.uint8) * 255
    y = np.ones((1, 1), dtype=np.uint8) * 255
    rgb_mean = mc.dispatch_multi_channel(
        [x, y],
        ['betterBlue', 'betterOrange'],
        'mean',
        [(0, 255), (0, 255)]
    )
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
