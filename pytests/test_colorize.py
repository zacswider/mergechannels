import numpy as np
import mergechannels as mc

def test_create_rgb_from_arr():
    x = np.ones((3,3), dtype=np.uint8)
    rgb = mc.create_rgb_from_arr(x)
    assert rgb.shape == (3,3,3)

def test_apply_color_map():
    x = np.ones((3,3), dtype=np.uint8)
    rgb = mc.apply_color_map(x)
    assert rgb.shape == (3,3,3)
    assert rgb.dtype == np.uint8
    assert np.allclose(
        rgb,
        np.array(
            [[[0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]],

            [[0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]],

            [[0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]]]
        )
    )
    x = np.ones((3,3), dtype=np.uint8) * 255
    rgb = mc.apply_color_map(x)
    assert np.allclose(
        rgb,
        np.array(
            [[[0, 188, 254],
            [0, 188, 254],
            [0, 188, 254]],

            [[0, 188, 254],
            [0, 188, 254],
            [0, 188, 254]],

            [[0, 188, 254],
            [0, 188, 254],
            [0, 188, 254]]]
        )
    )
