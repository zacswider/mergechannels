import matplotlib.pyplot as plt
import mergechannels as mc
import numpy as np


def test_matplotlib_viridis_cmap() -> None:
    """
    Get the viridis colormap from matplotlib
    """
    color = plt.get_cmap('viridis')
    try:  # try to convert from a matplotlib colormap
        if not color._isinit:  # type: ignore
            color._init()  # type: ignore
        cmap_values = (color._lut[: color.N, :3] * 255).astype('uint8')  # type: ignore
    except AttributeError:  # try to convert from a cmaps ColorMap
        try:
            cmap_values = (np.asarray(color.lut()[:, :3]) * 255).astype('uint8')  # type: ignore
        except AttributeError:  # must be a list of lists or an array castable to u8 (256, 3)
            cmap_values = np.asarray(color).astype('uint8')  # type: ignore
    assert cmap_values.shape == (256, 3)
    arr = np.random.randn(256, 256).astype('uint8')
    mc_colored = mc.apply_color_map(arr, cmap_values)
    np_colored = np.take(cmap_values, arr, axis=0)

    assert mc_colored.shape == (256, 256, 3)
    assert np_colored.shape == (256, 256, 3)
    assert mc_colored.shape == np_colored.shape

    mpl_colored = color(arr)
    assert mpl_colored.shape == (256, 256, 4)

    lut = (color(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    assert lut.shape == (256, 3)
