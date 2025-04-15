# mergechannels

This project was originally conceived because I often find myself wanting to apply and blend colormaps to images while working from Python, and for historical reasons many of my favorite colormaps are distributed as [FIJI](https://imagej.net/software/fiji/) lookup tables. I also care about things likes speed and memory usage (e.g., not casting large arrays to floating point dtypes), so I was interested in seeing if I could at least match matplotlib's colormapping performance with my own hand-rolled colorizer in Rust.

There are other great colormapping libraries available (e.g., [microfilm](https://github.com/guiwitz/microfilm), [cmap](https://github.com/pyapp-kit/cmap)) that are more feature-rich than this projecting on your needs, but which don't address the my goals. My hope is that this project can fill and un-met niche and otherwise maintain full compatibility with these and similar libraries.

## Installation

Install pre-compiled binaries from [PyPI](https://pypi.org/project/mergechannels/):
```bash
pip install mergechannels
```

Build from source on your own machine:
```bash
pip install git+https://github.com/zacswider/mergechannels.git
```

## Usage
*NOTE*: scikit-image is not a dependency of this project, but is used in the examples below to fetch images.

### colorize a single image

```python
from skimage import data
import matplotlib.pyplot as plt
import mergechannels as mc

img = data.camera()
fig, ax = plt.subplots()
ax.imshow(mc.apply_colormap(img, 'Orange Hot'))
```

### apply a different colormap to each channel
```python
from skimage import data
import matplotlib.pyplot as plt
import mergechannels as mc

cells, nuclei = data.cells3d().max(axis=0)
cells = to_uint8(cells / cells.max())       <- broken until normalization is handled
nuclei = to_uint8(nuclei / nuclei.max())    <- broken until normalization is handled
fig, (a, b, c) = plt.subplots(1, 3)
a.imshow(cells, cmap='gray')
b.imshow(nuclei, cmap='gray')

import mergechannels as mc
c.imshow(mc.merge([cells, nuclei], ['Orange Hot', 'Cyan Hot']))
```

## Acknowledgements
This project incorporates a number of colormaps that were hand-crafted by Christophe Leterrier and were originally distributed here under the MIT license: https://github.com/cleterrier/ChrisLUTs