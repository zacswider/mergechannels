[![CI](https://github.com/zacswider/mergechannels/actions/workflows/CI.yml/badge.svg)](https://github.com/zacswider/mergechannels/actions/workflows/CI.yml)
![PyPI - License](https://img.shields.io/pypi/l/mergechannels)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mergechannels)
![PyPI](https://img.shields.io/pypi/v/mergechannels)

# mergechannels

This project was originally conceived because I often find myself wanting to apply and blend colormaps to images while working from Python, and for historical reasons many of my favorite colormaps are distributed as [FIJI](https://imagej.net/software/fiji/) lookup tables. I also care about things likes speed and memory usage (e.g., not casting large arrays to floating point dtypes, not creating multiple whole arrays just to add them together), so I was interested in seeing if I could at least match matplotlib's colormapping performance with my own hand-rolled colorizer in Rust.



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
colorized = mc.apply_color_map(img, 'Red/Green')
fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=150)
for ax in axes: ax.axis('off')
(a, b) = axes
a.imshow(img, cmap='gray')
b.imshow(colorized)
plt.show()
print(colorized.shape, colorized.dtype)
>> (512, 512, 3) uint8
```
![colorize a single image](https://raw.githubusercontent.com/zacswider/README_Images/main/camera_red-green.png)


### apply a different colormap to each channel
```python
from skimage import data
import matplotlib.pyplot as plt
import mergechannels as mc

cells, nuclei = data.cells3d().max(axis=0)
assert cells.dtype == 'uint16' and nuclei.dtype == 'uint16'
fig, axes = plt.subplots(1, 2, figsize=(3, 6), dpi=300)
for ax in axes.ravel(): ax.axis('off')
(a, b) = axes.ravel()
a.imshow(mc.merge([cells, nuclei],['Orange Hot', 'Cyan Hot']))
b.imshow(mc.merge([cells, nuclei],['I Blue', 'I Forest'], blending='min'))
fig.tight_layout()
plt.show()
```
![max and min multicolor blending](https://raw.githubusercontent.com/zacswider/README_Images/main/overlay_normal_and_inverted.png)

### apply a colormap to a whole stack
```python
from skimage import data
from matplotlib import pyplot as plt
import mergechannels as mc

volume = data.cells3d()
cells = volume[:, 0]
nuclei = volume[:, 1]
merged = mc.merge([cells, nuclei],['Orange Hot', 'Cyan Hot'])
plt.imshow(merged[24]); plt.show()
```
![colorize a whole stack of images](https://raw.githubusercontent.com/zacswider/README_Images/main/merged_stacks.png)

### adjust the saturation limits when applying colormaps
``` python
from skimage import data
import matplotlib.pyplot as plt
import mergechannels as mc

cells, nuclei = data.cells3d().max(axis=0)
channels = [cells, nuclei]
colormaps = ['I Blue', 'I Forest']
fig, axes = plt.subplots(1, 2, figsize=(3, 6), dpi=300)
for ax in axes.ravel(): ax.axis('off')
(a, b) = axes.ravel()
a.imshow(mc.merge(channels, colormaps, blending='min'))
b.imshow(
    mc.merge(
        channels,
        colormaps,
        blending='min',
        percentiles=[(
            1,  # bottom 1% of pixels set to black point
            97,  # top 3% of pixels set to white point
        )]*len(channels),
    ),
)
fig.tight_layout()
plt.show()
```
![adjust saturation limits](https://raw.githubusercontent.com/zacswider/README_Images/main/adjust_sat_lims.png)


## Roadmap
mergechannels is currently incredibly simple. It can apply one or more colormaps to one or more 2D and 3D 8-bit or 16-bit images and that's it.
- Add support for any numerical dtype
- Add option to return any colormap as a matplotlib colormap
- Add option to pass external colormaps to mergechannels
- Add support for directly passing matplotlib colormaps instead of colormap names
- Parallelize colormap application on large images (if it's helpful)
- Add option to overlay binary or instance masks onto colorized images

## Acknowledgements

There are other great colormapping libraries available (e.g., [microfilm](https://github.com/guiwitz/microfilm), [cmap](https://github.com/pyapp-kit/cmap)) that are more feature-rich than this one, but which don't address my goals. My hope is that this project can fill an un-met niche and otherwise maintain full compatibility with these and similar libraries.

This project incorporates a number of colormaps that were hand-crafted by Christophe Leterrier and were originally distributed here under the MIT license: https://github.com/cleterrier/ChrisLUTs
