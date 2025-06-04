[![CI](https://github.com/zacswider/mergechannels/actions/workflows/CI.yml/badge.svg)](https://github.com/zacswider/mergechannels/actions/workflows/CI.yml)
![PyPI - License](https://img.shields.io/pypi/l/mergechannels)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mergechannels)
![PyPI](https://img.shields.io/pypi/v/mergechannels)

# mergechannels

This project was originally conceived because I often find myself wanting to apply and blend colormaps to images while working from Python, and many of my favorite colormaps are distributed as [FIJI](https://imagej.net/software/fiji/) lookup tables. I also care about things likes speed and memory usage, so I was interested in seeing if I could at least match matplotlib's colormapping performance with my own hand-rolled colorizer in Rust.

The current goal of this library is to be a simple, fast, and memory-efficient way to apply and blend colormaps to images. The api should be intuitive, flexible, and simple. If this is not the case in your hands, please open an issue.


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
*NOTE*: `skimage`, `matplotlib`, and `cmap` are not dependencies of this project, but are used in the examples below to fetch data/colormaps, and display images.


### apply a different colormap to each channel
The primary entrypoint for merging multiple channels is with `mergechannels.merge`. This function expects a sequence of arrays, a sequence of colormaps, a blending approach, and optionally a sequence of pre-determined saturation limits. The arrays is expected to be either u8 or u16 and 2 or 3D.

```python
from skimage import data
import matplotlib.pyplot as plt
import mergechannels as mc

cells, nuclei = data.cells3d().max(axis=0)

fig, axes = plt.subplots(1, 3)
for ax in axes: ax.axis('off')
a, b, c = axes
a.imshow(cells, cmap='gray')
b.imshow(nuclei, cmap='gray')
c.imshow(mc.merge([cells, nuclei], ['Orange Hot', 'Cyan Hot']))
```
![simple channel blending](https://raw.githubusercontent.com/zacswider/README_Images/main/simple_channel_merge.png)

#### What constitutes a colormap?
Colormaps can be the literal name of one of the FIJI colormaps compiled into the mergechannels binary, a matplotlib colormap, or a [cmap](https://pypi.org/project/cmap/) colormap. The example below creates a similar blending as above, but by explicitly passing pre-generated colormaps (one from the matplotlib library, one from the cmap library). These can also be combined with string literals.

```python
import cmap
import matplotlib.pyplot as plt
from skimage import data
import mergechannels as mc

cells, nuclei = data.cells3d().max(axis=0)
blue = cmap.Colormap('seaborn:mako')
copper = plt.get_cmap('copper')

fig, axes = plt.subplots(1, 3)
for ax in axes: ax.axis('off')
a, b, c = axes
a.imshow(mc.apply_color_map(nuclei, blue))
b.imshow(mc.apply_color_map(cells, copper))
c.imshow(mc.merge([nuclei, cells], [blue, copper]))
```
![channel blending with external cmaps](https://raw.githubusercontent.com/zacswider/README_Images/main/external_cmaps.png)

#### What are my blending options?
The `blending` argument to `mergechannels.merge` can be one of the following:
- `'max'`: the maximum RGB value of each pixel is used. This is the default (and intuitive) behavior.
- `'min'`: the minimum RGB value of each pixel is used. This is useful when combining inverted colormaps.
- `'mean'`: the mean RGB value of each pixel is used. This is typically most useful when combinding fluorescence with brightfield, but can often require re-scaling the images after blending.
- `'sum'`: the sum of the RGB values of each pixel is used (saturating). Results in high saturation images but can often be overwhelming and difficult to interpret.

The default and intuitive behavior is the use `'max'` blending, but oftentimes minimum blending is desired when combining inverted colormaps.
```python
from skimage import data
import matplotlib.pyplot as plt
import mergechannels as mc

cells, nuclei = data.cells3d().max(axis=0)
fig, axes = plt.subplots(1, 3, dpi=200)
for ax in axes: ax.axis('off')
a, b, c = axes
a.imshow(cells, cmap='gray')
b.imshow(nuclei, cmap='gray')
c.imshow(mc.merge([cells, nuclei],['I Blue', 'I Forest'], blending='min'))
```
![minimum blending with inverted colormaps](https://raw.githubusercontent.com/zacswider/README_Images/main/inverted_blending.png)

#### How can I control display brightness?
If desired, pre-determined saturation limits can be passed to `apply_color_map` to clip the images values to a range that best represents the contents of the image. These can be explicit pixel values passed with the `saturation_limits` argument, or as percentile values passed with the `percentiles` argument. If the latter, the percentile values will be used to calculate the saturation limits based on the distribution of pixel values in the images (this is sometimes referred to as "autoscaling). The default behavior is to calculate use the 1.1th percentile value as the dark point and the 99.9th percentile as the bright point.

```python
from skimage import data
import matplotlib.pyplot as plt
import mergechannels as mc

cells, nuclei = data.cells3d().max(axis=0)
channels = [cells, nuclei]
colormaps = ['I Blue', 'I Forest']
fig, axes = plt.subplots(1, 3, dpi=300)
for ax in axes: ax.axis('off')
(a, b, c) = axes
a.imshow(mc.merge(channels, colormaps, blending='min'))  # use the default autoscaling
b.imshow(
    mc.merge(
        channels,
        colormaps,
        blending='min',
        saturation_limits=[
            (1000, 20_000), # pre-determined dark and light points for ch1
            (1000, 50_000), # pre-determined dark and light points for ch2
        ],
    ),
)
c.imshow(
    mc.merge(
        channels,
        colormaps,
        blending='min',
        percentiles=[(
            1,  # bottom 1% of pixels set to black point
            97,  # top 3% of pixels set to white point
        )]*len(channels), # apply this to all channels
    ),
)
```
![adjust the brightness with explicit of percentile based approaches](https://raw.githubusercontent.com/zacswider/README_Images/main/brightness_adjust.png)


NOTE: if you are already working with appropriately scaled u8 images, you will see ~10X performance improvements (relative to the mergechannels and matplotlib naive default implementations) by passing `saturation_limits=(0, 255)` as this significantly reduces the amount of arithmetic done per pixel.


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


### colorize a single image
The primary entrypoint to applying a colormap to a single image is with `apply_color_map`. this function takes an array, a colormap, and an optional percentiles argument. The arrays is expected to be either u8 or u16 and 2 or 3D.

```python
from skimage import data
import matplotlib.pyplot as plt
import mergechannels as mc

img = data.camera()
colorized = mc.apply_color_map(img, 'Red/Green')

fig, axes = plt.subplots(1, 2)
for ax in axes: ax.axis('off')
(a, b) = axes
a.imshow(img, cmap='gray')
b.imshow(colorized)
plt.show()
print(colorized.shape, colorized.dtype)
>> (512, 512, 3) uint8
```
![colorize a single image](https://raw.githubusercontent.com/zacswider/README_Images/main/camera_red-green.png)

Similar to `mergechannels.merge`, `apply_color_map` will also accept colormaps directly from `matplotlib` and `cmap`, explicit saturation limits, or perctile values for autoscaling.

## Roadmap
mergechannels is currently incredibly simple. It can apply one or more colormaps to one or more 2/3D 8/16-bit images and that's it.
- ~~Add support to u8 and u16 images~~
- Add support for any numerical dtype
- Add option to return any colormap as a matplotlib colormap
- ~~Add option to pass external colormaps to mergechannels~~
- Parallelize colormap application on large images (if it's helpful)
- Add option to overlay binary or instance masks onto colorized images

## Acknowledgements

There are other great colormapping libraries available (e.g., [microfilm](https://github.com/guiwitz/microfilm), [cmap](https://github.com/pyapp-kit/cmap)) that are more feature-rich than this one, but which don't address my goals. My hope is that this project can fill an un-met niche and otherwise maintain full compatibility with these and similar libraries.

This project incorporates a number of colormaps that were hand-crafted by Christophe Leterrier and James Manton which were originally distributed [here](https://github.com/cleterrier/ChrisLUTs) and [here](https://sites.imagej.net/JDM_LUTs/) respectively.
