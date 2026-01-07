[![CI](https://github.com/zacswider/mergechannels/actions/workflows/CI.yml/badge.svg)](https://github.com/zacswider/mergechannels/actions/workflows/CI.yml)
![PyPI - License](https://img.shields.io/pypi/l/mergechannels)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mergechannels)
![PyPI](https://img.shields.io/pypi/v/mergechannels)

# mergechannels

This project was originally conceived because I often find myself wanting to apply and blend colormaps to images while working from Python, and many of my favorite colormaps are distributed as [FIJI](https://imagej.net/software/fiji/) lookup tables. I also care about things like speed and memory usage, so I was interested in seeing if I could at least match matplotlib's colormapping performance with my own hand-rolled colorizer in Rust (success! mergechannels is typically several times faster than matplotlib).

The current goal of this library is to be a simple, fast, and memory-efficient way to apply and blend colormaps to images. The api should be intuitive, flexible, and simple. If this is not the case in your hands, please open an issue.


## Installation

Install pre-compiled binaries from [PyPI](https://pypi.org/project/mergechannels/):
```bash
pip install mergechannels
```

Build from source on your own machine (requires [Rust toolchain](https://rustup.rs/)):
```bash
pip install git+https://github.com/zacswider/mergechannels.git
```


## Usage
*NOTE*: `skimage`, `matplotlib`, and `cmap` are not dependencies of this project, but are used in the examples below to fetch data/colormaps, and display images.


### Apply a different colormap to each channel
The primary entrypoint for merging multiple channels is with `mergechannels.merge`. This function expects a sequence of arrays, a sequence of colormaps, a blending approach, and optionally a sequence of pre-determined saturation limits. The arrays are expected to be either u8 or u16 and 2 or 3D.

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

#### Using external colormaps
Colormaps can be the literal name of one of the colormaps compiled into the mergechannels binary (see a list as the bottom of the page), a matplotlib colormap, or a [cmap](https://pypi.org/project/cmap/) colormap. The example below creates a similar blending as above, but by explicitly passing pre-generated colormaps (one from the matplotlib library, one from the cmap library). These can also be combined with string literals.

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

#### Blending options
The `blending` argument to `mergechannels.merge` can be one of the following:
- `'max'`: the maximum RGB value of each pixel is used. This is the default (and intuitive) behavior.
- `'min'`: the minimum RGB value of each pixel is used. This is useful when combining inverted colormaps.
- `'mean'`: the mean RGB value of each pixel is used. This is typically most useful when combining fluorescence with brightfield, but can often require re-scaling the images after blending.
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

#### Control display brightness
If desired, pre-determined saturation limits can be passed to `apply_color_map` or `merge` to clip the images values to a range that best represents the contents of the image. These can be explicit pixel values passed with the `saturation_limits` argument, or as percentile values passed with the `percentiles` argument. If the latter, the percentile values will be used to calculate the saturation limits based on the distribution of pixel values in the images (this is sometimes referred to as "autoscaling"). The default behavior is to calculate use the 1.1th percentile value as the dark point and the 99.9th percentile as the bright point.

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
![adjust the brightness with explicit or percentile based approaches](https://raw.githubusercontent.com/zacswider/README_Images/main/brightness_adjust.png)


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
The primary entrypoint to applying a colormap to a single image is with `apply_color_map`. this function takes an array, a colormap, and an optional percentiles argument. The array is expected to be either u8 or u16 and 2 or 3D.

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

Similar to `mergechannels.merge`, `apply_color_map` will also accept colormaps directly from `matplotlib` and `cmap`, explicit saturation limits, or percentile values for autoscaling.

### export colormaps to matplotlib
If you want to use mergechannels' built-in colormaps with matplotlib directly, you can export them as `matplotlib.colors.ListedColormap` objects using `get_mpl_cmap`. This requires matplotlib to be installed, which can be done with the optional dependency:

```bash
uv pip install matplotlib
```
or:
```bash
uv pip install "mergechannels[matplotlib]>=0.5.5"
```

```python
from skimage import data
import matplotlib.pyplot as plt
import mergechannels as mc

img = data.camera()

# Get a mergechannels colormap as a matplotlib ListedColormap
cmap = mc.get_mpl_cmap('Red/Green')

# Use it directly with matplotlib
fig, axes = plt.subplots(1, 2)
for ax in axes: ax.axis('off')
(a, b) = axes
a.imshow(img, cmap='gray')
b.imshow(img, cmap=cmap)
plt.show()
```
![colorize a single image](https://raw.githubusercontent.com/zacswider/README_Images/main/camera_red-green.png)


You can also retrieve the raw colormap data as a numpy array using `get_cmap_array`:

```python
import mergechannels as mc

# Get the raw (256, 3) uint8 array of RGB values
cmap_array = mc.get_cmap_array('betterBlue')
print(cmap_array.shape, cmap_array.dtype)
>> (256, 3) uint8
```


#### Overlay segmentation masks on top of colorized/merged channels
Both `apply_color_map` and `merge` support overlaying binary or instance masks on top of the colorized images. The examples below use `apply_color_map` but the arguments are identical for `merge`.

```python
from skimage import data
from scipy import ndimage
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# get an image and create some masks
_, nuclei = data.cells3d().max(axis=0)
thresh = nuclei > threshold_otsu(nuclei)
thresh = ndimage.binary_fill_holes(thresh)
max_filter = ndimage.maximum_filter(thresh, size=3, mode='reflect')
min_filter = ndimage.minimum_filter(thresh, size=3, mode='reflect')
boundaries = max_filter != min_filter

# overlay the masks with mergechannels
import mergechannels as mc

fig, (a, b, c) = plt.subplots(1, 3, dpi=300)
for ax in (a, b, c): ax.axis('off')
a.imshow(mc.apply_color_map(nuclei, 'betterBlue'))  # no overlay
b.imshow(mc.apply_color_map(nuclei, 'betterBlue', masks=[boundaries]))  # add mask overlay
c.imshow(mc.apply_color_map(nuclei, 'betterBlue', masks=[boundaries], mask_colors=['#f00']))  # non-default color
plt.show()
```
![Overlay a single mask array with different color settings](https://raw.githubusercontent.com/zacswider/README_Images/main/overlay_masks.png)


Multiple masks can be overlaid with different color or alpha-blending values.

```python
from skimage import (
    data,
    measure,
)
from scipy import ndimage
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt

# get an image and create some masks
cells, nuclei = data.cells3d().max(axis=0)
thresh = nuclei > threshold_otsu(nuclei)
labels = np.asarray(measure.label(ndimage.binary_fill_holes(thresh)))
bright_nuclei_threshold = threshold_otsu(nuclei[thresh])
label_vals = [l for l in np.unique(labels) if l!=0]

# categorize two different types of nuclei masks
def only_keep_these_labels(arr, labels):
    out = arr.copy()
    mask = np.isin(out, labels)
    out[~mask] = 0
    return out

bright_nuclei_labels = [lv for lv in label_vals if nuclei[labels == lv].mean() > bright_nuclei_threshold ]
bright_nuclei_masks = only_keep_these_labels(
    arr=labels,
    labels=bright_nuclei_labels,
)
dim_nuclei_masks = only_keep_these_labels(
    arr=labels,
    labels=[lv for lv in label_vals if lv not in bright_nuclei_labels],
)

# overlay the masks with mergechannels
import mergechannels as mc

fig, (a, b, c) = plt.subplots(1, 3, dpi=300)
for ax in (a, b, c):
    ax.axis('off')

a.imshow(bright_nuclei_masks, cmap=mc.get_mpl_cmap('glasbey'))
b.imshow(dim_nuclei_masks, cmap=mc.get_mpl_cmap('glasbey'))
c.imshow(
    mc.apply_color_map(
        arr=nuclei,
        color='Grays',
        masks=[bright_nuclei_masks, dim_nuclei_masks],
        mask_colors=['betterOrange', 'betterBlue'],
        mask_alphas=[0.2, 0.2],
    )
)  # add mask overlay
plt.show()
```
![Overlay multiple mask arrays with different colors](https://raw.githubusercontent.com/zacswider/README_Images/main/overlay_masks_different_color.png)


Mask color specifications accept:
- Colormap names (see an exhaustive list at the bottom of README or print mergechannels.COLORMAPS)
- Hex strings (e.g., `'#FF0000'`)
- RGB tuples or sequences (e.g., `(255, 0, 0)` or `[255, 0, 0]`)


## Dependencies
Mergechannels only depends on numpy, a matrix of compatible versions is shown below. Mergechannels can also interop with matplotlib and cmap (see the `Usage` sections below), but these dependencies are optional for core functionality.

| Python | 1.25.0 | 1.26.0 | 2.0.0 | 2.1.0 | 2.2.0 | 2.3.0 | 2.4.0 |
|--------|--------|--------|-------|-------|-------|-------|-------|
| 3.9    | ✅      | ✅      | ✅     | ❌     | ❌     | ❌     | ❌     |
| 3.10   | ✅      | ✅      | ✅     | ✅     | ✅     | ❌     | ❌     |
| 3.11   | ✅      | ✅      | ✅     | ✅     | ✅     | ✅     | ✅     |
| 3.12   | ❌      | ✅      | ✅     | ✅     | ✅     | ✅     | ✅     |
| 3.13   | ❌      | ❌      | ✅     | ✅     | ✅     | ✅     | ✅     |
| 3.14   | ❌      | ❌      | ❌     | ❌     | ❌     | ❌     | ✅     |
| 3.14t   | ❌      | ❌      | ❌     | ❌     | ❌     | ❌     | ✅     |


## Threading and Parallelism
Mergechannels is fully compatible with free-threaded Python (3.13t/3.14t). The extension declares itself thread-safe (`gil_used(false)`), so it won't re-enable the GIL in no-GIL builds, enabling true parallelism with Python's `ThreadPoolExecutor`.

By default, `parallel=True` uses [Rayon](https://github.com/rayon-rs/rayon) for internal parallelization across image rows/planes. This also works well alongside Python threading.

To configure Rayon's thread count, set the `RAYON_NUM_THREADS` environment variable **before** importing mergechannels:
```python
import os
os.environ['RAYON_NUM_THREADS'] = '4'  # Must be set before import
import mergechannels as mc
```


## Performance

Benchmarks show that with appropriately scaled images, (i.e., if pre-determined saturation limits are passed to `mc.merge` or `mc.apply_color_map`) mergechannel is either on par or significantly faster than the underlying numpy operations used by Matplotlib. Note: you can run the benchmarks on your own machine by creating a virtual environment with the dev dependencies `uv sync --dev && source .venv/bin/activate` and running the benchmark code `py.test --benchmark-only`

## Roadmap
- ~~Add support to u8 and u16 images~~
- Add support for any numerical dtype
- ~~Add option to return any colormap as a matplotlib colormap~~
- ~~Add option to pass external colormaps to mergechannels~~
- ~~Parallelize colormap application on large images (it is helpful!)~~
- ~~Add option to overlay binary or instance masks onto colorized images~~
- ~~Add support for free-threaded Python~~

## Acknowledgements

There are other great colormapping libraries available (e.g., [microfilm](https://github.com/guiwitz/microfilm), [cmap](https://github.com/pyapp-kit/cmap)) that are more feature-rich than this one, but which don't address my goals. My hope is that this project can fill an un-met niche and otherwise maintain full compatibility with these and similar libraries.

This project incorporates a number of colormaps that were hand-crafted by Christophe Leterrier and James Manton which were originally distributed [here](https://github.com/cleterrier/ChrisLUTs) and [here](https://sites.imagej.net/JDM_LUTs/) respectively.

## Colormaps
### FIJI built-in LUTs

<p>16_colors: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_16_colors.png" style="vertical-align: middle"></p>
<p>3-3-2 RGB: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_3-3-2 RGB.png" style="vertical-align: middle"></p>
<p>5_ramps: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_5_ramps.png" style="vertical-align: middle"></p>
<p>6_shades: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_6_shades.png" style="vertical-align: middle"></p>
<p>Blue: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Blue.png" style="vertical-align: middle"></p>
<p>Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Cyan.png" style="vertical-align: middle"></p>
<p>Cyan Hot: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Cyan Hot.png" style="vertical-align: middle"></p>
<p>Fire: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Fire.png" style="vertical-align: middle"></p>
<p>Grays: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Grays.png" style="vertical-align: middle"></p>
<p>Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Green.png" style="vertical-align: middle"></p>
<p>Green Fire Blue: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Green Fire Blue.png" style="vertical-align: middle"></p>
<p>HiLo: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_HiLo.png" style="vertical-align: middle"></p>
<p>ICA: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_ICA.png" style="vertical-align: middle"></p>
<p>ICA2: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_ICA2.png" style="vertical-align: middle"></p>
<p>ICA3: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_ICA3.png" style="vertical-align: middle"></p>
<p>Ice: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Ice.png" style="vertical-align: middle"></p>
<p>Magenta: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Magenta.png" style="vertical-align: middle"></p>
<p>Magenta Hot: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Magenta Hot.png" style="vertical-align: middle"></p>
<p>Orange Hot: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Orange Hot.png" style="vertical-align: middle"></p>
<p>Rainbow RGB: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Rainbow RGB.png" style="vertical-align: middle"></p>
<p>Red: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Red.png" style="vertical-align: middle"></p>
<p>Red Hot: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Red Hot.png" style="vertical-align: middle"></p>
<p>Red/Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Red%25Green.png" style="vertical-align: middle"></p>
<p>Spectrum: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Spectrum.png" style="vertical-align: middle"></p>
<p>Thermal: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Thermal.png" style="vertical-align: middle"></p>
<p>Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Yellow.png" style="vertical-align: middle"></p>
<p>Yellow Hot: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_Yellow Hot.png" style="vertical-align: middle"></p>
<p>blue_orange_icb: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_blue_orange_icb.png" style="vertical-align: middle"></p>
<p>brgbcmyw: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_brgbcmyw.png" style="vertical-align: middle"></p>
<p>cool: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_cool.png" style="vertical-align: middle"></p>
<p>edges: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_edges.png" style="vertical-align: middle"></p>
<p>gem: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_gem.png" style="vertical-align: middle"></p>
<p>glasbey: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_glasbey.png" style="vertical-align: middle"></p>
<p>glasbey_inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_glasbey_inverted.png" style="vertical-align: middle"></p>
<p>glasbey_on_dark: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_glasbey_on_dark.png" style="vertical-align: middle"></p>
<p>glow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_glow.png" style="vertical-align: middle"></p>
<p>mpl-inferno: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_mpl-inferno.png" style="vertical-align: middle"></p>
<p>mpl-magma: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_mpl-magma.png" style="vertical-align: middle"></p>
<p>mpl-plasma: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_mpl-plasma.png" style="vertical-align: middle"></p>
<p>mpl-viridis: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_mpl-viridis.png" style="vertical-align: middle"></p>
<p>phase: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_phase.png" style="vertical-align: middle"></p>
<p>physics: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_physics.png" style="vertical-align: middle"></p>
<p>royal: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_royal.png" style="vertical-align: middle"></p>
<p>sepia: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_sepia.png" style="vertical-align: middle"></p>
<p>smart: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_smart.png" style="vertical-align: middle"></p>
<p>thal: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_thal.png" style="vertical-align: middle"></p>
<p>thallium: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_thallium.png" style="vertical-align: middle"></p>
<p>unionjack: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/builtin_luts_unionjack.png" style="vertical-align: middle"></p>

### [My custom LUTs](https://github.com/zacswider/LUTs)

<p>OIMB1: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/zac_luts_OIMB1.png" style="vertical-align: middle"></p>
<p>OIMB2: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/zac_luts_OIMB2.png" style="vertical-align: middle"></p>
<p>OIMB3: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/zac_luts_OIMB3.png" style="vertical-align: middle"></p>
<p>betterBlue: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/zac_luts_betterBlue.png" style="vertical-align: middle"></p>
<p>betterCyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/zac_luts_betterCyan.png" style="vertical-align: middle"></p>
<p>betterGreen: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/zac_luts_betterGreen.png" style="vertical-align: middle"></p>
<p>betterOrange: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/zac_luts_betterOrange.png" style="vertical-align: middle"></p>
<p>betterRed: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/zac_luts_betterRed.png" style="vertical-align: middle"></p>
<p>betterYellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/zac_luts_betterYellow.png" style="vertical-align: middle"></p>

### [Christophe Leterrier's LUTs](https://github.com/cleterrier/ChrisLUTs)

<p>3color-BMR: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_3color-BMR.png" style="vertical-align: middle"></p>
<p>3color-CGY: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_3color-CGY.png" style="vertical-align: middle"></p>
<p>3color-RMB: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_3color-RMB.png" style="vertical-align: middle"></p>
<p>3color-YGC: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_3color-YGC.png" style="vertical-align: middle"></p>
<p>BOP blue: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_BOP blue.png" style="vertical-align: middle"></p>
<p>BOP orange: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_BOP orange.png" style="vertical-align: middle"></p>
<p>BOP purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_BOP purple.png" style="vertical-align: middle"></p>
<p>I Blue: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_I Blue.png" style="vertical-align: middle"></p>
<p>I Bordeaux: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_I Bordeaux.png" style="vertical-align: middle"></p>
<p>I Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_I Cyan.png" style="vertical-align: middle"></p>
<p>I Forest: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_I Forest.png" style="vertical-align: middle"></p>
<p>I Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_I Green.png" style="vertical-align: middle"></p>
<p>I Magenta: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_I Magenta.png" style="vertical-align: middle"></p>
<p>I Purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_I Purple.png" style="vertical-align: middle"></p>
<p>I Red: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_I Red.png" style="vertical-align: middle"></p>
<p>I Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_I Yellow.png" style="vertical-align: middle"></p>
<p>OPF fresh: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_OPF fresh.png" style="vertical-align: middle"></p>
<p>OPF orange: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_OPF orange.png" style="vertical-align: middle"></p>
<p>OPF purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_OPF purple.png" style="vertical-align: middle"></p>
<p>Turbo: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/christ_luts_Turbo.png" style="vertical-align: middle"></p>

### [James Manton's LUTs](https://sites.imagej.net/JDM_LUTs/)

<p>Circus Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Cherry.png" style="vertical-align: middle"></p>
<p>Circus Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Cyan.png" style="vertical-align: middle"></p>
<p>Circus Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Green.png" style="vertical-align: middle"></p>
<p>Circus Ink Black: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Ink Black.png" style="vertical-align: middle"></p>
<p>Circus Ink Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Ink Cherry.png" style="vertical-align: middle"></p>
<p>Circus Ink Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Ink Cyan.png" style="vertical-align: middle"></p>
<p>Circus Ink Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Ink Green.png" style="vertical-align: middle"></p>
<p>Circus Ink Mint: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Ink Mint.png" style="vertical-align: middle"></p>
<p>Circus Ink Purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Ink Purple.png" style="vertical-align: middle"></p>
<p>Circus Ink Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Ink Yellow.png" style="vertical-align: middle"></p>
<p>Circus Mint: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Mint.png" style="vertical-align: middle"></p>
<p>Circus Purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Purple.png" style="vertical-align: middle"></p>
<p>Circus Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Circus Yellow.png" style="vertical-align: middle"></p>
<p>Duo Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Cherry.png" style="vertical-align: middle"></p>
<p>Duo Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Cyan.png" style="vertical-align: middle"></p>
<p>Duo Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Green.png" style="vertical-align: middle"></p>
<p>Duo Intense Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Intense Cherry.png" style="vertical-align: middle"></p>
<p>Duo Intense Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Intense Cyan.png" style="vertical-align: middle"></p>
<p>Duo Intense Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Intense Green.png" style="vertical-align: middle"></p>
<p>Duo Intense Mint: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Intense Mint.png" style="vertical-align: middle"></p>
<p>Duo Intense Purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Intense Purple.png" style="vertical-align: middle"></p>
<p>Duo Intense Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Intense Yellow.png" style="vertical-align: middle"></p>
<p>Duo Mint: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Mint.png" style="vertical-align: middle"></p>
<p>Duo Purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Purple.png" style="vertical-align: middle"></p>
<p>Duo Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Duo Yellow.png" style="vertical-align: middle"></p>
<p>Grays g=0.25: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=0.25.png" style="vertical-align: middle"></p>
<p>Grays g=0.25 inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=0.25 inverted.png" style="vertical-align: middle"></p>
<p>Grays g=0.50: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=0.50.png" style="vertical-align: middle"></p>
<p>Grays g=0.50 inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=0.50 inverted.png" style="vertical-align: middle"></p>
<p>Grays g=0.75: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=0.75.png" style="vertical-align: middle"></p>
<p>Grays g=0.75 inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=0.75 inverted.png" style="vertical-align: middle"></p>
<p>Grays g=1.00 inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=1.00 inverted.png" style="vertical-align: middle"></p>
<p>Grays g=1.25: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=1.25.png" style="vertical-align: middle"></p>
<p>Grays g=1.25 inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=1.25 inverted.png" style="vertical-align: middle"></p>
<p>Grays g=1.50: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=1.50.png" style="vertical-align: middle"></p>
<p>Grays g=1.50 inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=1.50 inverted.png" style="vertical-align: middle"></p>
<p>Grays g=1.75: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=1.75.png" style="vertical-align: middle"></p>
<p>Grays g=1.75 inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=1.75 inverted.png" style="vertical-align: middle"></p>
<p>Grays g=2.00: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=2.00.png" style="vertical-align: middle"></p>
<p>Grays g=2.00 inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Grays g=2.00 inverted.png" style="vertical-align: middle"></p>
<p>Ink Black: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Black.png" style="vertical-align: middle"></p>
<p>Ink Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Cherry.png" style="vertical-align: middle"></p>
<p>Ink Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Cyan.png" style="vertical-align: middle"></p>
<p>Ink Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Green.png" style="vertical-align: middle"></p>
<p>Ink Mint: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Mint.png" style="vertical-align: middle"></p>
<p>Ink Purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Purple.png" style="vertical-align: middle"></p>
<p>Ink Wash Black: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Wash Black.png" style="vertical-align: middle"></p>
<p>Ink Wash Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Wash Cherry.png" style="vertical-align: middle"></p>
<p>Ink Wash Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Wash Cyan.png" style="vertical-align: middle"></p>
<p>Ink Wash Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Wash Green.png" style="vertical-align: middle"></p>
<p>Ink Wash Mint: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Wash Mint.png" style="vertical-align: middle"></p>
<p>Ink Wash Purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Wash Purple.png" style="vertical-align: middle"></p>
<p>Ink Wash Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Wash Yellow.png" style="vertical-align: middle"></p>
<p>Ink Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Ink Yellow.png" style="vertical-align: middle"></p>
<p>Parabolic RGB: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Parabolic RGB.png" style="vertical-align: middle"></p>
<p>Phase Bold Green-Magenta: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Bold Green-Magenta.png" style="vertical-align: middle"></p>
<p>Phase Bold Ink Green-Magenta: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Bold Ink Green-Magenta.png" style="vertical-align: middle"></p>
<p>Phase Bold Ink Mint-Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Bold Ink Mint-Cherry.png" style="vertical-align: middle"></p>
<p>Phase Bold Ink Yellow-Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Bold Ink Yellow-Cyan.png" style="vertical-align: middle"></p>
<p>Phase Bold Mint-Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Bold Mint-Cherry.png" style="vertical-align: middle"></p>
<p>Phase Bold Yellow-Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Bold Yellow-Cyan.png" style="vertical-align: middle"></p>
<p>Phase Green-Magenta: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Green-Magenta.png" style="vertical-align: middle"></p>
<p>Phase Ink Green-Magenta: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Ink Green-Magenta.png" style="vertical-align: middle"></p>
<p>Phase Ink Mint-Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Ink Mint-Cherry.png" style="vertical-align: middle"></p>
<p>Phase Ink Yellow-Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Ink Yellow-Cyan.png" style="vertical-align: middle"></p>
<p>Phase Mint-Cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Mint-Cherry.png" style="vertical-align: middle"></p>
<p>Phase Yellow-Cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Phase Yellow-Cyan.png" style="vertical-align: middle"></p>
<p>Pop blue: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop blue.png" style="vertical-align: middle"></p>
<p>Pop blue inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop blue inverted.png" style="vertical-align: middle"></p>
<p>Pop cherry: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop cherry.png" style="vertical-align: middle"></p>
<p>Pop cherry inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop cherry inverted.png" style="vertical-align: middle"></p>
<p>Pop cyan: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop cyan.png" style="vertical-align: middle"></p>
<p>Pop cyan inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop cyan inverted.png" style="vertical-align: middle"></p>
<p>Pop green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop green.png" style="vertical-align: middle"></p>
<p>Pop green inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop green inverted.png" style="vertical-align: middle"></p>
<p>Pop lavender inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop lavender inverted.png" style="vertical-align: middle"></p>
<p>Pop magenta: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop magenta.png" style="vertical-align: middle"></p>
<p>Pop magenta inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop magenta inverted.png" style="vertical-align: middle"></p>
<p>Pop mint: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop mint.png" style="vertical-align: middle"></p>
<p>Pop mint inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop mint inverted.png" style="vertical-align: middle"></p>
<p>Pop purple: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop purple.png" style="vertical-align: middle"></p>
<p>Pop purple inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop purple inverted.png" style="vertical-align: middle"></p>
<p>Pop red: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop red.png" style="vertical-align: middle"></p>
<p>Pop red inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop red inverted.png" style="vertical-align: middle"></p>
<p>Pop yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop yellow.png" style="vertical-align: middle"></p>
<p>Pop yellow inverted: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Pop yellow inverted.png" style="vertical-align: middle"></p>
<p>Quartetto MYGB Blue: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Quartetto MYGB Blue.png" style="vertical-align: middle"></p>
<p>Quartetto MYGB Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Quartetto MYGB Green.png" style="vertical-align: middle"></p>
<p>Quartetto MYGB Magenta: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Quartetto MYGB Magenta.png" style="vertical-align: middle"></p>
<p>Quartetto MYGB Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Quartetto MYGB Yellow.png" style="vertical-align: middle"></p>
<p>Quartetto RYGB Blue: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Quartetto RYGB Blue.png" style="vertical-align: middle"></p>
<p>Quartetto RYGB Green: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Quartetto RYGB Green.png" style="vertical-align: middle"></p>
<p>Quartetto RYGB Red: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Quartetto RYGB Red.png" style="vertical-align: middle"></p>
<p>Quartetto RYGB Yellow: <img src="https://raw.githubusercontent.com/zacswider/README_Images/main/jdm_luts_Quartetto RYGB Yellow.png" style="vertical-align: middle"></p>
