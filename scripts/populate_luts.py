# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///

"""
This script converts the luts in the `assets/converted` directory into
1) a lazy static hashmap that with the rgb values for each lut
2) a type hint file for the lut names so that the names are autocompleted in IDEs

This script can be invoked with `uv run scripts/populate_luts.py`
"""

from pathlib import Path

import numpy as np


def open_lut(txt_path: Path) -> np.ndarray:
    """
    Open the converted lut file
    """
    assert txt_path.exists(), f'{txt_path} does not exist'
    with open(txt_path, 'r') as f:
        lut = np.loadtxt(f)
    return lut.astype('uint8')


def main():
    """
    Build the lazy static hashmap and the type hint file for the lut names
    """
    curr_file = Path(__file__).absolute()
    assets_dir = curr_file.parent.parent / 'assets'
    subdirs = [
        d
        for d in assets_dir.iterdir()
        if d.is_dir() and d.name not in ['converted', 'samples'] and d.name != 'builtin_luts'
    ]

    lut_data: dict[str, np.ndarray] = {}
    sources = {}
    for subdir in subdirs:
        lut_files = [f.stem for f in subdir.iterdir() if f.is_file() and f.name.endswith('.lut')]
        if subdir.name == 'builtin_luts':
            special_luts = [
                'Fire',
                'Grays',
                'Ice',
                'Spectrum',
                '3-3-2 RGB',
                'Red',
                'Green',
                'Blue',
                'Cyan',
                'Magenta',
                'Yellow',
                'Red%Green',
            ]
            for special_name in special_luts:
                lut_files.append(special_name)
        for lut_file in lut_files:
            txt_path = assets_dir / 'converted' / f'{lut_file}.txt'
            lut = open_lut(txt_path)
            lut_name = txt_path.stem.replace('%', '/')
            lut_data[lut_name] = lut
            sources.setdefault(subdir.name, []).append((lut_name, lut))

    cmaps_file = curr_file.parent.parent / 'src' / 'cmaps.rs'
    assert cmaps_file.exists()

    cmaps_file_lines = ''
    cmaps_file_lines += 'use lazy_static::lazy_static;\n'
    cmaps_file_lines += 'use std::collections::HashMap;\n\n'
    cmaps_file_lines += 'type Colormap = [[u8; 3]; 256];\n\n'
    cmaps_file_lines += 'lazy_static! {\n'
    cmaps_file_lines += "\tpub static ref CMAPS: HashMap<&'static str, Colormap> = {\n"
    cmaps_file_lines += '\t\tlet mut m = HashMap::new();\n'

    for lut_name, lut in sorted(lut_data.items()):
        cmaps_file_lines += f'\t\tm.insert("{lut_name}", [\n'
        for i in range(256):
            cmaps_file_lines += f'\t\t\t[{lut[i, 0]}, {lut[i, 1]}, {lut[i, 2]}],\n'
        cmaps_file_lines += '\t\t]);\n\n'
    cmaps_file_lines += '\t\tm'
    cmaps_file_lines += '\t};'
    cmaps_file_lines += '}'

    with open(cmaps_file, 'w') as f:
        f.write(cmaps_file_lines)

    lut_names_type_hint_file = curr_file.parent.parent / 'python' / 'mergechannels' / '_luts.py'
    lut_names_lines = 'from typing import Literal\n\n'
    lut_names_lines += 'COLORMAPS = Literal[\n'
    for lut_name in sorted(lut_data.keys()):
        lut_names_lines += f"\t'{lut_name}',\n"
    lut_names_lines += ']\n'

    with open(lut_names_type_hint_file, 'w') as f:
        f.write(lut_names_lines)

    readme_path = curr_file.parent.parent / 'README.md'
    assert readme_path.exists()
    readme_lines = readme_path.read_text().splitlines()

    for idx, line in enumerate(readme_lines):
        if line.startswith('## Colormaps'):
            break
    readme_lines = readme_lines[: idx + 1]

    lut_mapping = {
        'FIJI built-in LUTs': {
            'source': 'builtin_luts',
            'link': None,
        },
        'My custom LUTs': {'source': 'zac_luts', 'link': 'https://github.com/zacswider/LUTs'},
        "Christophe Leterrier's LUTs": {
            'source': 'christ_luts',
            'link': 'https://github.com/cleterrier/ChrisLUTs',
        },
        "James Manton's LUTs": {
            'source': 'jdm_luts',
            'link': 'https://sites.imagej.net/JDM_LUTs/',
        },
    }

    for source_title, source_info in lut_mapping.items():
        source = source_info['source']
        link = source_info['link']
        luts = sources[source]
        if link is not None:
            source_title = f'[{source_title}]({link})'

        readme_lines.append(f'### {source_title}\n')
        for lut_name, _ in sorted(luts, key=lambda x: x[0]):
            lut_path = (
                'https://raw.githubusercontent.com/zacswider/README_Images/main'
                f'/{source}_{lut_name.replace("/", "%25")}.png'
            )
            readme_lines.append(
                (
                    f'<p>{lut_name.replace("%", "/")}: <img src="{lut_path}" '
                    'style="vertical-align: middle"></p>'
                )
            )
        readme_lines.append('')
    readme_path.write_text('\n'.join(readme_lines))


if __name__ == '__main__':
    main()
