# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///

'''
This script converts the luts in the `assets/converted` directory into 
1) a lazy static hashmap that with the rgb values for each lut
2) a type hint file for the lut names so that the names are autocompleted in IDEs

This script can be invoked with `uv run scripts/populate_luts.py`
'''

from pathlib import Path
import numpy as np


def main():
    curr_file = Path(__file__).absolute()
    luts_dir = curr_file.parent.parent / "assets" / "converted"
    lut_files = list(luts_dir.glob("*.txt"))
    lut_data: dict[str, np.ndarray] = {}
    for lut_file in lut_files:
        with open(lut_file, "r") as f:
            lut_vals = []
            for line in f.readlines():
                line_vals = np.array(line.strip().split('\t'), dtype='uint8')
                assert len(line_vals) == 3
                lut_vals.append(line_vals)
            lut = np.array(lut_vals, dtype='uint8')
            assert lut.shape == (256, 3)
            lut_data[lut_file.stem.replace('%', '/')] = lut
    
    cmaps_file = curr_file.parent.parent / "src" / "cmaps.rs"
    assert cmaps_file.exists()

    cmaps_file_lines = ""
    cmaps_file_lines += "use lazy_static::lazy_static;\n"
    cmaps_file_lines += "use std::collections::HashMap;\n\n"
    cmaps_file_lines += "type Colormap = [[u8; 3]; 256];\n\n"
    cmaps_file_lines += "lazy_static! {\n"
    cmaps_file_lines += "\tpub static ref CMAPS: HashMap<&'static str, Colormap> = {\n"
    cmaps_file_lines += "\t\tlet mut m = HashMap::new();\n"
    
    for lut_name, lut in lut_data.items():
        cmaps_file_lines += f'\t\tm.insert("{lut_name}", [\n'
        for i in range(256):
            cmaps_file_lines += f'\t\t\t[{lut[i, 0]}, {lut[i, 1]}, {lut[i, 2]}],\n'
        cmaps_file_lines += '\t\t]);\n\n'
    cmaps_file_lines += "\t\tm"
    cmaps_file_lines += "\t};"
    cmaps_file_lines += "}"

    with open(cmaps_file, "w") as f:
        f.write(cmaps_file_lines)

    lut_names_type_hint_file = curr_file.parent.parent / 'python' / 'mergechannels' / '_luts.py'
    lut_names_lines = 'from typing import Literal\n\n'
    lut_names_lines += 'COLORMAPS = Literal[\n'
    for lut_name in lut_data:
        lut_names_lines += f"\t'{lut_name}',\n"
    lut_names_lines += ']\n'

    with open(lut_names_type_hint_file, 'w') as f:
        f.write(lut_names_lines)


if __name__ == "__main__":
    main()