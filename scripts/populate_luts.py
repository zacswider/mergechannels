# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///
import os
from pathlib import Path
import numpy as np


def main():
    curr_file = Path(__file__).absolute()
    luts_dir = curr_file.parent.parent / "assets" / "luts"
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
            lut_data[lut_file.stem] = lut
    
    cmaps_file = curr_file.parent.parent / "src" / "colorize" / "cmaps.rs"
    assert cmaps_file.exists()

    cmaps_file_lines = ''
    
    for lut_name, lut in lut_data.items():
        cmaps_file_lines += f'pub const {lut_name.upper()}: [[u8; 3]; 256] = [\n'
        for i in range(256):
            cmaps_file_lines += f'\t[{lut[i, 0]}, {lut[i, 1]}, {lut[i, 2]}],\n'
        cmaps_file_lines += '];\n\n'

    with open(cmaps_file, "w") as f:
        f.write(cmaps_file_lines)




if __name__ == "__main__":
    main()