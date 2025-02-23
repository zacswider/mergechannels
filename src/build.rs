use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

use ndarray::Array2;

fn list_lut_files(dir: &str) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let paths = std::fs::read_dir(dir)?
        .filter_map(|res| res.ok())
        .map(|dir_entry| dir_entry.path())
        .filter_map(|path| {
            if path.extension().map_or(false, |ext| ext == "lut") {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    Ok(paths)
}

fn open_lut_file(lut_file: &PathBuf) -> Result<Array2<u8>, Box<dyn Error>> {
    let mut lut: Array2<u8> = Array2::zeros((3, 256));
    let file = File::open(lut_file)
        .map_err(|e| format!("Failed to open the file {}: {}", lut_file.display(), e))?;

    let reader = BufReader::new(file);
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| {
            format!(
                "Failed to read line {} of {}: {}",
                idx,
                lut_file.display(),
                e
            )
        })?;

        let intensity_mapping: Vec<u8> = line
            .split(' ')
            .map(|c| {
                c.parse::<u8>()
                    .map_err(|e| format!("Failed to parse '{}' to u8: {}", c, e))
            })
            .collect::<Result<Vec<u8>, String>>()?;

        for (ch, val) in intensity_mapping.iter().enumerate() {
            lut[[ch, idx]] = *val;
        }
    }

    Ok(lut)
}

// fn main() -> io::Result<()> {
//     let lut_dir = "/Users/zacswider/Desktop/luts";
//     let lut_files = list_lut_files(&lut_dir).expect("Failed to read files");
//     for lut_file in lut_files {
//         match open_lut_file(&lut_file) {
//             Ok(_) => println!("succesfully processed {}", lut_file.display()),
//             Err(_) => eprintln!("failed to process {}", lut_file.display()),
//         }
//     }

//     Ok(())
// }
