use ndarray::{prelude::*, stack};
use std::fs::File;
use std::io::{self, BufRead};

fn main() -> io::Result<()> {
    let path = "/Applications/Fiji.app/luts/sepia.lut";

    if let Ok(file) = File::open(&path) {
        let reader = io::BufReader::new(file);

        let mut lut: Array2<u8> = Array::zeros((3, 256));

        for (index, line) in reader.lines().enumerate() {
            let line = line?;
            let words: Vec<&str> = line.split(' ').collect();
            let nums: Vec<u8> = words.iter().map(|x| x.parse::<u8>().unwrap()).collect();

            for (ch, lut_val) in nums.iter().enumerate() {
                lut[[ch, index]] = *lut_val;
            }
        }

        println!("shape is {:?}", lut.shape())
    } else {
        println!("Failed to open the file");
    }

    Ok(())
}
