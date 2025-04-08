pub fn max_blending(px_vals: &Vec<[u8; 3]>) -> [u8; 3] {
    let mut r: u8 = 0;
    let mut g: u8 = 0;
    let mut b: u8 = 0;
    for px_val in px_vals {
        if px_val[0] > r {
            r = px_val[0]
        }
        if px_val[1] > g {
            g = px_val[1]
        }
        if px_val[2] > b {
            b = px_val[2]
        }
    }
    [r, g, b]
}

pub fn sum_blending(px_vals: &Vec<[u8; 3]>) -> [u8; 3] {
    let mut r: u8 = 0;
    let mut g: u8 = 0;
    let mut b: u8 = 0;
    for px_val in px_vals {
        r = r.saturating_add(px_val[0]);
        g = g.saturating_add(px_val[1]);
        b = b.saturating_add(px_val[2]);
    }
    [r, g, b]
}

pub fn min_blending(px_vals: &Vec<[u8; 3]>) -> [u8; 3] {
    let mut r: u8 = 255;
    let mut g: u8 = 255;
    let mut b: u8 = 255;
    for px_val in px_vals {
        if px_val[0] < r {
            r = px_val[0]
        }
        if px_val[1] < g {
            g = px_val[1]
        }
        if px_val[2] < b {
            b = px_val[2]
        }
    }
    [r, g, b]
}

pub fn mean_blending(px_vals: &Vec<[u8; 3]>) -> [u8; 3] {
    let mut r: u16 = 0;
    let mut g: u16 = 0;
    let mut b: u16 = 0;
    for px_val in px_vals {
        r = r.saturating_add(px_val[0] as u16);
        g = g.saturating_add(px_val[1] as u16);
        b = b.saturating_add(px_val[2] as u16);
    }
    let n_channels: u16 = px_vals.len() as u16;
    let r: u8 = (r / n_channels) as u8;
    let g: u8 = (g / n_channels) as u8;
    let b: u8 = (b / n_channels) as u8;
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_blending() {
        let px_vals = vec![[100, 100, 100], [200, 200, 200]];
        let result = max_blending(&px_vals);
        assert_eq!(result, [200, 200, 200]);
    }

    #[test]
    fn test_sum_blending() {
        let px_vals = vec![[100, 100, 100], [200, 200, 200]];
        let result = sum_blending(&px_vals);
        assert_eq!(result, [255, 255, 255]);
    }

    #[test]
    fn test_min_blending() {
        let px_vals = vec![[100, 100, 100], [200, 200, 200]];
        let result = min_blending(&px_vals);
        assert_eq!(result, [100, 100, 100]);
    }

    #[test]
    fn test_mean_blending() {
        let px_vals = vec![[100, 100, 100], [200, 200, 200]];
        let result = mean_blending(&px_vals);
        assert_eq!(result, [150, 150, 150]);
    }
}
