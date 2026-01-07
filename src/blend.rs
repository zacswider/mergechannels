use smallvec::SmallVec;

pub const MAX_N_CH: usize = 8;

pub type BlendFn = fn(&SmallVec<[[u8; 3]; MAX_N_CH]>) -> [u8; 3];

pub fn max_blending(px_vals: &SmallVec<[[u8; 3]; MAX_N_CH]>) -> [u8; 3] {
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

pub fn sum_blending(px_vals: &SmallVec<[[u8; 3]; MAX_N_CH]>) -> [u8; 3] {
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

pub fn min_blending(px_vals: &SmallVec<[[u8; 3]; MAX_N_CH]>) -> [u8; 3] {
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

pub fn mean_blending(px_vals: &SmallVec<[[u8; 3]; MAX_N_CH]>) -> [u8; 3] {
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

/// Alpha blend a mask color on top of a base color
/// Takes the base RGB color and blends the mask color on top with the given alpha
/// Formula: result = base * (1 - alpha) + mask * alpha
pub fn alpha_blend(base: [u8; 3], mask: [u8; 3], alpha: f32) -> [u8; 3] {
    let alpha = alpha.clamp(0.0, 1.0);
    let inv_alpha = 1.0 - alpha;
    [
        (base[0] as f32 * inv_alpha + mask[0] as f32 * alpha) as u8,
        (base[1] as f32 * inv_alpha + mask[1] as f32 * alpha) as u8,
        (base[2] as f32 * inv_alpha + mask[2] as f32 * alpha) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    fn create_test_data() -> SmallVec<[[u8; 3]; MAX_N_CH]> {
        let mut px_vals: SmallVec<[[u8; 3]; MAX_N_CH]> = SmallVec::new();
        px_vals.push([100, 100, 100]);
        px_vals.push([200, 200, 200]);
        px_vals
    }

    #[test]
    fn test_max_blending() {
        let px_vals = create_test_data();
        let result = max_blending(&px_vals);
        assert_eq!(result, [200, 200, 200]);
    }

    #[test]
    fn test_sum_blending() {
        let px_vals = create_test_data();
        let result = sum_blending(&px_vals);
        assert_eq!(result, [255, 255, 255]);
    }

    #[test]
    fn test_min_blending() {
        let px_vals = create_test_data();
        let result = min_blending(&px_vals);
        assert_eq!(result, [100, 100, 100]);
    }

    #[test]
    fn test_mean_blending() {
        let px_vals = create_test_data();
        let result = mean_blending(&px_vals);
        assert_eq!(result, [150, 150, 150]);
    }
}
