use ndarray::{Array2, ArrayView2};

/// Check if a 3x3 window has non-uniform values using direct slice access.
/// This is the fast path for interior pixels where no bounds checking is needed.
#[inline(always)]
fn is_boundary_interior<T: PartialEq + Copy>(arr: &ArrayView2<T>, y: usize, x: usize) -> bool {
    // SAFETY: Caller guarantees y in [1, h-2] and x in [1, w-2]
    unsafe {
        let first = *arr.uget((y - 1, x - 1));

        // Check all 9 positions, early exit on difference
        // Row y-1
        if *arr.uget((y - 1, x)) != first {
            return true;
        }
        if *arr.uget((y - 1, x + 1)) != first {
            return true;
        }
        // Row y
        if *arr.uget((y, x - 1)) != first {
            return true;
        }
        if *arr.uget((y, x)) != first {
            return true;
        }
        if *arr.uget((y, x + 1)) != first {
            return true;
        }
        // Row y+1
        if *arr.uget((y + 1, x - 1)) != first {
            return true;
        }
        if *arr.uget((y + 1, x)) != first {
            return true;
        }
        if *arr.uget((y + 1, x + 1)) != first {
            return true;
        }
        false
    }
}

/// Check if an edge pixel has any neighbor with a different value.
/// Only visits valid in-bounds neighbors (no reflection/padding).
#[inline]
fn is_boundary_edge<T: PartialEq + Copy>(
    arr: &ArrayView2<T>,
    cy: usize,
    cx: usize,
    h: usize,
    w: usize,
) -> bool {
    let first = arr[[cy, cx]];
    let y_start = cy.saturating_sub(1);
    let y_end = (cy + 1).min(h - 1);
    let x_start = cx.saturating_sub(1);
    let x_end = (cx + 1).min(w - 1);

    for y in y_start..=y_end {
        for x in x_start..=x_end {
            if arr[[y, x]] != first {
                return true;
            }
        }
    }
    false
}

/// Detect boundary pixels where not all neighbors have the same value.
///
/// A pixel is considered a boundary if any value in its 3x3 neighborhood differs
/// from the others (i.e., not all neighbors are equal).
///
/// # Arguments
/// * `arr` - 2D array view of values that support equality comparison
///
/// # Returns
/// * `Array2<bool>` - Boolean array where true indicates a boundary pixel
pub fn find_boundaries<T: PartialEq + Copy>(arr: ArrayView2<T>) -> Array2<bool> {
    let (height, width) = arr.dim();
    let mut result = Array2::<bool>::from_elem((height, width), false);

    // Handle degenerate cases
    if height < 3 || width < 3 {
        // All pixels are edge pixels, use slow path
        for y in 0..height {
            for x in 0..width {
                result[[y, x]] = is_boundary_edge(&arr, y, x, height, width);
            }
        }
        return result;
    }

    // Process top edge (y = 0)
    for x in 0..width {
        result[[0, x]] = is_boundary_edge(&arr, 0, x, height, width);
    }

    // Process bottom edge (y = height - 1)
    for x in 0..width {
        result[[height - 1, x]] = is_boundary_edge(&arr, height - 1, x, height, width);
    }

    // Process left and right edges (y = 1..height-1)
    for y in 1..height - 1 {
        result[[y, 0]] = is_boundary_edge(&arr, y, 0, height, width);
        result[[y, width - 1]] = is_boundary_edge(&arr, y, width - 1, height, width);
    }

    // Process interior (fast path - no bounds checking needed)
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            result[[y, x]] = is_boundary_interior(&arr, y, x);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_uniform_region_no_boundaries() {
        // All same value - no boundaries in interior
        let arr = array![[1u16, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],];
        let result = find_boundaries(arr.view());
        // All should be false - uniform region
        assert!(result.iter().all(|&v| !v));
    }

    #[test]
    fn test_single_different_pixel() {
        // Single different pixel in center
        let arr = array![[0u16, 0, 0], [0, 1, 0], [0, 0, 0],];
        let result = find_boundaries(arr.view());
        // All pixels should be boundaries (all touch the different center pixel)
        assert!(result.iter().all(|&v| v));
    }

    #[test]
    fn test_two_regions() {
        // Two distinct regions
        let arr = array![[1u16, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2],];
        let result = find_boundaries(arr.view());
        // Expected: boundaries at columns 1 and 2 (adjacent to the edge)
        let expected = array![
            [false, true, true, false],
            [false, true, true, false],
            [false, true, true, false],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bool_boundaries() {
        let arr = array![
            [false, false, true, true],
            [false, false, true, true],
            [false, false, true, true],
        ];
        let result = find_boundaries(arr.view());
        let expected = array![
            [false, true, true, false],
            [false, true, true, false],
            [false, true, true, false],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_small_arrays_u16() {
        // 1x1 uniform - no boundary
        let arr = array![[5u16]];
        let result = find_boundaries(arr.view());
        assert!(!result[[0, 0]]);

        // 2x2 uniform - no boundaries
        let arr = array![[1u16, 1], [1, 1]];
        let result = find_boundaries(arr.view());
        assert!(result.iter().all(|&v| !v));

        // 2x2 mixed - all boundaries
        let arr = array![[1u16, 2], [1, 1]];
        let result = find_boundaries(arr.view());
        assert!(result.iter().all(|&v| v));
    }

    // ===================
    // i32 tests
    // ===================

    #[test]
    fn test_i32_uniform_region_no_boundaries() {
        // All same value - no boundaries
        let arr = array![[1i32, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],];
        let result = find_boundaries(arr.view());
        assert!(result.iter().all(|&v| !v));
    }

    #[test]
    fn test_i32_single_different_pixel() {
        // Single different pixel in center - all pixels are boundaries
        let arr = array![[0i32, 0, 0], [0, 1, 0], [0, 0, 0],];
        let result = find_boundaries(arr.view());
        assert!(result.iter().all(|&v| v));
    }

    #[test]
    fn test_i32_two_regions() {
        // Two distinct regions side by side
        let arr = array![[1i32, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2],];
        let result = find_boundaries(arr.view());
        let expected = array![
            [false, true, true, false],
            [false, true, true, false],
            [false, true, true, false],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_i32_negative_values() {
        // Test with negative label values (common in some labeling schemes)
        let arr = array![[-1i32, -1, 0, 0], [-1, -1, 0, 0], [-1, -1, 0, 0],];
        let result = find_boundaries(arr.view());
        let expected = array![
            [false, true, true, false],
            [false, true, true, false],
            [false, true, true, false],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_i32_small_arrays() {
        // 1x1 - no boundary
        let arr = array![[5i32]];
        let result = find_boundaries(arr.view());
        assert!(!result[[0, 0]]);

        // 2x2 uniform - no boundaries
        let arr = array![[1i32, 1], [1, 1]];
        let result = find_boundaries(arr.view());
        assert!(result.iter().all(|&v| !v));

        // 2x2 mixed - all boundaries
        let arr = array![[1i32, 2], [1, 1]];
        let result = find_boundaries(arr.view());
        assert!(result.iter().all(|&v| v));
    }

    #[test]
    fn test_i32_multiple_labels() {
        // Multiple distinct labels (typical segmentation mask)
        let arr = array![
            [1i32, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
        ];
        let result = find_boundaries(arr.view());
        // Boundaries at label transitions: cols 1-2 and 3-4
        let expected = array![
            [false, true, true, true, true, false],
            [false, true, true, true, true, false],
            [false, true, true, true, true, false],
        ];
        assert_eq!(result, expected);
    }

    // ===================
    // Cross-dtype consistency tests
    // ===================

    #[test]
    fn test_dtype_consistency_two_regions() {
        // Same pattern should produce same results across dtypes
        let arr_u8 = array![[1u8, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2],];
        let arr_u16 = array![[1u16, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2],];
        let arr_i32 = array![[1i32, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2],];
        let arr_bool = array![
            [false, false, true, true],
            [false, false, true, true],
            [false, false, true, true],
        ];

        let result_u8 = find_boundaries(arr_u8.view());
        let result_u16 = find_boundaries(arr_u16.view());
        let result_i32 = find_boundaries(arr_i32.view());
        let result_bool = find_boundaries(arr_bool.view());

        assert_eq!(result_u8, result_u16);
        assert_eq!(result_u8, result_i32);
        assert_eq!(result_u8, result_bool);
    }

    #[test]
    fn test_dtype_consistency_uniform() {
        // Uniform arrays should all produce no boundaries
        let arr_u8 = array![[5u8, 5, 5], [5, 5, 5], [5, 5, 5],];
        let arr_u16 = array![[5u16, 5, 5], [5, 5, 5], [5, 5, 5],];
        let arr_i32 = array![[5i32, 5, 5], [5, 5, 5], [5, 5, 5],];
        let arr_bool = array![[true, true, true], [true, true, true], [true, true, true],];

        let result_u8 = find_boundaries(arr_u8.view());
        let result_u16 = find_boundaries(arr_u16.view());
        let result_i32 = find_boundaries(arr_i32.view());
        let result_bool = find_boundaries(arr_bool.view());

        assert!(result_u8.iter().all(|&v| !v));
        assert!(result_u16.iter().all(|&v| !v));
        assert!(result_i32.iter().all(|&v| !v));
        assert!(result_bool.iter().all(|&v| !v));
    }

    #[test]
    fn test_dtype_consistency_center_pixel() {
        // Single different center pixel - all should be boundaries
        let arr_u8 = array![[0u8, 0, 0], [0, 1, 0], [0, 0, 0],];
        let arr_u16 = array![[0u16, 0, 0], [0, 1, 0], [0, 0, 0],];
        let arr_i32 = array![[0i32, 0, 0], [0, 1, 0], [0, 0, 0],];
        let arr_bool = array![
            [false, false, false],
            [false, true, false],
            [false, false, false],
        ];

        let result_u8 = find_boundaries(arr_u8.view());
        let result_u16 = find_boundaries(arr_u16.view());
        let result_i32 = find_boundaries(arr_i32.view());
        let result_bool = find_boundaries(arr_bool.view());

        assert!(result_u8.iter().all(|&v| v));
        assert!(result_u16.iter().all(|&v| v));
        assert!(result_i32.iter().all(|&v| v));
        assert!(result_bool.iter().all(|&v| v));
    }
}
