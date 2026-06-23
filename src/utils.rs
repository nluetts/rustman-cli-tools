use anyhow::{Result, anyhow};
use ndarray::{Array1, ArrayBase, Data, Ix1, array};
use std::cmp::Ordering::Greater;

/// Calculate area of single trapezoid.
fn singletrapz(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    0.5 * f64::abs(x1 - x0) * (y1 + y0)
}

/// Linearly interpolate y-value at position x between two points (x0, y0) and (x1, y1).
pub fn lininterp(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    (y1 * (x - x0) + y0 * (x1 - x)) / dx
}

/// Integrate vector `y` in interval [`left`, `right`] using trapezoidal integration.
///
/// If `left` and `right` do not fall on the `x`-grid, additional data points will be interpolated linearly.
/// (i.e. the width of the first and last trapezoid will be somewhat smaller).
/// If `left` and/or `right` falls outside the `x`-range, the integration window will be cropped
/// to the available range.
pub fn trapz<'a, S, T>(
    x: &'a ArrayBase<S, Ix1>,
    y: &'a ArrayBase<T, Ix1>,
    left: f64,
    right: f64,
    local_baseline: bool,
) -> Result<f64>
where
    S: Data<Elem = f64>,
    T: Data<Elem = f64>,
{
    let (mut left, right) = if left < right {
        (left, right)
    } else {
        (right, left)
    };

    let n = x.len() - 1;
    if n != y.len() - 1 {
        return Err(anyhow!("x and y must have the same length!"));
    }
    if n < 1 {
        return Err(anyhow!("x and y must contain more than 2 elements!"));
    }
    if x[0] >= right || x[n] <= left {
        return Err(anyhow!("Integration window out of bounds."));
    }

    let mut area: f64;
    // subtract local linear baseline, defined by start and end-point of integration window
    if local_baseline {
        let xs = array![left, right];
        let ys = linear_resample_array(&x, &y, &xs);
        if ys.iter().any(|x| (*x).is_nan()) {
            return Err(anyhow!("Integration window out of bounds."));
        }
        area = -singletrapz(left, right, ys[0], ys[1])
    } else {
        area = 0.0_f64;
    }

    let mut inside_integration_window = false;
    let mut lastiter = false;
    let mut j = 2;

    while j <= n {
        let mut x0 = x[j - 1];
        let mut x1 = x[j];
        let mut y0 = y[j - 1];
        let mut y1 = y[j];

        if x1 <= left {
            j += 1;
            continue;
        } else if !inside_integration_window {
            // this will only run once, when we enter the integration window
            // test whether x0 should be replaced by left
            if x0 < left {
                y0 = lininterp(left, x0, x1, y0, y1);
                x0 = left;
            } else {
                // this case means that left <= x[0]
                left = x0;
            }
            inside_integration_window = true;
        }

        // test whether x1 should be replaced by right
        if x1 >= right {
            // we move out of the integration window

            if x1 != right {
                y1 = lininterp(right, x0, x1, y0, y1)
            };
            x1 = right;
            lastiter = true; // we shall break the loop after this iteration
        }

        area += singletrapz(x0, x1, y0, y1);

        if lastiter {
            break;
        }

        j += 1;
    }
    Ok(area)
}

/// Linearly interpolate x, y datapoints on grid where grid and xs overlap.
///
/// Returns NAN in range where xs and grid do not overlap
pub fn linear_resample_array<S, T, V>(
    xs: &ArrayBase<S, Ix1>,
    ys: &ArrayBase<T, Ix1>,
    grid: &ArrayBase<V, Ix1>,
) -> Array1<f64>
where
    S: Data<Elem = f64>,
    T: Data<Elem = f64>,
    V: Data<Elem = f64>,
{
    let segments = xs
        .iter()
        .zip(ys.iter())
        .zip(xs.iter().skip(1).zip(ys.iter().skip(1)))
        .map(|((x0, y0), (x1, y1))| (*x0, *y0, *x1, *y1))
        .collect::<Vec<_>>();

    let mut yp = Vec::with_capacity(grid.len());

    for xi in grid.iter() {
        if let Some((x0, y0, x1, y1)) = segments.iter().find(|(x0, _, x1, _)| xi >= x0 && xi < x1) {
            yp.push(lininterp(*xi, *x0, *x1, *y0, *y1));
            continue;
        }
        // only applies if xi happens to be == the last value in xs
        else if let Some((_, _, _, y1)) = segments.iter().last().filter(|(_, _, x1, _)| xi == x1)
        {
            yp.push(*y1);
            continue;
        }
        // applies if xi does not lie within the range of xs
        else {
            yp.push(f64::NAN)
        };
    }
    Array1::from_vec(yp)
}

/// get the index of element in `x` which is closest to `xi`
pub fn nearest_index<'a, T>(x: &'a ArrayBase<T, Ix1>, xi: f64) -> Option<usize>
where
    T: Data<Elem = f64>,
{
    if let Some((idx, _)) = x
        .iter()
        .map(|x| (x - xi).abs())
        .enumerate()
        // NaN values will always be considerer Greater than valid floats,
        // so index will be found next to valid float, if not all values
        // are NaN
        .min_by(|(_, xi), (_, xj)| xi.partial_cmp(xj).unwrap_or(Greater))
    {
        Some(idx)
    } else {
        None
    }
}

#[cfg(test)]
//  (f = x->  exp(3x), F = x->        1/3*exp(3x)),
//  (f = x->  1.2^(x), F = x->   1.2^(x)/log(1.2)),
//  (f = x->   sin(x), F = x->            -cos(x)),
//  (f = x-> 1/(2x+3), F = x-> 1/2*log(abs(2x+3)))
mod tests {
    use super::{linear_resample_array, trapz};
    use ndarray::{self, Array1};

    #[test]
    fn test_parse_header() {
        let x: ndarray::Array1<f64> = ndarray::ArrayBase::range(0.0, 10.0, 0.001);
        let y: Array1<f64> = x.map(|xi| f64::exp(3.0 * xi));
        let area: f64 = trapz(&x, &y, 3.15, 8.55, false).unwrap();
        let area_analytic = 1.0 / 3.0 * (f64::exp(3.0 * 8.55) - f64::exp(3.0 * 3.15));
        assert_eq!(area, area_analytic);
    }
    #[test]
    fn test_linear_resample() {
        let xs = ndarray::array![1., 2., 3., 4., 5.];
        let ys = ndarray::array![1., 2., 3., 4., 5.];
        let grid = ndarray::array![1.5, 2.5, 2.0, 5.0]; // TODO: 5.0 should also be interpolated
        let _res = linear_resample_array(&xs, &ys, &grid);
    }
}

// Polynomial least-squares fitting implementation developed with assistance
// from OpenAI GPT-5.5
pub mod polyfit {
    use anyhow::{Result, anyhow, bail};
    use ndarray_linalg::LeastSquaresSvd;

    pub fn fit(xs: &[f64], ys: &[f64], degree: usize) -> Result<Vec<f64>> {
        let m = xs.len();

        if m != ys.len() {
            bail!("x and y must be of same length, got {}, {}", m, ys.len());
        }

        let cols = degree
            .checked_add(1)
            .ok_or_else(|| anyhow!("polynomial degree is too large"))?;

        if m < cols {
            bail!(
                "not enough points to fit polynomial of degree {}: need at least {}, got {}",
                degree,
                cols,
                m
            );
        }

        if xs.iter().chain(ys.iter()).any(|v| !v.is_finite()) {
            bail!("x and y values must all be finite");
        }

        let mean = xs.iter().sum::<f64>() / m as f64;

        // Scale by maximum absolute deviation from the mean.
        // This maps the x-values roughly into [-1, 1].
        let scale = xs.iter().map(|&x| (x - mean).abs()).fold(0.0_f64, f64::max);

        if scale == 0.0 && degree > 0 {
            bail!(
                "cannot fit polynomial of degree {} when all x values are equal",
                degree
            );
        }

        // For degree 0 with constant x, any nonzero scale works.
        let scale = if scale == 0.0 { 1.0 } else { scale };

        // Build Vandermonde matrix using normalized x:
        //
        // z = (x - mean) / scale
        //
        // [1, z, z^2, ..., z^degree]
        let mut vandermonde = ndarray::Array2::<f64>::zeros((m, cols));

        for (i, &x) in xs.iter().enumerate() {
            let z = (x - mean) / scale;
            let mut z_pow = 1.0;

            for j in 0..cols {
                vandermonde[(i, j)] = z_pow;
                z_pow *= z;
            }
        }

        let y = ndarray::Array1::from_vec(ys.to_vec());

        // Coefficients for normalized coordinate z:
        //
        // y = a[0] + a[1] z + a[2] z^2 + ...
        let normalized_coefficients = vandermonde.least_squares(&y)?.solution;

        // Convert coefficients from powers of z back to powers of x.
        //
        // z = (x - mean) / scale
        //
        // a[j] z^j = a[j] ((x - mean) / scale)^j
        //
        // Expanding this gives coefficients for:
        //
        // y = c[0] + c[1] x + c[2] x^2 + ...
        let coefficients =
            denormalize_coefficients(normalized_coefficients.as_slice().unwrap(), mean, scale);

        Ok(coefficients)
    }

    fn denormalize_coefficients(normalized: &[f64], mean: f64, scale: f64) -> Vec<f64> {
        let degree = normalized.len() - 1;
        let mut coefficients = vec![0.0; normalized.len()];

        let mut neg_mean_powers = vec![1.0; degree + 1];
        for i in 1..=degree {
            neg_mean_powers[i] = neg_mean_powers[i - 1] * -mean;
        }

        let inv_scale = 1.0 / scale;
        let mut inv_scale_powers = vec![1.0; degree + 1];
        for i in 1..=degree {
            inv_scale_powers[i] = inv_scale_powers[i - 1] * inv_scale;
        }

        for j in 0..=degree {
            let factor = normalized[j] * inv_scale_powers[j];

            for k in 0..=j {
                coefficients[k] += factor * binomial(j, k) * neg_mean_powers[j - k];
            }
        }

        coefficients
    }

    fn binomial(n: usize, k: usize) -> f64 {
        let k = k.min(n - k);

        let mut result = 1.0;

        for i in 1..=k {
            result *= (n + 1 - i) as f64;
            result /= i as f64;
        }

        result
    }
    #[cfg(test)]
    mod tests {
        use super::fit;

        fn assert_close(actual: f64, expected: f64, tol: f64) {
            assert!(
                (actual - expected).abs() <= tol,
                "expected {expected}, got {actual}, diff = {}",
                (actual - expected).abs()
            );
        }

        fn assert_coeffs_close(actual: &[f64], expected: &[f64], tol: f64) {
            assert_eq!(
                actual.len(),
                expected.len(),
                "coefficient length mismatch: expected {}, got {}",
                expected.len(),
                actual.len()
            );

            for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() <= tol,
                    "coefficient {i}: expected {e}, got {a}, diff = {}",
                    (a - e).abs()
                );
            }
        }

        fn assert_error_contains<T>(result: anyhow::Result<T>, expected: &str)
        where
            T: std::fmt::Debug,
        {
            let err = result.expect_err("expected error, got Ok");
            let msg = err.to_string();

            assert!(
                msg.contains(expected),
                "expected error to contain {expected:?}, got {msg:?}"
            );
        }

        #[test]
        fn fits_exact_line() -> anyhow::Result<()> {
            // y = 2 + 3x
            let xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
            let ys: Vec<f64> = xs.iter().map(|&x| 2.0 + 3.0 * x).collect();

            let coeffs = fit(&xs, &ys, 1)?;

            // Coefficients are returned as:
            // y = c[0] + c[1] x + ...
            assert_coeffs_close(&coeffs, &[2.0, 3.0], 1e-10);

            Ok(())
        }

        #[test]
        fn fits_exact_quadratic_with_nonzero_mean_x() -> anyhow::Result<()> {
            // y = 5 - 3x + 0.25x^2
            //
            // Using x values away from zero helps exercise the normalization and
            // denormalization logic.
            let xs = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
            let ys: Vec<f64> = xs.iter().map(|&x| 5.0 - 3.0 * x + 0.25 * x * x).collect();

            let coeffs = fit(&xs, &ys, 2)?;

            assert_coeffs_close(&coeffs, &[5.0, -3.0, 0.25], 1e-8);

            Ok(())
        }

        #[test]
        fn degree_zero_fit_returns_mean_y() -> anyhow::Result<()> {
            let xs = [1.0, 2.0, 3.0, 4.0];
            let ys = [2.0, 4.0, 6.0, 8.0];

            let coeffs = fit(&xs, &ys, 0)?;

            assert_eq!(coeffs.len(), 1);
            assert_close(coeffs[0], 5.0, 1e-12);

            Ok(())
        }
        #[test]
        fn errors_when_lengths_differ() {
            let xs = [1.0, 2.0, 3.0];
            let ys = [1.0, 2.0];

            assert_error_contains(fit(&xs, &ys, 1), "x and y must be of same length");
        }

        #[test]
        fn errors_when_not_enough_points() {
            let xs = [1.0, 2.0];
            let ys = [3.0, 5.0];

            assert_error_contains(
                fit(&xs, &ys, 2),
                "not enough points to fit polynomial of degree 2",
            );
        }

        #[test]
        fn errors_when_x_values_are_all_equal_for_nonconstant_fit() {
            let xs = [1.0, 1.0, 1.0];
            let ys = [2.0, 3.0, 4.0];

            assert_error_contains(
                fit(&xs, &ys, 1),
                "cannot fit polynomial of degree 1 when all x values are equal",
            );
        }

        #[test]
        fn errors_when_input_contains_nan() {
            let xs = [1.0, 2.0, f64::NAN];
            let ys = [2.0, 4.0, 6.0];

            assert_error_contains(fit(&xs, &ys, 1), "x and y values must all be finite");
        }
    }
}
