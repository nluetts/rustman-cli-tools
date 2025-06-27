use anyhow::{anyhow, Result};
use ndarray::{array, Array1, ArrayBase, Data, Ix1};
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
        let res = linear_resample_array(&xs, &ys, &grid);
    }
}
