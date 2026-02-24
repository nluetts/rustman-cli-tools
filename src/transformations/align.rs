use crate::common::Dataset;
use crate::transformations::Transformer;
use crate::utils::linear_resample_array;
use anyhow::{Result, anyhow};
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;
use clap::Parser;
use ndarray::{Array2, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct AlignTransform {
    #[clap(
        short,
        long,
        default_value_t = 0.01,
        help = "Maximum absolut value of cost function, adapt only if alignment fails."
    )]
    pub tuning: f64,
    #[clap(
        short,
        long,
        default_value_t = 10,
        help = "Half-width of window around maximum of reference frame."
    )]
    pub window_size: usize,
}

impl Transformer for AlignTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let ref_spectrum: ndarray::ArrayView2<_> = dataset.data.slice(s![.., 0..=1]);
        // Find maximum in reference frame
        let Some(idx_ref_max) = ref_spectrum
            .rows()
            .into_iter()
            .enumerate()
            .fold(None, |acc, (i, row)| {
                let Some((idx, ymax)) = acc else {
                    return Some((i, row[1]));
                };
                let y = row[1];
                if y > ymax {
                    return Some((i, y));
                } else {
                    return Some((idx, ymax));
                }
            })
            .map(|(idx, _y)| idx)
        else {
            return Err(anyhow!("Unable to identify maximum in reference spectrum."));
        };

        let ref_grid = ref_spectrum.slice(s![.., 0]).into_owned();
        // Crop reference spectrum
        let ref_spectrum = crop(ref_spectrum, idx_ref_max);

        for i in (2..dataset.data.ncols()).step_by(2) {
            // Set all x-values to the reference grid—if we don't do this, a
            // later average transform will not work properly
            for j in 0..ref_grid.len() {
                dataset.data[[j, i]] = ref_grid[j];
            }
            let spectrum = dataset.data.slice(s![.., i..=i + 1]);

            let problem = OptAlignment {
                frame_ref: ref_spectrum.view(),
                frame_b: spectrum.view(),
                debug: false,
            };
            let solver = NelderMead::new(vec![-f64::abs(self.tuning), f64::abs(self.tuning)]);
            let res = Executor::new(problem, solver)
                .configure(|state| state.param(0.0))
                .run()?;
            let dx = match res.state().best_param {
                None => {
                    return Err(anyhow!(
                        "frame alignment failed, optimization did not return optimized parameters"
                    ));
                }
                Some(param) => param,
            };

            let shifted_grid = &spectrum.slice(s![.., 0]) + dx;
            let aligned_frame = linear_resample_array(
                &shifted_grid,
                &spectrum.slice(s![.., 1]),
                &spectrum.slice(s![.., 0]),
            );
            let mut frame = dataset.data.column_mut(i + 1);
            for (fr, afr) in frame.iter_mut().zip(aligned_frame.iter()) {
                *fr = *afr
            }
        }

        // Filter out NaN
        let mut valid_data = Array2::default((0, dataset.data.shape()[1]));
        for row in dataset.data.rows() {
            if !row.iter().any(|x| !x.is_finite()) {
                valid_data.push_row(row)?;
            }
        }
        dataset.data = valid_data;
        Ok(())
    }

    fn write_metadata_yaml(&self, dataset: &mut Dataset) -> Result<()> {
        let metadata = self.config_to_string()?;
        dataset.metadata += &metadata;
        dataset.metadata += "---\n";
        Ok(())
    }

    fn apply(&mut self, dataset: &mut Dataset) -> Result<()> {
        self.transform(dataset)?;
        self.write_metadata_yaml(dataset)?;
        Ok(())
    }
}

fn crop(spectrum: ArrayView2<f64>, idx: usize) -> Array2<f64> {
    spectrum
        .axis_iter(Axis(0))
        .enumerate()
        .filter_map(|(i, row)| {
            // TODO remove these hard coded bounds
            if i >= idx - 10 && i <= idx + 10 {
                Some([row[0], row[1]])
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .into()
}

struct OptAlignment<'a> {
    frame_ref: ArrayView2<'a, f64>,
    frame_b: ArrayView2<'a, f64>,
    debug: bool,
}

impl<'a> CostFunction for OptAlignment<'a> {
    type Param = f64; // x shift
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let mut iter_b = self.frame_b.axis_iter(Axis(0)).peekable();
        let mut sum = 0.0;
        for (rowi, rowj) in self
            .frame_ref
            .axis_iter(Axis(0))
            .zip(self.frame_ref.axis_iter(Axis(0)).skip(1))
        {
            let (xi, yi, xj, yj) = (rowi[0], rowi[1], rowj[0], rowj[1]);
            if self.debug {
                dbg!((xi, yi, xj, yj));
            }
            while let Some(rowb) = iter_b.peek() {
                let xb = rowb[0] + param;
                let yb = rowb[1];
                if xb < xi {
                    iter_b.next();
                    continue;
                }
                if xb > xj {
                    if self.debug {
                        eprintln!("---- break: {xb}, {yb}");
                    }
                    break;
                }
                if self.debug {
                    eprintln!("---- interpolate: {xb}, {yb}")
                }
                // xi <= xb <= xj → we interpolate
                let yinterp = yi * ((1.0 - (xb - xi)) / (xj - xi)).abs()
                    + yj * ((1.0 - (xj - xb)) / (xj - xi)).abs();
                sum += (yinterp - yb).powi(2);
                iter_b.next();
            }
        }
        Ok(sum.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argmin::core::CostFunction;
    use ndarray::array;

    #[test]
    fn test_cost_function() {
        // --- Setup Data ---

        // Frame A (Reference): A simple line from x=0 to x=4
        // y = x
        let frame_ref = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],];

        // A fixed shift along x we use for this test.
        let shift = 0.5;

        // Frame B (Target): The same line, but shifted w.r.t the x axis.
        let frame_b = array![
            [0.0, 0.0 + shift],
            [1.0, 1.0 + shift],
            [2.0, 2.0 + shift],
            [3.0, 3.0 + shift],
            [4.0, 4.0 + shift],
            [5.0, 5.0 + shift],
        ];

        let problem = OptAlignment {
            frame_ref: frame_ref.view(),
            frame_b: frame_b.view(),
            debug: false,
        };

        // --- Test 1: Optimal Alignment ---
        // We expect the cost to be 0.0 when `param` == `shift`
        let cost_optimal = problem.cost(&shift).unwrap();

        println!("Cost at optimal param ({}): {}", shift, cost_optimal);
        assert!(
            cost_optimal == 0.0,
            "Cost should be zero at optimal alignment, is {}",
            cost_optimal
        );

        // --- Test 2: Misalignment ---
        // If we use param = 0.0, the lines are offset. Cost should be higher.
        let bad_param = 0.0;
        let cost_bad = problem.cost(&bad_param).unwrap();

        println!("Cost at bad param (0.0): {}", cost_bad);
        assert!(
            cost_bad > cost_optimal,
            "Cost should be higher when misaligned (optimal cost = {}, misaligned cost = {})",
            cost_optimal,
            cost_bad
        );

        // --- Test 3: Minimization ---
        let solver = NelderMead::new(vec![-1.5, 0.0]);
        let executor = Executor::new(problem, solver);
        let result = executor.run().unwrap();

        println!("Result: {:?}", result.state);

        let best_param = result.state.best_param.unwrap();
        assert!(
            (best_param - shift).abs() < 1e-9,
            "Expected param approx {}, got {}",
            shift,
            best_param
        );

        // The final cost should be very low
        assert!(result.state.best_cost < 1e-9, "Cost should be near zero");
    }
}
