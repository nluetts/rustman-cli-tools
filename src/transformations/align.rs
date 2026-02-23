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
        default_value_t = 0.1,
        help = "Maximum absolut value of cost function, adapt only if alignment fails."
    )]
    pub cost_max_abs: f64,
}

impl Transformer for AlignTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let ref_spectrum: ndarray::ArrayView2<_> = dataset.data.slice(s![.., 0..=1]);
        // Find maximum in reference frame
        let ref_grid_max = ref_spectrum
            .rows()
            .into_iter()
            .fold((f64::NAN, f64::NAN), |(xmax, ymax), row| {
                let x = row[0];
                let y = row[1];

                if ymax.is_nan() || y > ymax {
                    return (x, y);
                } else {
                    return (xmax, ymax);
                }
            })
            .0;
        // Crop reference spectrum
        let ref_spectrum = crop(ref_spectrum, ref_grid_max);

        dbg!(&ref_spectrum);

        for i in (2..dataset.data.ncols()).step_by(2) {
            let spectrum = dataset.data.slice(s![.., i..=i + 1]);

            let problem = OptAlignment {
                frame_a: ref_spectrum.view(),
                frame_b: spectrum.view(),
                debug: false,
            };
            let solver = NelderMead::new(vec![
                -f64::abs(self.cost_max_abs),
                f64::abs(self.cost_max_abs),
            ]);
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

            // let problem = OptAlignment {
            //     frame_a: ref_spectrum.view(),
            //     frame_b: spectrum.view(),
            //     debug: true,
            // };
            // let _ = problem.cost(&dx);

            let shifted_grid = &spectrum.slice(s![.., 0]) + dbg!(dx);
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

fn crop(ref_spectrum: ArrayView2<f64>, ref_grid_max: f64) -> Array2<f64> {
    let idx: ndarray::Array1<_> = ref_spectrum
        .axis_iter(Axis(0))
        .enumerate()
        .filter_map(|(i, row)| {
            // TODO remove these hard coded bounds (0.5)
            if row[0] > ref_grid_max - 0.5 && row[0] < ref_grid_max + 0.5 {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    ref_spectrum
        .slice(s![idx[0]..idx[idx.len() - 1], ..])
        .into_owned()
}

struct OptAlignment<'a> {
    frame_a: ArrayView2<'a, f64>,
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
            .frame_a
            .axis_iter(Axis(0))
            .zip(self.frame_a.axis_iter(Axis(0)).skip(1))
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
