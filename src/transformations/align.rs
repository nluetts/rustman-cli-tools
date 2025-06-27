use crate::common::Dataset;
use crate::transformations::Transformer;
use crate::utils::linear_resample_array;
use anyhow::{anyhow, Result};
use argmin::core::{CostFunction, Executor};
use argmin::solver::brent::BrentOpt;
use clap::Parser;
use ndarray::{s, Array1, ArrayBase, Data, Ix1};
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
        let nrows = dataset.data.nrows();
        let ref_grid = dataset.data.slice(s![.., 0]).into_owned();
        let ref_frame = dataset.data.slice(s![.., 1]).into_owned();
        for i in (2..dataset.data.ncols()).step_by(2) {
            // set all x-axes to values from reference frame (frame 1)
            for j in 0..nrows {
                dataset.data[[j, i]] = ref_grid[j];
            }
            let mut frame = dataset.data.column_mut(i + 1);
            let init_param = 0.0;
            let problem = OptAlignment::new(&ref_frame, &frame)?;
            let solver = BrentOpt::new(-f64::abs(self.cost_max_abs), f64::abs(self.cost_max_abs));
            let res = Executor::new(problem, solver)
                .configure(|state| state.param(init_param))
                .run()?;
            let dx = match res.state().best_param {
                None => {
                    return Err(anyhow!(
                        "frame alignment failed, optimization did not return optimized parameters"
                    ))
                }
                Some(param) => param,
            };
            let shifted_grid = &ref_grid + dx;
            let aligned_frame = linear_resample_array(&shifted_grid, &frame, &ref_grid);
            for j in 0..nrows {
                frame[j] = aligned_frame[j]
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

struct OptAlignment<'a, S, T>
where
    S: Data<Elem = f64>,
    T: Data<Elem = f64>,
{
    frame_a: &'a ArrayBase<S, Ix1>,
    frame_b: &'a ArrayBase<T, Ix1>,
}

impl<'a, S, T> OptAlignment<'a, S, T>
where
    S: Data<Elem = f64>,
    T: Data<Elem = f64>,
{
    fn new(frame_a: &'a ArrayBase<S, Ix1>, frame_b: &'a ArrayBase<T, Ix1>) -> Result<Self> {
        if frame_a.len() == frame_b.len() {
            Ok(Self { frame_a, frame_b })
        } else {
            Err(anyhow!(
                "frames that shall be aligned must be of same length"
            ))
        }
    }
}

impl<'a, S, T> CostFunction for OptAlignment<'a, S, T>
where
    S: Data<Elem = f64>,
    T: Data<Elem = f64>,
{
    type Param = f64; // x shift
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let grid: Array1<f64> = (1..self.frame_a.len()).map(|x| x as f64).collect();
        let x_shifted = &grid + *param;
        let ys = linear_resample_array(&x_shifted, self.frame_b, &grid);
        let mut sum = 0.0;
        for (y1, y0) in ys.iter().zip(self.frame_a) {
            // this seems to work rather well, the cost function in the python implementation
            // (square of difference) does not work here
            let cst = -(y1 * y0).abs();
            if !cst.is_nan() {
                sum += cst;
            }
        }
        Ok(sum)
    }
}

// impl<'a, S, T> Gradient for OptAlignment<'a, S, T>
// where
//     S: Data<Elem = f64>,
//     T: Data<Elem = f64>,
// {
//     type Param = f64;
//     type Gradient = Vec<f64>;

//     fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient> {
//         Ok(vec![*param].forward_diff(&|p| self.cost(&p[0]).unwrap()))
//     }
// }

// impl<'a, S, T> Hessian for OptAlignment<'a, S, T>
// where
//     S: Data<Elem = f64>,
//     T: Data<Elem = f64>,
// {
//     type Param = Array1<f64>; // x and y shift
//     type Hessian = Array2<f64>;

//     fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian> {
//         Ok(param.forward_hessian(&|p| self.gradient(p).unwrap()))
//     }
// }
