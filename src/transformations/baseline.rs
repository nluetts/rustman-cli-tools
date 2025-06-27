use crate::common::{Dataset, Pair};
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use splines::Key;

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct BaselineTransform {
    #[clap(short, long, help = "x,y points to draw spline baseline.")]
    pub(crate) points: Vec<Pair<f64>>,
    #[clap(
        short,
        long,
        action,
        help = "If flag is set, add baseline to dataset instead of subtracting it."
    )]
    pub(crate) store: bool,
}

impl Transformer for BaselineTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        if self.points.len() < 2 {
            return Ok(());
        }
        let spline = {
            let mut keys = vec![];
            let n_pts = self.points.len();
            for i in 0..n_pts {
                if i == 0 || i == n_pts - 2 {
                    keys.push(Key::new(
                        self.points[i].a,
                        self.points[i].b,
                        splines::Interpolation::Linear,
                    ));
                } else {
                    keys.push(Key::new(
                        self.points[i].a,
                        self.points[i].b,
                        splines::Interpolation::CatmullRom,
                    ));
                }
            }
            splines::Spline::from_vec(keys)
        };
        if self.store {
            // store baseline as a new frame
            let x_p: Array1<f64> = dataset.data.column(0).to_owned();
            let y_p: Array1<f64> = x_p
                .iter()
                .map(|x| spline.sample(*x).unwrap_or(0.0))
                .collect();
            let baseline: Array2<f64> = ndarray::stack![Axis(1), x_p, y_p];
            dataset.data = ndarray::concatenate(Axis(1), &[dataset.data.view(), baseline.view()])?;
        } else {
            // subtract baseline
            for j in (0..dataset.data.ncols()).step_by(2) {
                for i in 0..dataset.data.nrows() {
                    dataset.data[[i, j + 1]] -= spline.sample(dataset.data[[i, j]]).unwrap_or(0.0);
                }
            }
        }
        Ok(())
    }
}
