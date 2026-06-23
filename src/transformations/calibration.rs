use core::f64;

use crate::common::{Dataset, Pair};
use crate::transformations::Transformer;
use crate::utils::polyfit::fit;
use anyhow::{Result, bail};
use clap::Parser;
use ndarray::{ArrayBase, Dim, ViewRepr};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct CalibrationTransform {
    #[clap(short, long, help = "x,y reference data points for calibration.")]
    pub(crate) points: Vec<Pair<f64>>,
    #[clap(short, long, help = "order of polynomial fit")]
    pub(crate) order: usize,
    #[clap(
        short,
        long,
        help = "window in which to calculate centroid to determine peak positions (no centroid if window = 0)"
    )]
    #[serde(default)]
    pub(crate) window: f64,
    coefficients: Vec<f64>,
}

impl Transformer for CalibrationTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        if self.window < 0.0 {
            bail!("Weighting window must be >= 0, got {}", self.window)
        }
        if self.window > 0.0 {
            if dataset.data.columns().into_iter().len() != 2 {
                bail!("Weighting only works for a single spectrum!")
            }
            self.replace_positions_with_centroids(&dataset);
        }
        self.fit()?;
        // Iterate over all x-axes
        for xs in dataset.data.axis_iter_mut(ndarray::Axis(1)).step_by(2) {
            self.eval_inplace(xs);
        }
        Ok(())
    }
}

impl CalibrationTransform {
    pub fn new(points: &[Pair<f64>], order: usize) -> Self {
        Self {
            points: points.to_vec(),
            order,
            window: 0.0,
            coefficients: Vec::new(),
        }
    }

    /// Replace x-values of `positions` by closest centroids
    pub fn replace_positions_with_centroids(&mut self, dataset: &Dataset) {
        for cal_x in self.points.iter_mut().map(|Pair { a, b: _ }| a) {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            let mut ymin = f64::INFINITY;
            // first pass: find ymin
            for (&x, &y) in dataset.data.column(0).iter().zip(dataset.data.column(1)) {
                if (x - *cal_x).abs() <= self.window {
                    if y < ymin {
                        ymin = y;
                    }
                }
            }
            // second pass: calculate centroid
            for (&x, &y) in dataset.data.column(0).iter().zip(dataset.data.column(1)) {
                if (x - *cal_x).abs() <= self.window {
                    numerator += x * (y - ymin);
                    denominator += y - ymin;
                }
            }
            *cal_x = numerator / denominator;
        }
    }

    pub fn fit(&mut self) -> Result<()> {
        // TODO: allocating to intermediate Vecs here is not ideal, could
        // certainly be avoided with different fit API
        let xs: Vec<f64> = self.points.iter().map(|Pair { a, b: _ }| *a).collect();
        let ys: Vec<f64> = self.points.iter().map(|Pair { a: _, b }| *b).collect();

        self.coefficients = fit(&xs, &ys, self.order)?;
        Ok(())
    }

    pub fn eval_inplace(&self, mut xs: ArrayBase<ViewRepr<&mut f64>, Dim<[usize; 1]>>) {
        for x in xs.iter_mut() {
            let mut x_cal = 0.0;
            for (n, c) in self.coefficients.iter().enumerate() {
                x_cal += c * x.powi(n as i32)
            }
            *x = x_cal;
        }
    }
}

impl Default for CalibrationTransform {
    fn default() -> Self {
        Self {
            points: Default::default(),
            order: 1,
            window: 0.0,
            coefficients: Default::default(),
        }
    }
}
