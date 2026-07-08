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
        help = "number of datapoints to use for centroid calculation (no centroid if window = 0)"
    )]
    #[serde(default)]
    pub(crate) window: usize,
    coefficients: Vec<f64>,
}

impl Transformer for CalibrationTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        // Window is now usize, so it can't be negative
        // But we should check for reasonable values
        if self.window > 0 {
            if dataset.data.columns().into_iter().len() != 2 {
                bail!("Weighting only works for a single spectrum!")
            }
            if self.window > dataset.data.column(0).len() {
                bail!(
                    "Window size {} is larger than dataset size {}",
                    self.window,
                    dataset.data.column(0).len()
                )
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
            window: 0,
            coefficients: Vec::new(),
        }
    }

    /// Replace x-values of `positions` by closest centroids
    pub fn replace_positions_with_centroids(&mut self, dataset: &Dataset) -> Result<()> {
        for cal_x in self.points.iter_mut().map(|Pair { a, b: _ }| a) {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            let mut ymin = f64::INFINITY;

            // Find the index of the datapoint closest to cal_x
            let mut center_idx = None;
            let mut dxmin = f64::INFINITY;
            for (i, x) in dataset.data.column(0).iter().enumerate() {
                let dx = (*cal_x - *x).abs();
                if dx < dxmin {
                    dxmin = dx;
                    center_idx = Some(i);
                }
            }

            let Some(center_idx) = center_idx else {
                bail!("Unable to locate {cal_x} in dataset!")
            };

            // Calculate the range of datapoints to consider
            let start_idx = center_idx.saturating_sub(self.window);
            let end_idx = std::cmp::min(center_idx + self.window, dataset.data.column(0).len() - 1);

            // first pass: find ymin in the window
            for i in start_idx..=end_idx {
                let y = dataset.data.column(1)[i];
                if y < ymin {
                    ymin = y;
                }
            }

            // second pass: calculate centroid
            for i in start_idx..=end_idx {
                let x = dataset.data.column(0)[i];
                let y = dataset.data.column(1)[i];
                numerator += x * (y - ymin);
                denominator += y - ymin;
            }

            let new_cal_x = numerator / denominator;

            *cal_x = new_cal_x;
        }
        Ok(())
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
            window: 0,
            coefficients: Default::default(),
        }
    }
}
