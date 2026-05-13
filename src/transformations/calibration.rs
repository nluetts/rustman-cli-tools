use crate::common::{Dataset, Pair};
use crate::transformations::Transformer;
use anyhow::{Result, anyhow};
use clap::Parser;
use ndarray::{ArrayBase, Dim, ViewRepr};
use ndarray_linalg::LeastSquaresSvd;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct CalibrationTransform {
    #[clap(short, long, help = "x,y reference data points for calibration.")]
    pub(crate) points: Vec<Pair<f64>>,
    #[clap(short, long, help = "order of polynomial fit")]
    pub(crate) order: usize,
    coefficients: Vec<f64>,
}

impl Transformer for CalibrationTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        self.fit()?;
        // Iterate over all x-axes
        for xs in dataset.data.axis_iter_mut(ndarray::Axis(1)).step_by(2) {
            self.eval_inplace(xs);
        }
        Ok(())
    }
}

impl CalibrationTransform {
    fn fit(&mut self) -> Result<()> {
        let m = self.points.len();
        if m == 0 || m - 1 < self.order {
            return Err(anyhow!(
                "Not enough anchor points to perform fit of order {} ({})",
                self.order,
                self.points.len()
            ));
        }

        // Perform polyfit via Vandermonde Matrix
        let mut vandermonde = ndarray::Array2::ones((self.points.len(), self.order + 1));
        let mut ys = ndarray::Array1::ones(self.points.len());
        for (i, (x, y)) in self.points.iter().map(|p| (p.a, p.b)).enumerate() {
            ys[i] = y;
            for k in 1..=self.order {
                vandermonde[[i, k]] = x.powi(k as i32)
            }
        }

        self.coefficients.clear();
        for c in vandermonde.least_squares(&ys)?.solution {
            self.coefficients.push(c);
        }
        Ok(())
    }

    fn eval_inplace(&self, mut xs: ArrayBase<ViewRepr<&mut f64>, Dim<[usize; 1]>>) {
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
            coefficients: Default::default(),
        }
    }
}
