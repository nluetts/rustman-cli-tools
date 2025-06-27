use crate::common::{Dataset, Pair};
use crate::transformations::Transformer;
use crate::utils::trapz;
use anyhow::Result;
use clap::Parser;
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct IntegrateTransform {
    #[clap(help = "Left and right integration bound, separated by comma.")]
    pub(crate) bounds: Vec<Pair<f64>>,
    #[clap(
        short,
        long,
        action,
        help = "Subtract local baseline (straight line from integration start- to end-point)."
    )]
    pub(crate) local_baseline: bool,
}

impl Transformer for IntegrateTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let mut integrals: Array2<f64> =
            Array2::zeros((dataset.data.ncols() / 2, self.bounds.len() * 2));
        for (i, (xs, ys)) in dataset
            .data
            .axis_iter(Axis(1))
            .step_by(2)
            .zip(dataset.data.axis_iter(Axis(1)).skip(1).step_by(2))
            .enumerate()
        {
            for (j, bd) in self.bounds.iter().enumerate() {
                integrals[[i, j * 2]] = (i + 1) as f64;
                integrals[[i, j * 2 + 1]] = trapz(&xs, &ys, bd.a, bd.b, self.local_baseline)?;
            }
        }
        dataset.data = integrals;
        Ok(())
    }
}
