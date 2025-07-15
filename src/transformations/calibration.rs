use crate::common::{Dataset, Pair};
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct CalibrationTransform {
    #[clap(short, long, help = "x,y reference data points for calibration.")]
    pub(crate) points: Vec<Pair<f64>>,
}

impl Transformer for CalibrationTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        if let Some((slope, intercept)) = linregress(&self.points) {
            // Iterate over all x-axes
            for xs in dataset.data.axis_iter_mut(ndarray::Axis(1)).step_by(2) {
                for x in xs {
                    *x = *x * slope + intercept
                }
            }
        }
        Ok(())
    }
}

fn linregress(pts: &[Pair<f64>]) -> Option<(f64, f64)> {
    // Zero reference points cannot be processed.
    if pts.len() == 0 {
        return None;
    }

    // One point can be used to figure out an offset.
    if pts.len() == 1 {
        let Pair { a, b } = pts.iter().nth(0).unwrap();
        return Some((1.0, b - a));
    }

    // More than one points can be used to estimate a slope and intercept.
    let mean_x = pts.iter().map(|Pair { a, b: _ }| a).sum::<f64>() / (pts.len() as f64);
    let mean_y = pts.iter().map(|Pair { a: _, b }| b).sum::<f64>() / (pts.len() as f64);

    let slope = {
        let numerator: f64 = pts
            .iter()
            .map(|Pair { a, b }| (a - mean_x) * (b - mean_y))
            .sum();
        let denominator: f64 = pts
            .iter()
            .map(|Pair { a, b: _ }| (a - mean_x).powi(2))
            .sum();
        numerator / denominator
    };

    let intercept = mean_y - slope * mean_x;

    Some((slope, intercept))
}
