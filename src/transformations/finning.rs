use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use ndarray::{s, Array1, Axis};
use ndarray_stats::interpolate::Nearest;
use ndarray_stats::QuantileExt;
use noisy_float::prelude::n64;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct FinningTransform {
    #[clap(help = "Multiple of standard deviation which flags point as spike.")]
    pub(crate) threshold: f64,
    #[clap(
        short,
        long,
        default_value("100"),
        help = "Maximum number of iterations the finning algorithm runs."
    )]
    pub(crate) iterations: usize,
}

impl Transformer for FinningTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        // since we do not want to change order of columns in input dataset,
        // we have to copy the data into a buffer to calculate the median
        // for each pixel
        let number_scans = dataset.data.ncols() / 2;
        if number_scans < 3 {
            let message = format!(
                "Not enough scans to perform finning, got {}, need at least 3.",
                number_scans
            );
            return Err(anyhow::Error::msg(message));
        }
        let mut intensities_buffer = Array1::<f64>::zeros(number_scans);
        for mut row in dataset.data.slice_mut(s![.., 1..;2]).axis_iter_mut(Axis(0)) {
            intensities_buffer.assign(&row);
            let mut intensities_median =
                match intensities_buffer.quantile_axis_skipnan_mut(Axis(0), n64(0.5), &Nearest) {
                    Ok(ms) => ms.into_scalar(),
                    Err(err) => return Err(anyhow::Error::from(err)),
                };
            let mut intensities_std = intensities_buffer.std(1.0);
            let mut n = match row.argmax() {
                Ok(index) => index,
                Err(err) => return Err(anyhow::Error::from(err)),
            };
            let mut iterations: usize = 0;
            while row[n] > intensities_median + self.threshold * intensities_std {
                iterations += 1;
                row[n] = intensities_median;
                intensities_buffer.assign(&row);
                intensities_median =
                    match intensities_buffer.quantile_axis_skipnan_mut(Axis(0), n64(0.5), &Nearest)
                    {
                        Ok(ms) => ms.into_scalar(),
                        Err(err) => return Err(anyhow::Error::from(err)),
                    };
                intensities_std = intensities_buffer.std(1.0);
                n = match row.argmax() {
                    Ok(index) => index,
                    Err(err) => return Err(anyhow::Error::from(err)),
                };
                if iterations > self.iterations {
                    break;
                }
            }
        }
        Ok(())
    }
}
