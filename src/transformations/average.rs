use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use ndarray::{Array2, Axis, array, s};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize, Default)]
#[serde(tag = "transformation")]
pub struct AverageTransform {
    count: Option<usize>,
}

impl Transformer for AverageTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let mask = s![.., 1..;2]; // every second column
        self.count = Some(dataset.data.slice(mask).len_of(Axis(1)));
        let average_intensity = dataset.data.slice(mask).mean_axis(Axis(1)).unwrap();
        let wavenumber_axis = dataset.data.slice(s![.., 0]);
        let mut averaged = Array2::default((0, 2));
        let mut buf = array![0.0, 0.0];
        for (x, y) in wavenumber_axis.iter().zip(average_intensity.iter()) {
            // Filter NaN
            if x.is_nan() || y.is_nan() {
                continue;
            }
            buf[0] = *x;
            buf[1] = *y;
            averaged.push_row(buf.view())?;
        }

        dataset.data = averaged;
        Ok(())
    }
}
