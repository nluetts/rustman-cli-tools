use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use ndarray::{s, Axis};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct AverageTransform {}

impl Transformer for AverageTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let mask = s![.., 1..;2]; // every second column
        let average_intensity = dataset.data.slice(mask).mean_axis(Axis(1)).unwrap();
        let wavenumber_axis = dataset.data.slice(s![.., 0]);
        dataset.data = ndarray::stack(Axis(1), &[wavenumber_axis, average_intensity.view()])?;
        Ok(())
    }
}
