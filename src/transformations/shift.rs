use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use ndarray::s;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct RamanShiftTransform {
    #[clap(help = "Laser wavelength in nm.")]
    pub wavelength: f64,
    #[clap(
        short,
        long,
        default_value("1.000264"),
        help = "Refractive index of air used to calculate vacuum wavenumbers from wavelength."
    )]
    pub refractive_index: f64,
    #[clap(
        short,
        long,
        help = "Optional corrective offset added to calculated wavenumbers."
    )]
    pub correction: Option<f64>,
    #[serde(skip)]
    #[clap(skip)]
    pub gui_text_buffers: RamanShiftIOBuffers,
}

#[derive(Default, Debug, Clone)]
pub struct RamanShiftIOBuffers {
    pub wavelength: String,
    pub correction: String,
    pub refractive_index: String,
}

impl Transformer for RamanShiftTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let correction = self.correction.unwrap_or(0.0);
        dataset
            .data
            .slice_mut(s![.., 0..;2])
            // this parallel inplace map is perhaps an overkill ... but why not
            .par_map_inplace(|x| {
                *x = (1e7_f64 / self.wavelength - 1e7_f64 / *x) / self.refractive_index + correction
            });
        Ok(())
    }
}
