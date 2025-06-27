use crate::common::Dataset;
use crate::gui::TransformerGUI;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct CountConversionTransform {
    #[clap(help = "CCD exposure time in seconds.")]
    pub(crate) exposure: f64,
    // default value from PyLoN calibration certificate
    #[clap(
        short,
        long,
        help = "Count to photoelectron conversion factor.",
        default_value_t = 1.42857
    )]
    pub(crate) conversion_factor: f64,
    #[serde(skip)]
    #[clap(skip)]
    pub gui_text_buffers: CountConversionIOBuffers,
}

#[derive(Default, Debug, Clone)]
pub struct CountConversionIOBuffers {
    pub exposure: String,
    pub conversion_factor: String,
}

impl Transformer for CountConversionTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let num_rows = dataset.data.nrows();
        let num_cols = dataset.data.ncols();
        let mut prev_dx = 1.0;
        for j in (1..num_cols).step_by(2) {
            for i in 0..num_rows {
                let dx;
                if i == num_rows - 1 {
                    dx = prev_dx;
                } else {
                    // j - 1 : wavelength axes
                    dx = (dataset.data[[i + 1, j - 1]] - dataset.data[[i, j - 1]]).abs();
                    prev_dx = dx;
                }
                dataset.data[[i, j]] /= dx * self.exposure * self.conversion_factor;
            }
        }
        Ok(())
    }
}

impl Default for CountConversionTransform {
    fn default() -> Self {
        let mut cct = CountConversionTransform {
            exposure: 300.0,
            conversion_factor: 1.42857,
            gui_text_buffers: CountConversionIOBuffers::default(),
        };
        cct.update_text_buffers();
        cct
    }
}
