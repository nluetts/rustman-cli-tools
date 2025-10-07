use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct RoiTransform {
    #[clap(help = "ROI to select")]
    pub(crate) roi: usize,
    #[clap(help = "Number of ROIs")]
    pub(crate) num_rois: usize,
}

impl Transformer for RoiTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let num_frames = dataset.data.ncols() / 2;
        let selected_frames: Vec<_> = (1..=num_frames)
            .skip(self.roi - 1)
            .step_by(self.num_rois)
            .collect();
        dataset.data = dataset.select_frames(&selected_frames, false)?;
        Ok(())
    }
}
