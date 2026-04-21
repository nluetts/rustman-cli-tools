use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use ndarray::OwnedRepr;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct RoiAverageTransform {
    #[clap(help = "Number of ROIs")]
    pub(crate) num_rois: usize,
}

impl Transformer for RoiAverageTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let cols: Vec<_> = dataset.data.axis_iter(ndarray::Axis(1)).collect();
        let mut averaged: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
            ndarray::Array2::default((dataset.data.shape()[0], 0));
        for chnk in cols.chunks(self.num_rois * 2) {
            let Some(xs) = chnk.first() else { continue };
            let mut ys: ndarray::ArrayBase<OwnedRepr<f64>, _> = ndarray::ArrayBase::zeros(xs.len());
            for ys_chnk in chnk.iter().skip(1).step_by(2) {
                for (yi, yic) in ys.iter_mut().zip(ys_chnk.iter()) {
                    *yi += yic / (self.num_rois as f64)
                }
            }
            averaged.push_column(*xs)?;
            averaged.push_column(ys.view())?;
        }

        dataset.data = averaged;
        Ok(())
    }
}
