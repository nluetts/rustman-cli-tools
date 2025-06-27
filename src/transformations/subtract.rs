use crate::transformations::Transformer;
use crate::{common::Dataset, utils::linear_resample_array};
use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct SubtractTransform {
    #[clap(help = "Column number of frame to subtract")]
    pub(crate) subtrahend: usize,
    #[clap(
        short,
        long,
        help = "Frame(s) to subtract from (if none given, subract subtrahend from all other frames in dataset)"
    )]
    pub(crate) minuends: Option<Vec<usize>>,
    #[clap(
        short,
        long,
        action,
        help = "If flag is set, subtract frame intensities without interpolating on same grid first"
    )]
    pub(crate) direct: bool,
}

impl Transformer for SubtractTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let mut minuends = if let Some(minuends) = &self.minuends {
            if minuends.contains(&self.subtrahend) {
                return Err(anyhow!(
                    "the minuend frames must not contain the subtrahend frame"
                ));
            }
            dataset.select_frames(minuends, false)?
        } else {
            dataset.select_frames(&[self.subtrahend], true)?
        };
        let subtrahend = dataset.select_frames(&[self.subtrahend], false)?;
        let grid = subtrahend.column(0);
        let sub_ys = subtrahend.column(1);
        for n in (0..minuends.ncols() - 1).step_by(2) {
            let ys = if !self.direct {
                linear_resample_array(&minuends.column(n), &minuends.column(n + 1), &grid)
            } else {
                // ignore spectral axes of minuends and subtract intensity data directly
                minuends.column(n + 1).to_owned()
            };
            let difference = ys - &sub_ys;
            minuends.column_mut(n + 1).assign(&difference);
            if !self.direct {
                minuends.column_mut(n).assign(&grid);
            }
        }
        dataset.data = minuends;
        Ok(())
    }
}
