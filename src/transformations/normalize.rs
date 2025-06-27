use crate::common::{Dataset, Pair};
use crate::transformations::Transformer;
use crate::utils::{nearest_index, trapz};
use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct NormalizeTransform {
    #[clap(help = "Normalize data by this intensity at this x-value.")]
    pub(crate) xi: f64,
    #[clap(help = "If provided, integrate data between xi and xj and normalize to area.")]
    pub(crate) xj: Option<f64>,
    #[clap(
        short,
        long,
        action,
        help = "If flag is set, subtract local baseline when integrating.",
        requires = "xj"
    )]
    pub(crate) local_baseline: bool,
    #[clap(short, long, action, help = "Select frames to normalize")]
    pub(crate) target_frames: Option<Vec<usize>>,
    #[clap(short, long, action, help = "Select a region to filter")]
    pub(crate) filter_range: Option<Pair<f64>>,
    #[serde(skip)]
    #[clap(skip)]
    pub gui_text_buffers: NormalizeIOBuffers,
}

#[derive(Default, Debug, Clone)]
pub struct NormalizeIOBuffers {
    pub xi: String,
    pub xj: String,
    pub y_min: String,
    pub y_max: String,
}

impl Transformer for NormalizeTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let frames_iter = dataset.iter_mut_selected_frames(&self.target_frames);
        for (xs, mut ys) in frames_iter {
            let norm = match self.xj {
                // normalize to y-value closest to xi
                None => {
                    match nearest_index(&xs, self.xi) {
                        // unwrap: index from nearest_index() should always be valid
                        Some(idx) => *ys.get(idx).unwrap(),
                        None => return Err(anyhow!("could not find {} in dataset.", self.xi)),
                    }
                }
                // normalize to intergral between xi and xj
                Some(xj) => trapz(&xs, &ys, self.xi, xj, self.local_baseline)?,
            };
            if let Some(Pair { a, b }) = self.filter_range {
                
            } else {
                for yi in ys.iter_mut() {
                    *yi /= norm;
                }
                
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {}
