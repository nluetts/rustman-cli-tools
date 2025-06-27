use crate::common::{Dataset, Pair};
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct MaskTransform {
    #[clap(help = "frame,pixel pairs of pixels that shall be masked")]
    pub(crate) mask: Vec<Pair<usize>>,
}

impl Transformer for MaskTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        // re-organize the data structure holding the mask to simplify
        // the actual masking
        let mut mask: HashMap<usize, HashSet<usize>> = HashMap::new();
        let ncols = dataset.data.ncols();
        let nrows = dataset.data.nrows();
        for Pair { a, b } in self.mask.iter() {
            // the masked points use 1-based indexing and have to be translated
            // into 0-based indices of the 2D array holding the data; note
            // that only every other odd column contains the intensities
            // that have to be manipulated, thus frame_idx = 2a - 1
            let (frame_idx, pixel_idx) = (2 * a - 1, b - 1);
            if frame_idx >= ncols || pixel_idx >= nrows {
                eprintln!("frame,pixel = {a},{b} is out of bounds");
                continue;
            }
            if !mask.contains_key(&pixel_idx) {
                mask.insert(pixel_idx, HashSet::new());
            }
            mask.get_mut(&pixel_idx).map(|js| js.insert(frame_idx));
        }

        // the re-organized mask is applied
        for (pixel_idx, frame_indices) in mask {
            // the mean of the intensities in non-masked frames is used to
            // replace the intensities in masked frames
            let mean = {
                let mut sum = 0.0;
                let mut n = 0;
                for (idx, val) in dataset
                    .data
                    .row(pixel_idx)
                    .iter()
                    .enumerate()
                    .skip(1)
                    .step_by(2)
                {
                    if frame_indices.contains(&idx) {
                        continue;
                    }
                    sum += val;
                    n += 1;
                }
                if n == 0 {
                    eprintln!("no data left for pixel {}, skipping", pixel_idx + 1);
                    continue;
                } else {
                    sum / n as f64
                }
            };
            for frame_idx in frame_indices {
                dataset.data[[pixel_idx, frame_idx]] = mean;
            }
        }
        Ok(())
    }
}
