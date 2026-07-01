use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use ndarray::Axis;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct MovingAverageTransform {
    #[clap(
        short,
        long,
        default_value_t = 3,
        help = "Window size for the moving average filter."
    )]
    pub window_size: usize,
}

impl Transformer for MovingAverageTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }

    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        if self.window_size < 1 {
            return Err(anyhow::anyhow!("Window size must be at least 1"));
        }

        // Iterate over each frame (every second column)
        for mut frame in dataset.iter_mut_frames() {
            let mut smoothed = frame.to_owned();

            // Apply moving average
            for i in 0..frame.len() {
                let start = if i >= self.window_size / 2 {
                    i - self.window_size / 2
                } else {
                    0
                };

                let end = std::cmp::min(start + self.window_size, frame.len());

                let window = &frame.slice(ndarray::s![start..end]);
                smoothed[i] = window.mean().unwrap_or(frame[i]);
            }

            frame.assign(&smoothed);
        }

        Ok(())
    }
}
