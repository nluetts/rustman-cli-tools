use super::Transformer;
use crate::common::Dataset;
use anyhow::Result;
use clap::Parser;
use ndarray::Axis;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct AppendTransform {
    #[clap(parse(from_os_str))]
    pub filepath: Option<std::path::PathBuf>,
    #[clap(
        short,
        long,
        help = "the character starting a comment",
        default_value = "#"
    )]
    pub comment: char,
    #[clap(short, long, help = "the delimiting character", default_value = ",")]
    pub delimiter: char,
    #[clap(
        short,
        long,
        help = "if true, append data horizontally (as rows), e.g. to add scans"
    )]
    pub horizontal: bool,
}

impl Transformer for AppendTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let new_dataset = Dataset::from_csv(&self.filepath, self.comment, self.delimiter)?;
        dataset.previous_comments += "\n";
        dataset.previous_comments += &new_dataset.previous_comments;
        dataset.data = if self.horizontal {
            ndarray::concatenate(Axis(0), &[dataset.data.view(), new_dataset.data.view()])?
        } else {
            ndarray::concatenate(Axis(1), &[dataset.data.view(), new_dataset.data.view()])?
        };
        Ok(())
    }
}

#[cfg(test)]
mod test {}
