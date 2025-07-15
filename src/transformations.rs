pub mod align;
pub mod append;
pub mod average;
pub mod baseline;
pub mod calibration;
pub mod count_conversion;
pub mod despike;
pub mod draw_baseline;
pub mod finning;
pub mod integrate;
pub mod mask_pixels;
pub mod normalize;
pub mod offset;
pub mod reshape;
pub mod select;
pub mod shift;
pub mod subtract;

use crate::common::Dataset;
use anyhow::Result;

pub trait Transformer: std::fmt::Debug {
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()>;
    fn config_to_string(&self) -> Result<String>;
    fn write_metadata_yaml(&self, dataset: &mut Dataset) -> Result<()> {
        let metadata = self.config_to_string()?;
        dataset.metadata += &metadata;
        dataset.metadata += "---\n";
        Ok(())
    }
    fn apply(&mut self, dataset: &mut Dataset) -> Result<()> {
        self.transform(dataset)?;
        self.write_metadata_yaml(dataset)?;
        Ok(())
    }
}
