use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct SelectTransform {
    #[clap(help = "Numbers of frames to keep (counts starts at 1).")]
    pub(crate) frames: Vec<usize>,
    #[clap(
        short,
        long,
        action,
        help = "If flag is set, discard selected frames and leave the non-selected."
    )]
    pub(crate) invert: bool,
}

impl Transformer for SelectTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        dataset.data = dataset.select_frames(&self.frames, self.invert)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::SelectTransform;
    use crate::{common::Dataset, transformations::Transformer};

    #[test]
    fn test_select_transform() {
        let mut dataset = Dataset::new_test_dummy();
        let mut trsf = SelectTransform {
            frames: vec![1],
            invert: true,
        };
        trsf.transform(&mut dataset).unwrap();
        assert_eq!(
            array![
                [13., 14., 15., 16., 17., 18.],
                [23., 24., 25., 26., 27., 28.],
                [33., 34., 35., 36., 37., 38.],
                [43., 44., 45., 46., 47., 48.],
                [53., 54., 55., 56., 57., 58.],
                [63., 64., 65., 66., 67., 68.],
                [73., 74., 75., 76., 77., 78.],
                [83., 84., 85., 86., 87., 88.],
            ],
            dataset.data
        );
        trsf.frames = vec![2, 3];
        trsf.invert = false;
        let _ = trsf.transform(&mut dataset);
        assert_eq!(
            array![
                [15., 16., 17., 18.],
                [25., 26., 27., 28.],
                [35., 36., 37., 38.],
                [45., 46., 47., 48.],
                [55., 56., 57., 58.],
                [65., 66., 67., 68.],
                [75., 76., 77., 78.],
                [85., 86., 87., 88.],
            ],
            dataset.data
        );
    }
}
