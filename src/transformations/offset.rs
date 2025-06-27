use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use ndarray::Array1;
use ndarray_stats::Quantile1dExt;
use noisy_float::types::N64;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct OffsetTransform {
    #[clap(help = "Offset data by this value")]
    pub(crate) offset: f64,
    #[clap(
        short,
        long,
        action,
        help = "If flag is set, subtract this percentile from the frame."
    )]
    pub(crate) percentile: bool,
    #[clap(short, long, help = "Apply offset to these frames.")]
    pub(crate) target_frames: Option<Vec<usize>>,
    #[serde(skip)]
    #[clap(skip)]
    pub gui_text_buffers: OffsetIOBuffers,
}

#[derive(Default, Debug, Clone)]
pub struct OffsetIOBuffers {
    pub value: String,
}

impl Transformer for OffsetTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let target_frames = match &self.target_frames {
            None => (0..(dataset.data.ncols() / 2 + 1)).collect(),
            Some(frames) => {
                dataset.verify_frames_in_bounds(frames)?;
                frames.clone()
            }
        };
        for (col_no, mut vals) in dataset.iter_mut_frames().enumerate() {
            if !target_frames.contains(&(col_no + 1)) {
                continue;
            }
            let offset = match self.percentile {
                true => {
                    // we filter out nan values explicitly
                    let mut tmp: Array1<N64> = vals
                        .iter()
                        .filter(|x| !x.is_nan())
                        .map(|x| N64::new(*x))
                        .collect();
                    let quantile = tmp.quantile_mut(
                        N64::from_f64(self.offset),
                        &ndarray_stats::interpolate::Nearest,
                    )?;
                    f64::from(-quantile)
                }
                false => self.offset,
            };
            vals += offset;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::OffsetTransform;
    use crate::common::Dataset;
    use crate::transformations::Transformer;
    use ndarray::array;

    #[test]
    fn test_offset_transfort() {
        let yaml_input = "transformation: OffsetTransform
offset: 2.0
percentile: false
target_scans:
- 1
- 4";
        let mut transform: OffsetTransform = serde_yaml::from_str(yaml_input).unwrap();
        let mut dataset = Dataset {
            metadata: "".to_string(),
            previous_comments: "".to_string(),
            data: array![
                [11., 12., 13., 14., 11., 12., 13., 14.],
                [21., 22., 23., 24., 21., 22., 23., 24.],
                [31., 32., 33., 34., 31., 32., 33., 34.],
                [41., 42., 43., 44., 41., 42., 43., 44.],
                [51., 52., 53., 54., 51., 52., 53., 54.],
                [61., 62., 63., 64., 61., 62., 63., 64.],
                [71., 72., 73., 74., 71., 72., 73., 74.],
                [81., 82., 83., 84., 81., 82., 83., 84.],
            ],
        };
        let exprected_data = array![
            [11., 14., 13., 14., 11., 12., 13., 16.],
            [21., 24., 23., 24., 21., 22., 23., 26.],
            [31., 34., 33., 34., 31., 32., 33., 36.],
            [41., 44., 43., 44., 41., 42., 43., 46.],
            [51., 54., 53., 54., 51., 52., 53., 56.],
            [61., 64., 63., 64., 61., 62., 63., 66.],
            [71., 74., 73., 74., 71., 72., 73., 76.],
            [81., 84., 83., 84., 81., 82., 83., 86.],
        ];
        transform.apply(&mut dataset).unwrap();
        assert_eq!(dataset.data, exprected_data)
    }
}
