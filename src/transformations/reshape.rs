use crate::common::Dataset;
use crate::transformations::Transformer;
use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct ReshapeTransform {
    #[clap(help = "New number of rows")]
    pub(crate) rows: usize,
}

/// Reshape data into new form, e.g. to partition dataset where several
/// scans were stored in a single pair of columns.
impl Transformer for ReshapeTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let number_rows = dataset.data.nrows();
        let number_cols = dataset.data.ncols();
        let number_cols_reshaped = number_cols * number_rows / self.rows;
        if self.rows * number_cols_reshaped != number_rows * number_cols {
            return Err(anyhow!(format!(
                "Cannot reshape data into form ({}, {}).",
                self.rows, number_cols_reshaped
            )));
        }

        let mut data_reshaped = ndarray::Array2::<f64>::zeros((self.rows, number_cols_reshaped));
        let mut a = 0; // a and b are indices of the reshaped array
        let mut b = 0;
        if number_cols_reshaped == 0 {
            return Err(anyhow!("number of reshaped rows must not be zero"));
        }
        for j in (0..number_cols_reshaped - 1).step_by(2) {
            for i in 0..self.rows {
                data_reshaped[[i, j]] = dataset.data[[a, b]];
                data_reshaped[[i, j + 1]] = dataset.data[[a, b + 1]];
                a += 1;
                if a == number_rows {
                    b += 2;
                    a = 0;
                }
            }
        }
        dataset.data = data_reshaped;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::common::Dataset;
    use crate::transformations::{reshape::ReshapeTransform, Transformer};
    use clap::Parser;
    use ndarray::array;

    #[test]
    fn test_reshape_transform() {
        let mut dataset = Dataset {
            metadata: "".to_string(),
            previous_comments: "".to_string(),
            data: array![
                [11., 12., 13., 14.],
                [21., 22., 23., 24.],
                [31., 32., 33., 34.],
                [41., 42., 43., 44.],
                [51., 52., 53., 54.],
                [61., 62., 63., 64.],
                [71., 72., 73., 74.],
                [81., 82., 83., 84.],
            ],
        };
        // transform into same shape
        let mut transform = ReshapeTransform::parse_from(["reshape", "8"]);
        transform.apply(&mut dataset).unwrap();
        // reshape into same number of rows must not change dataset
        assert_eq!(
            dataset.data,
            array![
                [11., 12., 13., 14.],
                [21., 22., 23., 24.],
                [31., 32., 33., 34.],
                [41., 42., 43., 44.],
                [51., 52., 53., 54.],
                [61., 62., 63., 64.],
                [71., 72., 73., 74.],
                [81., 82., 83., 84.],
            ]
        );
        // transform into wider shape
        let mut transform = ReshapeTransform::parse_from(["reshape", "4"]);
        transform.apply(&mut dataset).unwrap();
        // reshape into same number of rows must not change dataset
        assert_eq!(
            dataset.data,
            array![
                [11., 12., 51., 52., 13., 14., 53., 54.],
                [21., 22., 61., 62., 23., 24., 63., 64.],
                [31., 32., 71., 72., 33., 34., 73., 74.],
                [41., 42., 81., 82., 43., 44., 83., 84.],
            ]
        );
        // transform into more narrow shape
        let mut transform = ReshapeTransform::parse_from(["reshape", "16"]);
        transform.apply(&mut dataset).unwrap();
        // reshape into same number of rows must not change dataset
        assert_eq!(
            dataset.data,
            array![
                [11., 12.],
                [21., 22.],
                [31., 32.],
                [41., 42.],
                [51., 52.],
                [61., 62.],
                [71., 72.],
                [81., 82.],
                [13., 14.],
                [23., 24.],
                [33., 34.],
                [43., 44.],
                [53., 54.],
                [63., 64.],
                [73., 74.],
                [83., 84.],
            ]
        )
    }
}
