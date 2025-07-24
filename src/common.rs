use crate::gui::TransformerGUI;
use crate::spe_rs::SpeData;
use crate::transformations::calibration::CalibrationTransform;
use crate::transformations::offset::OffsetIOBuffers;
use crate::transformations::{
    align::AlignTransform, append::AppendTransform, average::AverageTransform,
    baseline::BaselineTransform, count_conversion::CountConversionTransform,
    despike::DespikeTransform, finning::FinningTransform, integrate::IntegrateTransform,
    mask_pixels::MaskTransform, normalize::NormalizeTransform, offset::OffsetTransform,
    reshape::ReshapeTransform, select::SelectTransform, shift::RamanShiftTransform,
    subtract::SubtractTransform,
};
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use csv::ReaderBuilder;
use egui_plot::PlotPoints;
use ndarray::{array, Array2, ArrayBase, Axis, Ix1, ViewRepr};
use ndarray_csv::Array2Reader;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::str::FromStr;

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy)]
pub struct Pair<T>
where
    T: Display,
{
    pub a: T,
    pub b: T,
}

#[derive(Debug)]
pub enum PairParsingError {
    General,
    A,
    B,
}

impl std::error::Error for PairParsingError {}

impl std::fmt::Display for PairParsingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PairParsingError::General => write!(
                f,
                "Could not parse string as pair of numbers, use \"<value>,<value>\", no spaces."
            ),
            PairParsingError::A => {
                write!(f, "Could not parse first value as number.")
            }
            PairParsingError::B => {
                write!(f, "Could not parse last value as number.")
            }
        }
    }
}

impl<T> std::str::FromStr for Pair<T>
where
    T: Display + FromStr,
{
    type Err = PairParsingError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (a_str, b_str) = match s.split_once(',') {
            None => return Err(PairParsingError::General),
            Some((a, b)) => (a, b),
        };
        let a = match T::from_str(a_str) {
            Err(_) => return Err(PairParsingError::A),
            Ok(a) => a,
        };
        let b = match T::from_str(b_str) {
            Err(_) => return Err(PairParsingError::B),
            Ok(b) => b,
        };
        Ok(Self { a, b })
    }
}

pub fn input_data_to_string(filepath: &Option<std::path::PathBuf>) -> Result<String> {
    let mut input_string = String::new();
    match filepath {
        Some(fp) => {
            File::open(fp)?.read_to_string(&mut input_string)?;
        }
        None => {
            let (tx, rx) = std::sync::mpsc::channel::<String>();
            // Try read from stdin in background thread. This
            // considered as timed-out if nothing is returned within
            // 100 ms.
            std::thread::spawn(move || {
                let mut input_string = String::new();
                match BufReader::new(std::io::stdin()).read_to_string(&mut input_string) {
                    Ok(_) => {
                        tx.send(input_string)
                            .unwrap_or_else(|e| eprintln!("ERROR: {e}"));
                    }
                    Err(_) => {
                        eprintln!("WARNING: could not read data from STDIN, proceeding with empty input data.");
                        tx.send(String::new())
                            .unwrap_or_else(|e| eprintln!("ERROR: {e}"));
                    }
                }
            });
            // wait for 100 ms, hopefully by then all data was read from stdin
            std::thread::sleep(std::time::Duration::from_millis(100));
            if let Ok(s) = rx.try_recv() {
                input_string = s
            }
        }
    };
    Ok(input_string)
}

#[derive(Clone, Default, Debug)]
pub struct Dataset {
    pub data: Array2<f64>,
    pub metadata: String,
    pub previous_comments: String,
}

impl Dataset {
    /// iterate over frames of dataset (every second column), yielding mutable refs
    pub fn iter_mut_frames(&mut self) -> impl Iterator<Item = ArrayBase<ViewRepr<&mut f64>, Ix1>> {
        self.data.axis_iter_mut(Axis(1)).skip(1).step_by(2)
    }
    /// iterate over selected frames of dataset (every second column), yielding mutable refs
    ///
    /// if targtes = None, iterate over all frames
    pub fn iter_mut_selected_frames(
        &mut self,
        targets: &Option<Vec<usize>>,
    ) -> impl Iterator<
        Item = (
            ArrayBase<ViewRepr<&mut f64>, Ix1>,
            ArrayBase<ViewRepr<&mut f64>, Ix1>,
        ),
    > {
        // TODO: Looks like selecting frames does currently not work here
        // indices of the x-axes of the frames in the dataset
        let _indices: Vec<_> = match targets {
            None => (0..=self.data.len()).step_by(2).collect(),
            Some(ts) => ts.iter().map(|n| (n - 1) * 2).collect(),
        };

        let mut refs = Vec::with_capacity(self.data.ncols() / 2);
        let mut refx = None;
        let mut count_ref = 0;
        // Make reference to every other column, starting at first
        // (index 0) column. (Iterator does not work due to borrow checker shenanigans.)
        for r in self.data.axis_iter_mut(Axis(1)) {
            match count_ref {
                0 => {
                    refx = Some(r);
                    count_ref += 1;
                },
                1 => {
                    if let Some(rx) = refx {
                        refs.push((rx, r));
                        refx = None;
                    }
                    count_ref = 0;
                },
                _ => unreachable!("Partinioned frames into chunks of size other than 2. This should not have happened, please file an issue.")
            }
        }
        refs.into_iter()
    }
    pub fn from_csv(
        filepath: &Option<std::path::PathBuf>,
        comment: char,
        delimiter: char,
    ) -> Result<Self> {
        let input_string = input_data_to_string(filepath)?;
        let mut previous_comments: String = input_string
            .lines()
            .filter(|line| line.starts_with(comment))
            .map(|line| format!("{}\n", line))
            .collect();
        let filepath_msg = match filepath {
            None => "comments from input:\n".to_string(),
            Some(fp) => format!(
                "comments from input file {}:\n",
                fp.canonicalize()?.display()
            ),
        };
        if !previous_comments.is_empty() {
            previous_comments = filepath_msg + &previous_comments;
        }

        let mut csv_reader_config = ReaderBuilder::new();
        csv_reader_config
            .has_headers(false)
            .comment(Some(comment as u8))
            .delimiter(delimiter as u8)
            .trim(csv::Trim::All);

        let mut csv_reader = csv_reader_config.from_reader(input_string.as_bytes());
        let data = csv_reader.deserialize_array2_dynamic()?;
        Ok(Dataset {
            data,
            metadata: String::new(),
            previous_comments,
        })
    }
    pub fn from_spe(filepath: &std::path::Path) -> Result<Self, Box<dyn Error>> {
        let spe = SpeData::from_path(filepath)?;
        let previous_comments = spe.get_meta_data_string()?;

        let frames = spe.get_frames();
        let wavelength = spe.get_wavelength();

        let data = Array2::from_shape_fn((wavelength.len(), frames.len() * 2), |(i, j)| {
            if j % 2 == 0 {
                wavelength[i]
            } else {
                frames[(j - 1) / 2][i] as f64
            }
        });

        Ok(Dataset {
            data,
            metadata: String::new(),
            previous_comments,
        })
    }
    /// Write floats in 2D array to stdout in CSV format
    pub fn write(&self, mut buf: impl Write) -> Result<()> {
        // write program version and commit SHA to output buffer
        let mut version = env!("CARGO_PKG_VERSION").to_string();
        if let Some(sha) = option_env!("PROJECT_VERSION") {
            version += format!(" (git commit {})", sha).as_str()
        };
        let app_info_string = format!("# Raman CLI Tools version {}.\n# ---\n", version);
        buf.write(app_info_string.as_bytes())
            .with_context(|| "Unable to write to buffer.".to_string())?;

        // write metadata to stdout buffer
        let metadata: String = self
            .metadata
            .lines()
            .map(|line| format!("# {}\n", line))
            .collect();
        buf.write(metadata.as_bytes())
            .with_context(|| "Unable to write to buffer.".to_string())?;
        let prev_comments: String = self
            .previous_comments
            .lines()
            .map(|line| format!("# {}\n", line))
            .collect();
        buf.write(prev_comments.as_bytes())
            .with_context(|| "Unable to write to buffer.".to_string())?;
        // write numeric data to stdout buffer
        let mut wrt = csv::WriterBuilder::new().delimiter(b',').from_writer(buf);
        for row in self.data.outer_iter() {
            let record = row.map(std::string::ToString::to_string);
            wrt.write_record(record.iter())
                .with_context(|| format!("Unable to write record '{}' to buffer.", record))?;
        }
        wrt.flush()
            .with_context(|| String::from("Unable to write dataset to buffer."))?;
        Ok(())
    }
    /// test that a frame index is in bounds, return error otherwise
    pub fn verify_one_frame_in_bounds(&self, frame_no: usize) -> Result<()> {
        if frame_no == 0 {
            return Err(anyhow!("frame count starts at 1, frame \"0\" is invalid"));
        }
        if (frame_no - 1) * 2 >= self.data.ncols() {
            return Err(anyhow!(
                "frame number {} is out of bounds, largest valid frame number = {}",
                frame_no,
                self.data.ncols() / 2
            ));
        }
        Ok(())
    }
    /// return a subset of frames in a freshly copied array
    pub fn select_frames(&self, frames: &[usize], invert: bool) -> Result<Array2<f64>> {
        self.verify_frames_in_bounds(&frames)?;
        let selection: Vec<usize> = (0..self.data.ncols())
            .step_by(2)
            .filter(|n| invert ^ frames.contains(&(n / 2 + 1)))
            .flat_map(|n| [n, n + 1])
            .collect();
        if selection.is_empty() {
            return Err(anyhow!("selection does not yield any frames: {:?}", frames));
        }
        Ok(self.data.select(Axis(1), &selection))
    }
    pub fn verify_frames_in_bounds(&self, frames: &[usize]) -> Result<()> {
        for frame in frames {
            self.verify_one_frame_in_bounds(*frame)?;
        }
        Ok(())
    }
    /// create small dataset for testing purposes
    #[allow(dead_code)]
    pub fn new_test_dummy() -> Self {
        Dataset {
            metadata: "".to_string(),
            previous_comments: "".to_string(),
            data: array![
                [11., 12., 13., 14., 15., 16., 17., 18.],
                [21., 22., 23., 24., 25., 26., 27., 28.],
                [31., 32., 33., 34., 35., 36., 37., 38.],
                [41., 42., 43., 44., 45., 46., 47., 48.],
                [51., 52., 53., 54., 55., 56., 57., 58.],
                [61., 62., 63., 64., 65., 66., 67., 68.],
                [71., 72., 73., 74., 75., 76., 77., 78.],
                [81., 82., 83., 84., 85., 86., 87., 88.],
            ],
        }
    }
    /// build vector of PlotPoints from 2D array
    pub fn to_plot_points(&self) -> Vec<PlotPoints> {
        self.data
            .axis_iter(Axis(1))
            .step_by(2)
            .zip(self.data.axis_iter(Axis(1)).skip(1).step_by(2))
            .map(|(xs, ys)| xs.iter().zip(ys).map(|(x, y)| [*x, *y]).collect())
            .collect()
    }
}

pub struct Pipeline {
    pub transformations: Vec<Box<dyn TransformerGUI>>,
}

/// Match name of tranformation struct in yaml header to identifier of transformation struct
macro_rules! parse_yaml_transformer {
     ( $transformer_struct_name:ident, $yaml_segment:ident, $( $x:ident ),* ) => { // x = transformer struct identifiers
        match $transformer_struct_name {
        $(
            stringify!($x) => {
                    let transformer: $x = serde_yaml::from_str($yaml_segment)
                        .with_context(|| format!("Offending YAML input:\n{}", $yaml_segment))?;
                    Ok(Box::new(transformer))
            }
        )*
        _ => Err(anyhow!("Input string matches no known transformer:\n{}", $yaml_segment)),
        }
    };
}

/// Parse a single segment of the yaml header as a transformer, if it contains
/// 'transformation: ...' entry.
fn yaml_segment_to_transform(segment: &String) -> Result<Box<dyn TransformerGUI>> {
    let re = Regex::new(r"(?m)^transformation: ([a-zA-Z]*)$").unwrap();
    let transformer_struct_name = match re
        .captures(segment)
        .and_then(|c| c.get(1)) // get first capture group
        .map(|c| c.as_str()) // make it a str
    {
        None => return Err(anyhow!(format!("No transformer declared in input string: {}", segment))),
        Some(name) => name,
    };
    parse_yaml_transformer!(
        transformer_struct_name,
        segment,
        // REGISTER: New transformer must be registered here to be parsable from yaml headers
        AlignTransform,
        AppendTransform,
        AverageTransform,
        CalibrationTransform,
        CountConversionTransform,
        DespikeTransform,
        BaselineTransform,
        FinningTransform,
        IntegrateTransform,
        MaskTransform,
        NormalizeTransform,
        OffsetTransform,
        RamanShiftTransform,
        ReshapeTransform,
        SelectTransform,
        SubtractTransform
    )
}

impl Pipeline {
    pub fn from_cli_args(cli_args: Vec<Vec<String>>) -> Self {
        let mut transformations: Vec<Box<dyn TransformerGUI>> = vec![];
        // set gui flag so we know we must not react to plotting commands
        // which would cause a panic
        for subargs in cli_args {
            // REGISTER: new transformers must be entered here manually
            // (consider using a macro in the future)
            if let Some(command) = subargs.first() {
                match command.as_str() {
                    "align" => transformations.push(Box::new(AlignTransform::parse_from(subargs))),
                    "append" => {
                        transformations.push(Box::new(AppendTransform::parse_from(subargs)))
                    }
                    "average" => {
                        transformations.push(Box::new(AverageTransform::parse_from(subargs)))
                    }
                    "baseline" => {
                        transformations.push(Box::new(BaselineTransform::parse_from(subargs)))
                    }
                    "calibrate" => {
                        transformations.push(Box::new(CalibrationTransform::parse_from(subargs)))
                    }
                    "despike" => {
                        transformations.push(Box::new(DespikeTransform::parse_from(subargs)))
                    }
                    "finning" => {
                        transformations.push(Box::new(FinningTransform::parse_from(subargs)))
                    }
                    "mask" => transformations.push(Box::new(MaskTransform::parse_from(subargs))),
                    "offset" => {
                        transformations.push(Box::new(OffsetTransform::parse_from(subargs)))
                    }
                    "reshape" => {
                        transformations.push(Box::new(ReshapeTransform::parse_from(subargs)))
                    }
                    "select" => {
                        transformations.push(Box::new(SelectTransform::parse_from(subargs)))
                    }
                    "shift" => {
                        transformations.push(Box::new(RamanShiftTransform::parse_from(subargs)))
                    }
                    "subtract" => {
                        transformations.push(Box::new(SubtractTransform::parse_from(subargs)))
                    }
                    "count-conversion" => transformations
                        .push(Box::new(CountConversionTransform::parse_from(subargs))),
                    "integrate" => {
                        transformations.push(Box::new(IntegrateTransform::parse_from(subargs)))
                    }
                    "normalize" => {
                        transformations.push(Box::new(NormalizeTransform::parse_from(subargs)))
                    }
                    "default" => transformations = default_transformations(),
                    _ => {} // transformers for which GUI is not implemented:
                            // "mask" => transformations.push(Box::new(MaskTransform::parse_from(subargs))),
                }
            };
        }
        Self { transformations }
    }
    pub fn from_yaml_header(yaml_header: &str) -> Result<Self> {
        let mut transformations = vec![];
        for segment in yaml_header.split("---") {
            let segment = segment.replace("# ", "").trim().to_string();
            if segment.contains("transformation: ") {
                transformations.push(yaml_segment_to_transform(&segment)?);
            }
        }
        Ok(Self { transformations })
    }
    pub fn apply(&mut self, ds: &mut Dataset) -> Result<()> {
        for transformation in &mut self.transformations {
            transformation.apply(ds)?;
        }
        Ok(())
    }
}

pub fn default_transformations() -> Vec<Box<dyn TransformerGUI>> {
    let mut transformations: Vec<Box<dyn TransformerGUI>> = vec![];
    transformations.push(Box::new(ReshapeTransform { rows: 1340 }));
    transformations.push(Box::new(FinningTransform {
        threshold: 2.5,
        iterations: 4,
    }));
    transformations.push(Box::new(AverageTransform {}));
    transformations.push(Box::new(OffsetTransform {
        offset: 0.05,
        percentile: true,
        target_frames: None,
        gui_text_buffers: OffsetIOBuffers {
            value: "0.05".into(),
        },
    }));
    let mut rst = RamanShiftTransform {
        wavelength: 532.1,
        refractive_index: 1.000264,
        correction: Some(0.0),
        ..Default::default()
    };
    rst.update_text_buffers();
    transformations.push(Box::new(rst));
    transformations.push(Box::new(CountConversionTransform::default()));
    transformations
}

#[cfg(test)]
mod tests {
    use crate::transformations::finning::FinningTransform;
    use serde_yaml;

    #[test]
    fn test_parse_header() {
        let mut test_header = "# ---
# transformation: ReshapeTransform
# rows: 1340
# transformation: FinningTransform
# threshold: 2.5
# iterations: 100
# transformation: AverageTransform
# transformation: PlotTransform
# ---
"
        .to_string();
        test_header = test_header.replace("# ", "").replace("---\n", "");
        let mut commands_yaml: Vec<String> = Vec::new();
        let mut current_yaml = String::new();
        // sort arguments by command
        for row in test_header.split_inclusive("\n") {
            if row.contains("transformation: ") {
                if !current_yaml.is_empty() {
                    commands_yaml.push(current_yaml);
                }
                current_yaml = row.to_owned();
            } else {
                current_yaml.push_str(row);
            }
        }
        commands_yaml.push(current_yaml);
        for yaml_input in commands_yaml.clone() {
            if yaml_input.contains("FinningTransform") {
                let transform: FinningTransform = serde_yaml::from_str(&yaml_input).unwrap();
                dbg!(transform);
            }
        }
        assert_eq!(commands_yaml, vec!["foo".to_string()]);
    }
}
