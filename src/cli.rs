use crate::common::{Dataset, Pipeline};
use crate::plot::PlotTransform;
use crate::transformations::calibration::CalibrationTransform;
use crate::transformations::{
    align::AlignTransform, append::AppendTransform, average::AverageTransform,
    count_conversion::CountConversionTransform, despike::DespikeTransform,
    draw_baseline::DrawBaselineTransform, finning::FinningTransform, integrate::IntegrateTransform,
    mask_pixels::MaskTransform, normalize::NormalizeTransform, offset::OffsetTransform,
    reshape::ReshapeTransform, select::SelectTransform, shift::RamanShiftTransform,
    subtract::SubtractTransform,
};
use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::io::BufWriter;

#[derive(Parser, Serialize, Deserialize, Debug)]
#[clap(name = "Raman CLI Tools")]
pub struct Cli {
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
    #[clap(subcommand)]
    #[serde(skip_serializing)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Deserialize, Debug)]
pub enum Commands {
    // REGISTER: new transformers must be entered here.
    /// Align frames.
    Align(AlignTransform),
    /// Append a dataset from a further input file.
    Append(AppendTransform),
    /// Average intensity.
    Average(AverageTransform),
    /// Draw and subtract a spline baseline (from all frames).
    Baseline(DrawBaselineTransform),
    /// Apply a linear calibration to the wavelength axis.
    Calibration(CalibrationTransform),
    /// Convert from counts to photoelectrons per second.
    CountConverion(CountConversionTransform),
    /// Apply laplace edge-detection despike algorithm.
    Despike(DespikeTransform),
    /// Apply finning despike algorithm.
    Finning(FinningTransform),
    /// Integrate frames in given interval(s).
    Integrate(IntegrateTransform),
    /// Manually mask data points by pixel and frame number
    Mask(MaskTransform),
    /// Normalize frames.
    Normalize(NormalizeTransform),
    /// Add offset to value columns.
    Offset(OffsetTransform),
    /// Plot the dataset.
    Plot(PlotTransform),
    /// Reshape dataset into different form.
    Reshape(ReshapeTransform),
    /// Calculate Raman shift.
    Shift(RamanShiftTransform),
    // Subtract frame from other frames.
    Subtract(SubtractTransform),
    /// Select frames.
    Select(SelectTransform),
    /// Run default transformers
    Default,
    /// Run in GUI mode.
    GUI,
}

const COMMANDS: [&str; 19] = [
    // REGISTER: new transformers must get entry here.
    "align",
    "append",
    "average",
    "baseline",
    "calibration",
    "count-conversion",
    "default",
    "despike",
    "finning",
    "gui",
    "integrate",
    "mask",
    "normalize",
    "offset",
    "plot",
    "reshape",
    "select",
    "shift",
    "subtract",
];

pub struct Preprocessor {
    pub args: Cli,
    pub subcommand_args: Option<Vec<Vec<String>>>,
    pub gui_mode: bool,
    pub reload_pipeline: bool,
}

impl Preprocessor {
    pub fn from_cli_args() -> Self {
        // this is basically a hack that let's us chain several commands in
        // clap by going through the args passed from the command line one by
        // one and splitting into a new sublist if a subcommand name is found
        let args_raw: Vec<String> = std::env::args().collect();
        let gui_mode = args_raw.iter().any(|arg| arg == "gui");
        let reload_pipeline = args_raw.iter().any(|arg| arg == "reload");
        // sort arguments by command
        let mut args_sorted_by_command: Vec<Vec<String>> = vec![vec![]];
        for arg in args_raw {
            if COMMANDS.contains(&arg.as_str()) {
                args_sorted_by_command.push(vec![arg]);
            } else {
                // we can unwrap because the vector is guaranteed to have a single element
                args_sorted_by_command.last_mut().unwrap().push(arg);
            }
        }
        // the first subset of arguments are always for the preprocessor
        let (preprocessor_args, subcommand_args) = match args_sorted_by_command.split_first() {
            None => (args_sorted_by_command.first().unwrap(), None),
            Some((p, s)) => (p, Some(s.to_vec())),
        };
        let preprocessor_args = Cli::parse_from(preprocessor_args);
        let mut prp = Preprocessor {
            args: preprocessor_args,
            subcommand_args,
            gui_mode,
            reload_pipeline,
        };
        if prp.args.filepath.is_some() {
            prp.args.filepath = Some(prp.args.filepath.unwrap().canonicalize().unwrap());
        }
        prp
    }

    pub fn get_input_data(&mut self) -> Result<Dataset> {
        let mut dataset = if self
            .args
            .filepath
            .as_ref()
            .is_some_and(|path| path.extension().unwrap_or_default() == "spe")
        {
            Dataset::from_spe(&self.args.filepath.as_ref().unwrap())
                .map_err(|e| anyhow!("Could not read SPE file: {e}"))?
        } else {
            Dataset::from_csv(&self.args.filepath, self.args.comment, self.args.delimiter)?
        };
        dataset.metadata =
            "preprocessor: arguments\n".to_owned() + &serde_yaml::to_string(&self.args)? + "---\n";
        Ok(dataset)
    }

    pub fn get_pipeline(&self) -> Pipeline {
        Pipeline::from_cli_args(self.subcommand_args.clone().unwrap_or_else(|| vec![vec![]]))
    }
    pub fn get_gui_pipeline(&self) -> Vec<Box<dyn crate::gui::TransformerGUI>> {
        vec![]
    }
    pub fn print_dataset(&self, dataset: &Dataset) -> Result<()> {
        let buf = BufWriter::new(std::io::stdout());
        dataset.write(buf)?;
        Ok(())
    }

    pub fn from_yaml_header(yaml_header: &str, gui_mode: bool) -> Result<Self> {
        let preprocessor_yaml = if let Some(yaml) = yaml_header
            .split("---")
            .map(|segment| segment.replace("# ", "").trim().to_string())
            .find(|segment| segment.contains("preprocessor: arguments"))
        {
            yaml
        } else {
            return Err(anyhow!(format!(
                "Unable to parse preprocessor from YAML header,\
                \nmissing 'preprocessor: arguments' segment"
            )));
        };
        let args = serde_yaml::from_str::<Cli>(&preprocessor_yaml)
            .with_context(|| format!("Offending YAML input:\n{}", preprocessor_yaml))?;
        Ok(Self {
            args,
            subcommand_args: None,
            gui_mode,
            reload_pipeline: false,
        })
    }
}
