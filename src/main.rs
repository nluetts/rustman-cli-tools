#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
mod cli;
mod common;
mod gui;
mod gui_plot_extensions;
mod plot;
mod spe_rs;
mod transformations;
mod utils;

mod test;

use crate::cli::Preprocessor;
use anyhow::Result;
use gui::gui_loop;

fn main() -> Result<()> {
    //gui_loop()?;
    //return Ok(());
    let mut preprocessor = Preprocessor::from_cli_args();
    if preprocessor.gui_mode {
        gui_loop(preprocessor)?;
    } else {
        let mut pipeline = preprocessor.get_pipeline();
        let mut dataset = preprocessor.get_input_data()?;
        pipeline.apply(&mut dataset)?;
        preprocessor.print_dataset(&dataset)?;
    }

    Ok(())
}
