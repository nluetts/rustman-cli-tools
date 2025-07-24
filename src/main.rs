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

use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

use crate::cli::Preprocessor;
use ansi_term::Colour::Yellow;
use anyhow::{anyhow, Result};
use common::{input_data_to_string, Dataset, Pipeline};
use gui::gui_loop;
use plot::PlotWindow;
use sha256::digest;

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
    // if preprocessor.args.watch {
    //    run_file_watch()?;
    //} else {
    //     run_once(preprocessor)?;
    // }

    Ok(())
}

fn run_once(mut preprocessor: Preprocessor) -> Result<(), anyhow::Error> {
    let mut dataset = preprocessor.get_input_data()?;
    let mut pipeline = preprocessor.get_pipeline();
    pipeline.apply(&mut dataset)?;
    preprocessor.print_dataset(&dataset)?;
    Ok(())
}

fn run_file_watch() -> Result<(), anyhow::Error> {
    let dataset_arcmutex = Arc::new(Mutex::new(Dataset::default()));
    let info_arcmutex = Arc::new(Mutex::new(String::new()));
    // dsam is moved into thread that handles data transformations
    let dataset_arcmutex_clone = dataset_arcmutex.clone();
    let info_arcmutex_clone = info_arcmutex.clone();
    let _join_handle = std::thread::spawn(move || -> Result<()> {
        let mut input_sha256 = "".to_string();
        let preprocessor = Preprocessor::from_cli_args();
        let mut count = 0;
        loop {
            let input_string = input_data_to_string(&preprocessor.args.filepath)?;
            // file may seem empty on write by accident, this is ignored here:
            if input_string.is_empty() {
                sleep(Duration::from_millis(50));
                continue;
            }
            let yaml_header: String = input_string
                .lines()
                .filter(|line| line.starts_with(preprocessor.args.comment))
                .map(|line| format!("{}\n", line))
                .collect();
            let new_input_sha256 = digest(yaml_header.clone());
            if new_input_sha256 != input_sha256 {
                eprintln!(
                    "{}",
                    Yellow.paint("File update detected, re-running pipeline ...")
                );
                count += 1;
            } else {
                // if file was not updated, we do nothing
                sleep(Duration::from_millis(50));
                continue;
            }
            input_sha256 = new_input_sha256;
            // reset info box text
            info_arcmutex.lock().unwrap().clear();
            // preprocessor reading the dataset from the source file defined in the
            // yaml header
            let mut inner_preprocessor = match Preprocessor::from_yaml_header(&yaml_header, false) {
                Err(e) => {
                    info(&info_arcmutex, e.to_string());
                    continue;
                }
                Ok(prp) => prp,
            };
            let mut dataset = match inner_preprocessor.get_input_data() {
                Err(e) => {
                    let msg = format!("Unable to fetch input data from input file: {}", e);
                    info(&info_arcmutex, msg);
                    continue;
                }
                Ok(dataset) => dataset,
            };
            let mut pipeline = match Pipeline::from_yaml_header(&yaml_header) {
                Err(e) => {
                    let msg = format!("Unable to parse YAML header as pipeline:\n\n{:?}", e);
                    info(&info_arcmutex, msg);
                    continue;
                }
                Ok(pipeline) => pipeline,
            };
            if let Err(e) = pipeline.apply(&mut dataset) {
                let msg = format!("Unable to apply pipeline:\n\n{:?}", e);
                info(&info_arcmutex, msg);
                continue;
            }
            // write transformation results back to watched file
            match &preprocessor.args.filepath {
                None => {
                    return Err(anyhow!(
                        "Watching files with data coming from STDIN is not supported."
                    ))
                }
                Some(filepath) => {
                    let filepath = filepath.clone();
                    let handle = std::fs::File::create(filepath)?;
                    let wrt = std::io::BufWriter::new(handle);
                    dataset.write(wrt)?;
                }
            };

            // FIXME: breaking of loop has to be handeled differently
            if count == 999999999 {
                break;
            }
            if let Ok(mut guard) = dataset_arcmutex.lock() {
                guard.data = dataset.data;
                guard.metadata = dataset.metadata;
            };
        }
        Ok(())
    });
    // TODO: if this is included, plot does not show, if not, fatal errors from pipeline are note reported
    // join_handle.join().unwrap()?;
    // setup plotting
    let options = eframe::NativeOptions {
        // initial_window_size: Some(egui::vec2(800.0, 500.0)),
        ..Default::default()
    };
    let pw = PlotWindow::new(
        dataset_arcmutex_clone,
        None,
        vec![],
        info_arcmutex_clone,
        None,
        None,
    );
    eframe::run_native("Dataset Plot", options, Box::new(|_cc| Box::new(pw)));
    Ok(())
}

fn info(iam: &Arc<Mutex<String>>, msg: String) {
    iam.lock().unwrap().clone_from(&msg);
    eprintln!("{}", &msg);
    eprintln!("Fix and save file again to retry.");
}
