#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::mpsc::{channel, Receiver, Sender},
};

use anyhow::Result;
use eframe::egui;
use egui::{Color32, Slider, Ui};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use image::ColorType;
use ndarray_stats::QuantileExt;
use sha256::digest;

use crate::{
    cli::Preprocessor,
    common::{default_transformations, Dataset, Pair, Pipeline},
    gui_plot_extensions::{
        IntegrateExtensionGUI, MaskExtensionGUI, NormalizeExtensionGUI, PlotExtensionGUI,
        PlotExtensionResult, SplineExtensionGUI,
    },
    plot::PALETTE,
    transformations::{
        align::AlignTransform,
        append::AppendTransform,
        average::AverageTransform,
        baseline::BaselineTransform,
        count_conversion::CountConversionTransform,
        despike::DespikeTransform,
        finning::FinningTransform,
        integrate::IntegrateTransform,
        mask_pixels::MaskTransform,
        normalize::{NormalizeIOBuffers, NormalizeTransform},
        offset::OffsetTransform,
        reshape::ReshapeTransform,
        select::SelectTransform,
        shift::RamanShiftTransform,
        subtract::SubtractTransform,
        Transformer,
    },
};

pub fn gui_loop(mut preprocessor: Preprocessor) -> Result<()> {
    let options = eframe::NativeOptions {
        // initial_window_size: Some(egui::vec2(800.0, 600.0)),
        // maximized: true,
        ..Default::default()
    };
    let mut pipeline = preprocessor.get_pipeline();
    pipeline // update text input buffers of all transformers
        .transformations
        .iter_mut()
        .for_each(|tranformation| tranformation.update_text_buffers());
    let dataset = preprocessor.get_input_data()?;
    // prepare file loading dialog in sub-thread
    let (tx_input_path, rx_input_path) = channel::<Option<PathBuf>>();
    let (tx_output_path, rx_output_path) = channel::<PathBuf>();
    spawn_file_loader_thread(rx_input_path, tx_output_path);
    let _result = eframe::run_native(
        "Raman GUI",
        options,
        Box::new(move |_cc| {
            Box::new(RamanGuiApp {
                request_file_load: tx_input_path,
                filepath_to_load: rx_output_path,
                pipeline,
                dataset: dataset.clone(),
                initial_dataset: dataset,
                ..RamanGuiApp::new(preprocessor)
            })
        }),
    );
    Ok(())
}

fn spawn_file_loader_thread(
    rx_input_path: Receiver<Option<PathBuf>>,
    tx_output_path: Sender<PathBuf>,
) {
    std::thread::spawn(move || {
        loop {
            match rx_input_path.recv() {
                Err(_) => break,
                // no file to load requested
                Ok(requested_file) => match requested_file {
                    None => {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        continue;
                    }
                    Some(fp) => {
                        let dir = fp.parent().unwrap_or(std::path::Path::new(""));
                        if let Some(output_filepath) = rfd::FileDialog::new()
                            .set_directory(dir)
                            .add_filter("CSV", &["csv"])
                            .pick_file()
                        {
                            let _result = tx_output_path.send(output_filepath);
                        }
                    }
                },
            }
            // drain receiver so no further file open dialogs pop up
            // if load button is clicked several times
            while let Ok(_) = rx_input_path.try_recv() {}
        }
    });
}

struct RamanGuiApp {
    active_step: Option<usize>,
    add_step: Option<usize>,
    dataset_cache: HashMap<String, Dataset>,
    dataset: Dataset,
    error_messages: VecDeque<String>,
    filepath_to_load: Receiver<PathBuf>,
    force_update: bool,
    initial_dataset: Dataset,
    input_file_path: PathBuf,
    insert_transformer: InsertTransformer,
    last_dataset_hash: String,
    output_file_path: PathBuf,
    pipeline: Pipeline,
    plot_extension: Option<Box<dyn PlotExtensionGUI>>,
    plot_points: Vec<PlotPoints>,
    preprocessor: Preprocessor,
    reload_pipeline: bool,
    remove_step: Option<usize>,
    request_file_load: Sender<Option<PathBuf>>,
}

impl eframe::App for RamanGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Re-run data pipeline, if hash of pipeline configuration changed
        if let Err(e) = self.run_pipeline_on_change() {
            // error_message is reset by run_pipeline_on_change, if it runs through
            self.error_messages
                .push_front(format!("Could not run pipeline: {e}"));
        }
        // put forms for transformers into side panel
        self.left_panel(ctx);
        // put plot and other visual information in center panel
        let plot_panel_rect = self.plot_panel(ctx);

        // handle events
        ctx.input(|input_state| {
            input_state.raw.events.iter().for_each(|event| match event {
                egui::Event::Screenshot { image, .. } => {
                    self.save_screenshot(input_state, plot_panel_rect, image)
                }
                // TODO: for some reason, this freezes the app
                // egui::Event::Key {
                //     key,
                //     physical_key: _,
                //     pressed,
                //     repeat: _,
                //     modifiers,
                // } => match key {
                //     egui::Key::P => {
                //         if modifiers.ctrl && !pressed {
                //             ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
                //         }
                //     }
                //     _unhandeled_keys => (),
                // },
                _unhandeled_events => (),
            });
        });
    }
}

impl RamanGuiApp {
    fn add_transformation_form(&mut self, ui: &mut Ui, i: usize) {
        egui::ComboBox::from_label("select transformation")
            .selected_text(format!("{:?}", self.insert_transformer))
            .show_ui(ui, |ui| {
                // REGISTER: put new transformers here to make them show up in GUI
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Align,
                    "Align",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Append,
                    "Append File",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Average,
                    "Average",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::CountConversion,
                    "Count-Conversion",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Baseline,
                    "Draw Baseline",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Despike,
                    "Despiking",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Finning,
                    "Finning",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Integrate,
                    "Integrate",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Mask,
                    "Mask Points",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Normalize,
                    "Normalize",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Offset,
                    "Offset",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::RamanShift,
                    "Raman Shift",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Reshape,
                    "Reshape",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Select,
                    "Select Frames",
                );
                ui.selectable_value(
                    &mut self.insert_transformer,
                    InsertTransformer::Subtract,
                    "Subtract Frames",
                );
            });
        ui.horizontal(|ui| {
            if ui.button("Cancel").clicked() {
                self.add_step = None;
                self.insert_transformer = InsertTransformer::None;
            }
            if ui.button("OK").clicked() {
                self.insert_transformation(i);
                if self.insert_transformer != InsertTransformer::None {
                    // if inserted transform was not None (which does not get inserted into pipeline), set it active
                    self.plot_extension = self
                        .pipeline
                        .transformations
                        .get(i)
                        .unwrap()
                        // TODO maybe there is a better way than cloning dataset here? Rc?
                        .get_plot_extension(self.dataset.clone());
                    self.active_step = Some(i);
                }
                self.insert_transformer = InsertTransformer::None;
                self.add_step = None;
            }
        });
    }

    fn left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("leftpanel")
            .min_width(250.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("IO Settings");
                    ui.label("Comment Character:");
                    let mut comment = self.preprocessor.args.comment.to_string();
                    ui.text_edit_singleline(&mut comment);
                    self.preprocessor.args.comment = comment.chars().next().unwrap_or('#');
                    ui.label("Delimiter:");
                    let mut delimiter = self.preprocessor.args.delimiter.to_string();
                    ui.text_edit_singleline(&mut delimiter);
                    self.preprocessor.args.delimiter = delimiter.chars().next().unwrap_or(',');
                    ui.horizontal(|ui| {
                        ui.heading("Transformation Pipeline");
                        if ui
                            .small_button("♥")
                            .on_hover_text("Load default pipeline.")
                            .clicked()
                        {
                            self.pipeline.transformations = default_transformations();
                        }
                    });
                    let n_steps = self.pipeline.transformations.len();
                    for i in 0..n_steps {
                        if ui
                            .small_button("+")
                            .on_hover_text("Add another tranformation.")
                            .clicked()
                        {
                            self.add_step = Some(i);
                        };
                        if self.add_step.is_some() && self.add_step.unwrap() == i {
                            self.add_transformation_form(ui, i);
                        }
                        self.transformer_form(ui, i);
                    }
                    if ui
                        .small_button("+")
                        .on_hover_text("Add another tranformation.")
                        .clicked()
                    {
                        self.add_step = Some(n_steps);
                    };
                    if self.add_step.is_some() && self.add_step.unwrap() == n_steps {
                        self.add_transformation_form(ui, n_steps);
                    }
                    if let Some(step) = self.remove_step {
                        _ = self.pipeline.transformations.remove(step);
                        self.remove_step = None;
                    }
                });
            });
    }

    fn plot_panel(&mut self, ctx: &egui::Context) -> egui::Rect {
        let panel = egui::CentralPanel::default();
        let resp = panel.show(ctx, |ui| {
            self.file_panel(ui, ctx);
            let mut allow_pan_when_extension_active = true;
            if let Some(ext) = &mut self.plot_extension {
                ext.modify_ui(ui);
                allow_pan_when_extension_active =
                    // allow panning if plot extension is not active
                    // or if panning is allowed by the extension
                    ext.is_pan_allowed() || !*ext.get_is_active_reference()
            };
            Plot::new("plot")
                .height(ctx.screen_rect().height() * 0.8)
                .legend(Legend::default())
                .allow_drag(allow_pan_when_extension_active)
                .show(ui, |plot_ui| {
                    let mut colorcycle = PALETTE.iter().cycle();
                    // plot scans
                    for (i, pts) in self.plot_points.iter().enumerate() {
                        match pts {
                            // Since PlotPoints is an Enum, we have to do a little
                            // unwrapping here in order to clone the points.
                            // Points must be cloned because `Line` needs to own them.
                            PlotPoints::Owned(ps) => {
                                let pts = PlotPoints::Owned(ps.clone());
                                let color = colorcycle.next().unwrap(); // since we cycle, there will always be a next element
                                plot_ui.line(Line::new(pts).color(*color).name(i + 1));
                            }
                            _ => {}
                        }
                    }
                    // plot extension elements
                    if let Some(ext) = &mut self.plot_extension {
                        ext.modify_plot(plot_ui)
                    }
                });
            // error log
            let scroll_area = egui::ScrollArea::vertical().max_height(100.0);
            while self.error_messages.len() > 5 {
                self.error_messages.pop_back();
            }
            ui.heading("Error log");
            scroll_area.show(ui, |ui| {
                for msg in self.error_messages.iter() {
                    ui.label(msg);
                }
            });
        });
        resp.response.rect
    }

    fn file_panel(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        let mut out_text = self
            .output_file_path
            .to_str()
            .unwrap_or("non UTF-8 characters in filepath are not allowed");

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label("input file:\t");
                ui.label("output file:\t");
            });
            ui.vertical(|ui| {
                egui::TextEdit::singleline(
                    &mut self
                        .input_file_path
                        .to_str()
                        .unwrap_or("non UTF-8 characters in filepath are not allowed"),
                )
                .frame(true)
                .interactive(true)
                .desired_width(700.0)
                .show(ui);
                egui::TextEdit::singleline(&mut out_text)
                    .cursor_at_end(true)
                    .desired_width(700.0)
                    .show(ui);
            });
            ui.vertical(|ui| {
                let button_width = 70.0;
                let b = egui::Button::new("...").min_size(egui::Vec2::new(button_width, 10.));
                if ui.add(b).clicked() {
                    self.request_file_load
                        .send(Some(self.input_file_path.clone()))
                        .expect("this should not have happend, please file an issue");
                }
                let b = egui::Button::new(egui::WidgetText::from("save csv"))
                    .min_size(egui::Vec2::new(button_width, 10.));
                if ui.add(b).clicked() {
                    let dir = self
                        .output_file_path
                        .parent()
                        .unwrap_or(std::path::Path::new(""));
                    let filename = self
                        .output_file_path
                        .file_name()
                        .map(|name| name.to_str().unwrap_or_default())
                        .unwrap_or_default();
                    if let Some(filepath) = rfd::FileDialog::new()
                        .set_directory(dir)
                        .add_filter("CSV", &["csv"])
                        .set_file_name(&filename)
                        .save_file()
                    {
                        let handle = std::fs::File::create(filepath).unwrap();
                        let wrt = std::io::BufWriter::new(handle);
                        self.dataset.write(wrt).unwrap();
                    }
                }
                let b = egui::Button::new(egui::WidgetText::from("save plot"))
                    .min_size(egui::Vec2::new(button_width, 10.));
                if ui.add(b).clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
                }
            });
            ui.checkbox(&mut self.reload_pipeline, "reload pipeline?")
        });
    }

    fn run_pipeline_on_change(&mut self) -> Result<()> {
        // check if pipeline from previous run should be loaded
        if self.preprocessor.reload_pipeline {
            self.preprocessor.reload_pipeline = false;
            // TODO: Refactor to make this part DRY
            let input_string =
                crate::common::input_data_to_string(&Some(self.input_file_path.to_owned()))?;
            let prp_result =
                Preprocessor::from_yaml_header(&input_string, true).map_err(|e| eprintln!("{e}"));
            if prp_result.is_ok() {
                let mut prp = prp_result.unwrap();
                self.initial_dataset = prp.get_input_data()?;
                self.dataset = self.initial_dataset.clone();
                self.pipeline = Pipeline::from_yaml_header(&input_string)?;
                self.pipeline
                    .transformations
                    .iter_mut()
                    .for_each(|trnsf| trnsf.update_text_buffers());
                self.input_file_path = prp.args.filepath.unwrap_or(PathBuf::default());
                self.output_file_path = make_output_filepath(&self.input_file_path);
            }
        }
        // check if new file should be loaded
        if let Ok(filepath) = self.filepath_to_load.try_recv() {
            self.output_file_path = make_output_filepath(&filepath);
            self.input_file_path = filepath.clone();
            self.preprocessor.args.filepath = Some(filepath);
            // if the input file can be parsed as a result from a previous run, load the prev. run
            let input_string =
                crate::common::input_data_to_string(&Some(self.input_file_path.to_owned()))?;
            let prp_result =
                Preprocessor::from_yaml_header(&input_string, true).map_err(|e| eprintln!("{e}"));
            if prp_result.is_ok() && self.reload_pipeline {
                let mut prp = prp_result.unwrap();
                self.initial_dataset = prp.get_input_data()?;
                self.dataset = self.initial_dataset.clone();
                self.pipeline = Pipeline::from_yaml_header(&input_string).map_err(|err| {
                    eprintln!(
                        "WARNING: Unable to read pipeline from input file {:?}: {}",
                        self.input_file_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("(unreadable file name)"),
                        err
                    );
                    err
                })?;
                self.pipeline
                    .transformations
                    .iter_mut()
                    .for_each(|trnsf| trnsf.update_text_buffers());
                self.input_file_path = prp.args.filepath.unwrap_or(PathBuf::default());
                self.output_file_path = make_output_filepath(&self.input_file_path);
            } else {
                let ds = self.preprocessor.get_input_data()?;
                self.initial_dataset = ds;
                self.dataset = self.initial_dataset.clone();
            }
            self.force_update = true;
            self.dataset_cache = HashMap::new(); // reset cache
        }

        // detect change by the hash of the serialized pipeline configuration
        let pipeline_hash = {
            let conf_str: String = self
                .pipeline
                .transformations
                .iter()
                .map(|trnsf| trnsf.config_to_string().unwrap())
                .collect();
            digest(conf_str)
        };
        // if the pipeline did not change, we do nothing
        if self.last_dataset_hash == pipeline_hash && !self.force_update {
            return Ok(());
        }
        self.last_dataset_hash = pipeline_hash;
        self.force_update = false;
        self.dataset = self.initial_dataset.clone();
        // otherwise, we re-apply the transformations, reusing cache if possible
        let mut last_transformer_hash = "".to_owned();
        for (i, trnsf) in self.pipeline.transformations.iter_mut().enumerate() {
            let is_last_iter = self.active_step.map(|n| n == i).unwrap_or_default();
            if is_last_iter && !trnsf.should_plot_dataset_state_after_transformation() {
                // if the dataset is to be plotted before the transformation
                // happens, we can stop iterating here
                break;
            }
            // use hash to salt new hash, to make hashes depend on the whole
            // history of the data pipeline
            let hash = digest(trnsf.config_to_string().unwrap() + &last_transformer_hash);
            if let Some(cache) = self.dataset_cache.get(&hash) {
                self.dataset = cache.clone();
            } else {
                if let Err(err) = trnsf.apply(&mut self.dataset) {
                    self.error_messages.push_front(err.to_string());
                    break;
                }
                self.dataset_cache
                    .insert(hash.clone(), self.dataset.clone());
            }
            if is_last_iter {
                break;
            }
            last_transformer_hash = hash;
        }
        // update the plot
        if let Some(trnsf) = self
            .active_step
            .and_then(|step| self.pipeline.transformations.get(step))
        {
            self.plot_extension = trnsf.get_plot_extension(self.dataset.clone());
        } else {
            self.plot_extension = None;
        }
        self.plot_points = self.dataset.to_plot_points();

        Ok(())
    }

    fn transformer_form(&mut self, ui: &mut Ui, i: usize) {
        ui.group(|ui| {
            let trnsf = self.pipeline.transformations.get_mut(i).unwrap();
            trnsf.render_form(ui);
            ui.horizontal(|ui| {
                if ui.button("Remove").clicked() {
                    self.remove_step = Some(i);
                    self.force_update = true;
                };
                if self.active_step.is_some() && self.active_step.unwrap() == i {
                    if ui.button("OK").clicked() {
                        self.active_step = None;
                        self.force_update = true;
                        if let Some(ext) = &self.plot_extension {
                            let extension_result = ext.get_extension_result();
                            trnsf.update_from_plot_extension(extension_result);
                        }
                        self.plot_extension = None;
                    };
                } else {
                    if ui.button("Select").clicked() {
                        self.active_step = Some(i);
                        self.force_update = true;
                    }
                }
            });
        });
    }

    fn insert_transformation(&mut self, i: usize) {
        let trnsf: Box<dyn TransformerGUI> = match &self.insert_transformer {
            InsertTransformer::None => return,
            InsertTransformer::Align => Box::new(AlignTransform { cost_max_abs: 0.1 }),
            InsertTransformer::Append => Box::new(AppendTransform {
                filepath: Some(PathBuf::from("")),
                delimiter: ',',
                comment: '#',
                horizontal: false,
            }),
            InsertTransformer::Average => Box::new(AverageTransform {}),
            InsertTransformer::Baseline => Box::new(BaselineTransform {
                points: vec![],
                store: false,
            }),
            InsertTransformer::CountConversion => Box::new(CountConversionTransform::default()),
            InsertTransformer::Despike => Box::new(DespikeTransform {
                siglim: 10.0,
                flim: 10.0,
            }),
            InsertTransformer::Finning => Box::new(FinningTransform {
                threshold: 2.5,
                iterations: 4,
            }),
            InsertTransformer::Integrate => Box::new(IntegrateTransform {
                bounds: vec![],
                local_baseline: true,
            }),
            InsertTransformer::Mask => Box::new(MaskTransform { mask: vec![] }),
            InsertTransformer::Normalize => {
                let iterx = self.dataset.data.axis_iter(ndarray::Axis(1)).step_by(2);
                let itery = self
                    .dataset
                    .data
                    .axis_iter(ndarray::Axis(1))
                    .skip(1)
                    .step_by(2);
                let x_max: f64 = iterx
                    .zip(itery)
                    .map(|(xs, ys)| {
                        let idx = ys.argmax_skipnan().unwrap_or(0);
                        xs[idx]
                    })
                    .sum::<f64>()
                    / self.dataset.data.ncols() as f64
                    * 2.0;
                Box::new(NormalizeTransform {
                    xi: x_max,
                    xj: None,
                    local_baseline: false,
                    target_frames: None,
                    gui_text_buffers: NormalizeIOBuffers::default(),
                })
            }
            InsertTransformer::Offset => Box::new(OffsetTransform {
                offset: 0.0,
                percentile: false,
                target_frames: None,
                gui_text_buffers: crate::transformations::offset::OffsetIOBuffers {
                    value: 0.0.to_string(),
                },
            }),
            InsertTransformer::RamanShift => Box::new({
                let mut rst = RamanShiftTransform {
                    wavelength: 532.1,
                    refractive_index: 1.000264,
                    correction: Some(0.0),
                    ..Default::default()
                };
                rst.update_text_buffers();
                rst
            }),
            InsertTransformer::Reshape => Box::new(ReshapeTransform { rows: 1340 }),
            InsertTransformer::Select => Box::new(SelectTransform {
                frames: vec![],
                invert: true,
            }),
            InsertTransformer::Subtract => Box::new(SubtractTransform {
                direct: false,
                minuends: None,
                subtrahend: 1,
            }),
        };
        self.pipeline.transformations.insert(i, trnsf);
    }

    fn save_screenshot(
        &mut self,
        input_state: &egui::InputState,
        rect: egui::Rect,
        image: &std::sync::Arc<egui::ColorImage>,
    ) {
        let mut filepath = self.output_file_path.to_owned();
        filepath.set_extension("png");
        let dir = filepath.parent().unwrap_or(std::path::Path::new(""));
        let filename = filepath
            .file_name()
            .map(|name| name.to_str().unwrap_or_default())
            .unwrap_or_default();
        if let Some(filepath) = rfd::FileDialog::new()
            .set_directory(dir)
            .add_filter("PNG", &["png"])
            .set_file_name(&filename)
            .save_file()
        {
            let pixels_per_point = input_state.pixels_per_point();
            let region = egui::Rect::from_two_pos(rect.left_top(), rect.right_bottom());
            let top_left_corner = image.region(&region, Some(pixels_per_point));
            let _ = image::save_buffer(
                filepath,
                top_left_corner.to_owned().as_raw(),
                top_left_corner.size[0] as u32,
                top_left_corner.size[1] as u32,
                ColorType::Rgba8,
            )
            .map_err(|e| eprintln!("Error while saving screenshot: {e}"));
        }
    }
}

fn make_output_filepath(filepath: &PathBuf) -> PathBuf {
    let mut filepath = filepath.to_owned();
    filepath.set_extension("");
    let mut fp = filepath.to_str().unwrap().to_owned();
    fp.extend("_processed.csv".chars());
    PathBuf::from(fp)
}

impl RamanGuiApp {
    fn new(preprocessor: Preprocessor) -> Self {
        let ds = Dataset::default();
        let pts = ds.to_plot_points();
        let output_file_path = match &preprocessor.args.filepath {
            Some(fp) => make_output_filepath(fp),
            None => PathBuf::from("processed.csv"),
        };
        let input_file_path = preprocessor
            .args
            .filepath
            .clone()
            .unwrap_or(PathBuf::default());
        let (tx_input_file, _) = channel::<Option<PathBuf>>();
        let (_, rx_output_path) = channel::<PathBuf>();

        Self {
            active_step: None,
            add_step: None,
            dataset_cache: HashMap::new(),
            dataset: ds.clone(),
            error_messages: VecDeque::with_capacity(10),
            filepath_to_load: rx_output_path,
            force_update: true,
            initial_dataset: ds,
            input_file_path,
            insert_transformer: InsertTransformer::None,
            last_dataset_hash: "".to_owned(),
            output_file_path,
            pipeline: Pipeline {
                transformations: vec![],
            },
            plot_extension: Some(Box::new(SplineExtensionGUI::new(vec![]))),
            plot_points: pts,
            preprocessor,
            reload_pipeline: true,
            remove_step: None,
            request_file_load: tx_input_file,
        }
    }
}

enum FloatInput<'a> {
    Number(&'a mut f64),
    OptionalNumber(&'a mut Option<f64>),
}

fn draw_fallable_text_edit(ui: &mut Ui, input: &mut String, num: FloatInput) {
    let parsing_result = input.parse::<f64>();
    let text_edit = match num {
        FloatInput::OptionalNumber(opt_num) => match (parsing_result, input.as_str()) {
            (Err(_), "None") => {
                *opt_num = None;
                egui::TextEdit::singleline(input)
            }
            (Ok(x), _) => {
                *opt_num = Some(x);
                egui::TextEdit::singleline(input)
            }
            // indicator input error in red, no other error handling
            _ => egui::TextEdit::singleline(input).text_color(Color32::from_rgb(255, 0, 0)),
        },
        FloatInput::Number(num) => match parsing_result {
            Ok(x) => {
                *num = x;
                egui::TextEdit::singleline(input)
            }
            Err(_) => egui::TextEdit::singleline(input).text_color(Color32::from_rgb(255, 0, 0)),
        },
    };
    text_edit.show(ui);
}

#[derive(Debug, PartialEq)]
enum InsertTransformer {
    // REGISTER: put new transformers here to make them show up in GUI
    None,
    Align,
    Append,
    Average,
    Baseline,
    CountConversion,
    Despike,
    Finning,
    Integrate,
    Mask,
    Normalize,
    Offset,
    RamanShift,
    Reshape,
    Select,
    Subtract,
}

pub trait TransformerGUI: Transformer {
    fn render_form(&mut self, ui: &mut Ui) -> ();
    fn get_plot_extension(&self, _ds: Dataset) -> Option<Box<dyn PlotExtensionGUI>> {
        None
    }
    #[allow(unused)] // only unused in default implementation
    fn update_from_plot_extension(&mut self, ext: PlotExtensionResult) -> () {}
    fn update_text_buffers(&mut self) -> () {}
    fn should_plot_dataset_state_after_transformation(&self) -> bool {
        true
    }
}

impl TransformerGUI for AlignTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Align");
        ui.add(Slider::new(&mut self.cost_max_abs, 0.01..=1.0).text("tuning parameter"));
    }
}

impl TransformerGUI for AppendTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Append File");
        let mut fp = match &self.filepath {
            None => "".to_owned(),
            Some(fp) => format!("{}", fp.display()),
        };
        ui.text_edit_singleline(&mut fp);
        self.filepath = Some(PathBuf::from(fp));
        ui.checkbox(&mut self.horizontal, "as new rows?");
    }
}

impl TransformerGUI for AverageTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Average");
    }
}

impl TransformerGUI for BaselineTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Draw Baseline");
        ui.checkbox(&mut self.store, "Store baseline separately");
    }
    fn get_plot_extension(&self, _ds: Dataset) -> Option<Box<dyn PlotExtensionGUI>> {
        let ext = SplineExtensionGUI::new(self.points.iter().map(|pt| [pt.a, pt.b]).collect());
        Some(Box::new(ext))
    }
    fn update_from_plot_extension(&mut self, ext: PlotExtensionResult) -> () {
        match ext {
            PlotExtensionResult::Spline(points) => self.points = points,
            _ => {
                panic!("Baseline transformer got wrong plot extension result. This should not have happend, please file an issue.")
            }
        }
    }
    fn should_plot_dataset_state_after_transformation(&self) -> bool {
        false
    }
}

impl TransformerGUI for CountConversionTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Count-Conversion");
        ui.label("Exposure in seconds");
        draw_fallable_text_edit(
            ui,
            &mut self.gui_text_buffers.exposure,
            FloatInput::Number(&mut self.exposure),
        );
        ui.label("Conversion Factor");
        draw_fallable_text_edit(
            ui,
            &mut self.gui_text_buffers.conversion_factor,
            FloatInput::Number(&mut self.conversion_factor),
        );
    }
    fn update_text_buffers(&mut self) -> () {
        self.gui_text_buffers.exposure = self.exposure.to_string();
        self.gui_text_buffers.conversion_factor = self.conversion_factor.to_string();
    }
}

impl TransformerGUI for DespikeTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Despiking");
        ui.add(Slider::new(&mut self.siglim, 0.0..=100.0).text("sigma limit"));
        ui.add(Slider::new(&mut self.flim, 0.0..=100.0).text("flim"));
    }
}

impl TransformerGUI for FinningTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Finning");
        ui.label("threshold");
        // let drag = DragValue::new(&mut self.threshold)
        //     .clamp_range(0.0..=6.0)
        //     .fixed_decimals(2)
        //     .speed(0.01);
        // ui.add(drag);
        ui.add(Slider::new(&mut self.threshold, 1.0..=5.0));
        let mut iterations: String = self.iterations.to_string();
        ui.label("Number of iterations:");
        ui.text_edit_singleline(&mut iterations);
        if let Ok(niter) = iterations.parse::<usize>() {
            self.iterations = niter;
        }
    }
}

impl TransformerGUI for IntegrateTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Integration");
        ui.checkbox(&mut self.local_baseline, "Subtract local baseline?");
        for (i, Pair { a: left, b: right }) in self.bounds.iter_mut().enumerate() {
            ui.label(format!("Integration window {}", i + 1));
            ui.horizontal(|ui| {
                ui.label("Left bound:");
                ui.add(egui::DragValue::new(left));
            });
            ui.horizontal(|ui| {
                ui.label("Right bound:");
                ui.add(egui::DragValue::new(right));
            });
        }
    }

    fn get_plot_extension(&self, ds: Dataset) -> Option<Box<dyn PlotExtensionGUI>> {
        Some(Box::new(IntegrateExtensionGUI {
            dataset: ds,
            bounds: self.bounds.to_owned(),
            ..Default::default()
        }))
    }

    fn update_from_plot_extension(&mut self, ext: PlotExtensionResult) -> () {
        match ext {
            PlotExtensionResult::Integrate(bounds) => self.bounds = bounds,
            _ => panic!("Integrate transformer got wrong plot extension result. This should not have happend, please file an issue."),
        }
    }

    fn update_text_buffers(&mut self) -> () {}

    fn should_plot_dataset_state_after_transformation(&self) -> bool {
        false
    }
}

impl TransformerGUI for MaskTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Mask Points");
    }
    fn get_plot_extension(&self, ds: Dataset) -> Option<Box<dyn PlotExtensionGUI>> {
        let ext = MaskExtensionGUI {
            ..MaskExtensionGUI::from_mask(&self.mask, ds)
        };
        Some(Box::new(ext))
    }
    fn update_from_plot_extension(&mut self, result: PlotExtensionResult) -> () {
        match result {
            PlotExtensionResult::Mask(mask) => self.mask = mask,
            _ => panic!("Baseline transformer got wrong plot extension result. This should not have happend, please file an issue."),
        }
    }
    fn should_plot_dataset_state_after_transformation(&self) -> bool {
        false
    }
}

impl TransformerGUI for NormalizeTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Normalize");
        ui.label("window start");
        draw_fallable_text_edit(
            ui,
            &mut self.gui_text_buffers.xi,
            FloatInput::Number(&mut self.xi),
        );
        ui.label("window end");
        draw_fallable_text_edit(
            ui,
            &mut self.gui_text_buffers.xj,
            FloatInput::OptionalNumber(&mut self.xj),
        );
    }
    fn update_text_buffers(&mut self) {
        self.gui_text_buffers.xi = self.xi.to_string();
        self.gui_text_buffers.xj = match self.xj {
            None => "None".to_string(),
            Some(x) => x.to_string(),
        }
    }
    fn get_plot_extension(&self, _ds: Dataset) -> Option<Box<dyn PlotExtensionGUI>> {
        Some(Box::new(NormalizeExtensionGUI {
            xi: self.xi,
            xj: self.xj,
            is_active: false,
        }))
    }
    fn update_from_plot_extension(&mut self, ext: PlotExtensionResult) -> () {
        match ext {
            PlotExtensionResult::Normalize((xi, xj)) => {
                self.xi = xi;
                self.xj = xj;
                self.update_text_buffers();
            }
            _ => {}
        }
    }
}

impl TransformerGUI for OffsetTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Correct Offset");
        ui.checkbox(&mut self.percentile, "Value as percentile?");
        ui.label("Offset: ");
        if self.percentile {
            ui.add(Slider::new(&mut self.offset, 0.0..=1.0));
        } else {
            draw_fallable_text_edit(
                ui,
                &mut self.gui_text_buffers.value,
                FloatInput::Number(&mut self.offset),
            );
        }
        let mut selection: String = self
            .target_frames
            .as_ref()
            .map(|frames| frames.iter().map(|n| format!("{} ", n)).collect())
            .unwrap_or_default();
        ui.label("Select frames to apply offset to: ");
        ui.text_edit_singleline(&mut selection);
        self.target_frames = if selection.is_empty() {
            None
        } else {
            Some(
                selection
                    .split_whitespace()
                    .filter_map(|str| str.parse::<usize>().ok())
                    .collect(),
            )
        };
    }
    fn update_text_buffers(&mut self) -> () {
        self.gui_text_buffers.value = self.offset.to_string();
    }
}

impl TransformerGUI for RamanShiftTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Raman Shift");
        ui.label("Laser Wavelength");
        draw_fallable_text_edit(
            ui,
            &mut self.gui_text_buffers.wavelength,
            FloatInput::Number(&mut self.wavelength),
        );
        ui.label("Correction (Offset)");
        let mut correction = self.correction.unwrap_or(0.0);
        draw_fallable_text_edit(
            ui,
            &mut self.gui_text_buffers.correction,
            FloatInput::Number(&mut correction),
        );
        self.correction = Some(correction);
        ui.label("Refractive Index of Air");
        draw_fallable_text_edit(
            ui,
            &mut self.gui_text_buffers.refractive_index,
            FloatInput::Number(&mut self.refractive_index),
        );
    }
    fn update_text_buffers(&mut self) -> () {
        self.gui_text_buffers.wavelength = self.wavelength.to_string();
        self.gui_text_buffers.refractive_index = self.refractive_index.to_string();
        if let Some(c) = self.correction {
            self.gui_text_buffers.correction = c.to_string();
        }
    }
}

impl TransformerGUI for ReshapeTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Reshape");
        ui.add(Slider::new(&mut self.rows, 1..=1340).text("rows"));
    }
    fn get_plot_extension(&self, _ds: Dataset) -> Option<Box<dyn PlotExtensionGUI>> {
        None
    }
}

impl TransformerGUI for SelectTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Select Frames");
        ui.checkbox(&mut self.invert, "invert selection");
        let mut selection: String = self.frames.iter().map(|n| format!("{} ", n)).collect();
        ui.text_edit_singleline(&mut selection);
        self.frames = selection
            .split_whitespace()
            .filter_map(|str| str.parse::<usize>().ok())
            .collect();
    }
    fn should_plot_dataset_state_after_transformation(&self) -> bool {
        false
    }
}

impl TransformerGUI for SubtractTransform {
    fn render_form(&mut self, ui: &mut Ui) -> () {
        ui.heading("Subtract Frames");
        ui.checkbox(&mut self.direct, "direct subtraction of y-values");
        let mut minuends: String = match &self.minuends {
            None => "".to_owned(),
            Some(ms) => ms.iter().map(|n| format!("{} ", n)).collect(),
        };
        ui.text_edit_singleline(&mut minuends);
        let minuends: Vec<usize> = minuends
            .split_whitespace()
            .filter_map(|str| str.parse::<usize>().ok())
            .collect();
        if minuends.is_empty() {
            self.minuends = None
        } else {
            self.minuends = Some(minuends)
        }
        let mut subtrahend: String = self.subtrahend.to_string();
        ui.text_edit_singleline(&mut subtrahend);
        if let Ok(s) = subtrahend.parse::<usize>() {
            self.subtrahend = s;
        }
    }
}
