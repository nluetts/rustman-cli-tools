use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::common::{Dataset, Pair};
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use eframe::egui;
use egui::{Color32, Ui};
use egui_plot::{Legend, Line, Plot, PlotPoints, PlotUi, Points, VLine};
use ndarray::Axis;
use serde::{Deserialize, Serialize};
use splines::{self, Key, Spline};

pub static PALETTE: [Color32; 8] = [
    Color32::from_rgb(102, 194, 165),
    Color32::from_rgb(252, 141, 98),
    Color32::from_rgb(141, 160, 203),
    Color32::from_rgb(231, 138, 195),
    Color32::from_rgb(166, 216, 84),
    Color32::from_rgb(255, 217, 47),
    Color32::from_rgb(229, 196, 148),
    Color32::from_rgb(179, 179, 179),
];

// ---- PlotTransform ---------------------------------------------------------

#[derive(Debug, Default, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct PlotTransform {
    #[clap(short, long, help = "Choose linewidth of plotted scans.")]
    pub line_width: Option<f32>,
    #[clap(short, long, help = "Choose x-limits of the plot.")]
    pub x_lim: Option<Pair<f64>>,
    #[clap(short, long, help = "Choose y-limits of the plot.")]
    pub y_lim: Option<Pair<f64>>,
    #[clap(
        short,
        long,
        action,
        help = "if flag is set, plot intensity versus pixels"
    )]
    pub pixels: bool,
    #[serde(skip)]
    #[clap(skip)]
    pub extensions: Vec<Arc<Mutex<dyn PlotExtension>>>,
}

impl Transformer for PlotTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let options = eframe::NativeOptions {
            // initial_window_size: Some(egui::vec2(800.0, 500.0)),
            ..Default::default()
        };
        let ds_arcmutex = match self.pixels {
            false => Arc::new(Mutex::new(dataset.clone())),
            true => {
                let mut ds = dataset.clone();
                // replace spectral axis by pixel number
                ds.data
                    .axis_iter_mut(Axis(1))
                    .step_by(2)
                    .for_each(|mut col| {
                        col.iter_mut().enumerate().for_each(|(i, x)| *x = i as f64)
                    });
                Arc::new(Mutex::new(ds))
            }
        };
        let pw = PlotWindow::new(
            ds_arcmutex.clone(),
            self.line_width,
            self.extensions.clone(),
            Arc::new(Mutex::new(String::new())),
            self.x_lim,
            self.y_lim,
        );
        eframe::run_native("Dataset Plot", options, Box::new(|_cc| Box::new(pw)));
        dataset.data = ds_arcmutex
            .lock()
            .expect("Unable to aquire lock to read data from plot.")
            .data
            .clone();
        Ok(())
    }
}

// ---- PlotWindow ------------------------------------------------------------

pub struct PlotWindow {
    dataset_arcmutex: Arc<Mutex<Dataset>>,
    extensions: Vec<Arc<Mutex<dyn PlotExtension>>>,
    line_width: f32,
    info: Arc<Mutex<String>>,
    x_lim: Option<Pair<f64>>,
    y_lim: Option<Pair<f64>>,
}

impl PlotWindow {
    pub fn new(
        ds: Arc<Mutex<Dataset>>,
        lw: Option<f32>,
        exts: Vec<Arc<Mutex<dyn PlotExtension>>>,
        info: Arc<Mutex<String>>,
        x_lim: Option<Pair<f64>>,
        y_lim: Option<Pair<f64>>,
    ) -> Self {
        let lw = lw.unwrap_or(1.0_f32);
        PlotWindow {
            dataset_arcmutex: ds,
            line_width: lw,
            extensions: exts,
            info,
            x_lim,
            y_lim,
        }
    }
}

impl eframe::App for PlotWindow {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.viewport().close_requested()) {
            // apply modifications to dataset when plot window is closed.
            let mut ds = self
                .dataset_arcmutex
                .lock()
                .expect("Unable to get lock to modify dataset.");
            for ext in self.extensions.iter_mut() {
                ext.lock().unwrap().on_close(&mut ds)
            }
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            // TODO: This forces plot window to constantly repaint, to allow 'watching'
            // a file. Should better be done with a callback.
            ctx.request_repaint();
            for ext in self.extensions.iter_mut() {
                ext.lock().unwrap().modify_ui(ui)
            }
            if self.info.lock().unwrap().as_str() != "" {
                ui.colored_label(
                    Color32::from_rgb(255, 0, 0),
                    self.info.lock().unwrap().as_str(),
                );
            }
            let mut plot = Plot::new("Scans").legend(Legend::default());
            if let Some(x_lim) = self.x_lim {
                plot = plot.include_x(x_lim.a);
                plot = plot.include_x(x_lim.b);
            }
            if let Some(y_lim) = self.y_lim {
                plot = plot.include_y(y_lim.a);
                plot = plot.include_y(y_lim.b);
            }
            plot.show(ui, |plot_ui| {
                let mut colorcycle = PALETTE.iter().cycle();
                // plot scans
                let ds = self
                    .dataset_arcmutex
                    .lock()
                    .expect("Unable to get lock for dataset.");
                for j in (0..ds.data.ncols()).step_by(2) {
                    let points: PlotPoints = (0..ds.data.nrows())
                        .map(|i| {
                            let x = ds.data[[i, j]];
                            let y = ds.data[[i, j + 1]];
                            [x, y]
                        })
                        .collect();
                    let color = colorcycle.next().unwrap(); // since we cycle, there will always be a next element
                    plot_ui.line(
                        Line::new(points)
                            .width(self.line_width)
                            .color(*color)
                            .name(j / 2 + 1),
                    );
                }
                // plot extension elements
                for ext in self.extensions.iter_mut() {
                    ext.lock().unwrap().modify_plot(plot_ui)
                }
            });
        });
    }
}

// ---- PlotExtension ---------------------------------------------------------

/// PlotExtension add functionality to a PlotWindow app
pub trait PlotExtension: std::fmt::Debug {
    /// Add widgets to the ui of the window, e.g. extra buttons.
    fn modify_ui(&mut self, _: &mut Ui);
    /// Add elements to the plot.
    fn modify_plot(&mut self, _: &mut PlotUi);
    /// Modify dataset of plot transform.
    fn on_close(&mut self, _: &mut MutexGuard<Dataset>);
}

// ---- VLineExtension --------------------------------------------------------

/// Draw vertical lines in plot.
#[derive(Debug, Default)]
struct VLineExtension {
    add_line_mode_enabled: bool,
    vlines: Vec<egui_plot::PlotPoint>,
}

impl PlotExtension for VLineExtension {
    fn on_close(&mut self, _: &mut MutexGuard<Dataset>) {}
    fn modify_plot(&mut self, plot_ui: &mut PlotUi) {
        if plot_ui.response().clicked() && self.add_line_mode_enabled {
            match plot_ui.pointer_coordinate() {
                None => {}
                Some(point) => {
                    self.vlines.push(point);
                }
            }
        }
        for pt in self.vlines.iter() {
            plot_ui.vline(VLine::new(pt.x));
        }
    }
    fn modify_ui(&mut self, ui: &mut Ui) {
        ui.toggle_value(&mut self.add_line_mode_enabled, "Add Line");
    }
}

// ---- SplineExtension -------------------------------------------------------

/// Draw spline baseline that is subtracted from all scans.
#[derive(Debug)]
pub struct SplineExtension {
    pub add_point_mode_enabled: bool,
    pub points: Vec<[f64; 2]>,
    pub sender: Sender<(Vec<[f64; 2]>, Spline<f64, f64>)>,
    pub spline: splines::Spline<f64, f64>,
}

impl PlotExtension for SplineExtension {
    fn on_close(&mut self, _ds: &mut MutexGuard<Dataset>) {
        self.sender
            .send((self.points.clone(), self.spline.clone()))
            .unwrap();
    }
    fn modify_ui(&mut self, ui: &mut Ui) {
        ui.toggle_value(&mut self.add_point_mode_enabled, "Add/Remove Points");
    }
    fn modify_plot(&mut self, plot_ui: &mut PlotUi) {
        if self.add_point_mode_enabled {
            if plot_ui.response().clicked() {
                self.add_point(plot_ui);
            } else if plot_ui.response().secondary_clicked() {
                self.remove_point(plot_ui);
            }
        }
        self.draw_spline(plot_ui);
        plot_ui.points(Points::new(self.points.clone()).radius(5.));
    }
}

impl SplineExtension {
    pub fn new(
        points: Vec<[f64; 2]>,
        sender: Sender<(Vec<[f64; 2]>, Spline<f64, f64>)>,
    ) -> SplineExtension {
        let mut spl = Self {
            points,
            sender,
            add_point_mode_enabled: false,
            spline: Spline::from_vec(vec![]),
        };
        spl.update_spline();
        spl
    }
    fn add_point(&mut self, plot_ui: &mut PlotUi) {
        if let Some(point) = plot_ui.pointer_coordinate() {
            self.points.push([point.x, point.y])
        }
        self.points
            .sort_by(|pt1, pt2| pt1[0].partial_cmp(&pt2[0]).unwrap());
        self.update_spline();
    }
    fn remove_point(&mut self, plot_ui: &mut PlotUi) {
        if let Some(point) = plot_ui.pointer_coordinate() {
            let span = {
                let [xmin, ymin] = plot_ui.plot_bounds().min();
                let [xmax, ymax] = plot_ui.plot_bounds().max();
                (xmax - xmin, ymax - ymin)
            };
            if let Some(index) = self.nearest_point_index(point, span) {
                self.points.remove(index);
            }
        }
        self.update_spline();
    }
    fn update_spline(&mut self) {
        let mut keys = vec![];
        let n_pts = self.points.len();
        if n_pts < 2 {
            return;
        }
        for i in 0..n_pts {
            if i == 0 || i == n_pts - 2 {
                keys.push(Key::new(
                    self.points[i][0],
                    self.points[i][1],
                    splines::Interpolation::Linear,
                ));
            } else {
                keys.push(Key::new(
                    self.points[i][0],
                    self.points[i][1],
                    splines::Interpolation::CatmullRom,
                ));
            }
        }
        self.spline = splines::Spline::from_vec(keys)
    }
    fn draw_spline(&mut self, plot_ui: &mut PlotUi) {
        let xmin = plot_ui.plot_bounds().min()[0];
        let xmax = plot_ui.plot_bounds().max()[0];
        let step = (xmax - xmin) / 1000.;
        let mut x = xmin;
        let mut points: Vec<[f64; 2]> = vec![];
        while x <= xmax {
            if let Some(y) = self.spline.sample(x) {
                points.push([x, y]);
            }
            x += step;
        }
        plot_ui.line(Line::new(points))
    }
    fn nearest_point_index(
        &mut self,
        point: egui_plot::PlotPoint,
        // span is the x- and y-span of the plot, used to normalize distances
        span: (f64, f64),
    ) -> Option<usize> {
        let mut distances = self
            .points
            .iter()
            .enumerate()
            .map(|(i, pt)| {
                (
                    i,
                    f64::sqrt(
                        ((pt[0] - point.x) / span.0).powi(2) + ((pt[1] - point.y) / span.1).powi(2),
                    ),
                )
            })
            .collect::<Vec<(usize, f64)>>();
        distances.sort_by(|elm1, elm2| {
            elm1.1
                .partial_cmp(&elm2.1)
                .unwrap_or(std::cmp::Ordering::Less)
        });
        distances.first().map(|(index, _)| *index)
    }
}
