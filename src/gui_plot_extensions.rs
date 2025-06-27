use std::ops::Index;

use egui::{Color32, Ui};
use egui_plot::{Line, PlotPoint, PlotPoints, PlotUi, Points};
use ndarray::Axis;
use noisy_float::{prelude::Float, types::N64};
use splines::{Key, Spline};

use crate::{
    common::{Dataset, Pair},
    utils::nearest_index,
};

#[derive(Debug)]
pub enum PlotExtensionResult {
    Integrate(Vec<Pair<f64>>),
    Mask(Vec<Pair<usize>>),
    Normalize((f64, Option<f64>)),
    Spline(Vec<Pair<f64>>),
}

pub trait PlotExtensionGUI {
    fn modify_ui(&mut self, ui: &mut Ui) {
        let extension_toggle_label = self.extension_toggle_label();
        let is_enabled_ref = self.get_is_active_reference();
        ui.toggle_value(is_enabled_ref, extension_toggle_label);
    }
    fn modify_plot(&mut self, _plot_ui: &mut PlotUi) {}
    fn get_extension_result(&self) -> PlotExtensionResult;
    fn get_is_active_reference(&mut self) -> &mut bool;
    fn extension_toggle_label(&self) -> String;
    fn is_pan_allowed(&self) -> bool {
        true
    }
}

impl PlotExtensionGUI for MaskExtensionGUI {
    fn modify_plot(&mut self, plot_ui: &mut PlotUi) {
        if self.mask_mode_enabled {
            if plot_ui.response().clicked() {
                self.add_point(plot_ui);
            }
            if plot_ui.response().secondary_clicked() {
                self.remove_point(plot_ui);
            }
        }
        let masked_points: Vec<[f64; 2]> = self
            .points
            .iter()
            .map(|pt| {
                [
                    self.dataset.data[[pt.b - 1, 2 * pt.a - 2]],
                    self.dataset.data[[pt.b - 1, 2 * pt.a - 1]],
                ]
            })
            .collect();
        plot_ui.points(Points::new(masked_points).radius(5.));
    }

    fn get_extension_result(&self) -> PlotExtensionResult {
        PlotExtensionResult::Mask(self.points.to_owned())
    }
    fn extension_toggle_label(&self) -> String {
        "Add/Remove Points".to_owned()
    }
    fn get_is_active_reference(&mut self) -> &mut bool {
        &mut self.mask_mode_enabled
    }
}

impl PlotExtensionGUI for SplineExtensionGUI {
    fn modify_plot(&mut self, plot_ui: &mut PlotUi) {
        self.draw_spline(plot_ui);
        if self.add_point_mode_enabled {
            if plot_ui.response().clicked() {
                self.add_point(plot_ui);
            } else if plot_ui.response().secondary_clicked() {
                self.remove_point(plot_ui);
            }
        }
    }
    fn get_extension_result(&self) -> PlotExtensionResult {
        let pts = self
            .points
            .iter()
            .map(|[a, b]| Pair {
                a: a.to_owned(),
                b: b.to_owned(),
            })
            .collect();
        PlotExtensionResult::Spline(pts)
    }

    fn get_is_active_reference(&mut self) -> &mut bool {
        &mut self.add_point_mode_enabled
    }

    fn extension_toggle_label(&self) -> String {
        "Add/Remove Points".to_owned()
    }
}

/// Draw spline baseline that is subtracted from all scans.
#[derive(Debug)]
pub struct SplineExtensionGUI {
    pub add_point_mode_enabled: bool,
    pub points: Vec<[f64; 2]>,
    pub spline: splines::Spline<f64, f64>,
}

impl SplineExtensionGUI {
    pub fn new(points: Vec<[f64; 2]>) -> SplineExtensionGUI {
        let mut spl = Self {
            points,
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
        plot_ui.line(Line::new(points));
        plot_ui.points(Points::new(self.points.clone()).radius(5.));
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

// ---- IntegrateExtension --------------------------------------------------

#[derive(Debug)]
pub struct IntegrateExtensionGUI {
    pub add_bound_mode: bool,
    pub bounds: Vec<Pair<f64>>,
    pub dataset: Dataset,
    pub local: Vec<bool>,
    pub new_bound: Option<Pair<f64>>,
}

impl Default for IntegrateExtensionGUI {
    fn default() -> Self {
        Self {
            add_bound_mode: false,
            bounds: vec![],
            dataset: Dataset::default(),
            local: vec![],
            new_bound: None,
        }
    }
}

impl IntegrateExtensionGUI {
    /// return two points that form a straight line that fall closest to the
    /// frames in the dataset
    fn get_closest_straight_line(&self, x0: f64, x1: f64) -> Vec<Option<[PlotPoint; 2]>> {
        let iter_frames = self // iterator over x and y columns of dataset
            .dataset
            .data
            .columns()
            .into_iter()
            .step_by(2)
            .zip(self.dataset.data.columns().into_iter().skip(1).step_by(2));
        let mut windows = vec![];
        for (xs, ys) in iter_frames {
            let x0i = nearest_index(&xs, x0);
            let x1i = nearest_index(&xs, x1);
            if x0i.is_none() || x1i.is_none() {
                windows.push(None);
            } else {
                let i = x0i.unwrap();
                let j = x1i.unwrap();
                windows.push(Some([
                    PlotPoint::new(xs[i], ys[i]),
                    PlotPoint::new(xs[j], ys[j]),
                ]));
            }
        }
        windows
    }
    fn remove_point(&mut self, plot_ui: &mut PlotUi) {
        if let Some(point) = plot_ui.pointer_coordinate() {
            let xspan = {
                let [xmin, _] = plot_ui.plot_bounds().min();
                let [xmax, _] = plot_ui.plot_bounds().max();
                xmax - xmin
            };
            if let Some(index) = self.nearest_bound_index(point, xspan) {
                self.bounds.remove(index);
            }
        }
    }
    fn nearest_bound_index(
        &mut self,
        point: egui_plot::PlotPoint,
        // xspan is the x-span of the plot, used to normalize distances
        xspan: f64,
    ) -> Option<usize> {
        let mut distances = self
            .bounds
            .iter()
            .enumerate()
            .map(|(i, bnd)| {
                (
                    i,
                    f64::sqrt(
                        ((bnd.a - point.x) / xspan).powi(2) + ((bnd.b - point.x) / xspan).powi(2),
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

impl PlotExtensionGUI for IntegrateExtensionGUI {
    fn get_extension_result(&self) -> PlotExtensionResult {
        PlotExtensionResult::Integrate(self.bounds.to_owned())
    }

    fn get_is_active_reference(&mut self) -> &mut bool {
        &mut self.add_bound_mode
    }

    fn extension_toggle_label(&self) -> String {
        "Add/Remove Bound".to_owned()
    }

    fn is_pan_allowed(&self) -> bool {
        false
    }

    fn modify_ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let extension_toggle_label = self.extension_toggle_label();
            ui.toggle_value(&mut self.add_bound_mode, extension_toggle_label);
            if !self.add_bound_mode {
                self.new_bound = None;
            }
            if ui.button("Accept Bound").clicked() {
                if let Some(bnd) = &self.new_bound {
                    self.bounds.push(bnd.clone());
                    self.new_bound = None;
                }
            }
            ui.label(format!("Bounds defined: {}", self.bounds.len()));
        });
    }

    fn modify_plot(&mut self, plot_ui: &mut PlotUi) {
        if self.add_bound_mode && plot_ui.response().hovered() {
            // Note the block { plot_ui.ctx()... } in this if-let binding.
            // It is required because input() otherwise keeps the context locked
            // and we do not get updated pointer origins.
            let press_origin = plot_ui.ctx().input(|i| i.pointer.press_origin());
            if let Some(origin) = press_origin {
                let origin_position = plot_ui.plot_from_screen(origin);
                let cursor_position = plot_ui.pointer_coordinate();
                // start drawing a new bound
                if self.new_bound.is_none() && cursor_position.is_some() {
                    let pos = cursor_position.unwrap();
                    self.new_bound = Some(Pair {
                        a: origin_position.x as f64,
                        b: pos.x,
                    });
                // update while drawing a new bound
                } else if self.new_bound.is_some() && cursor_position.is_some() {
                    let pos = cursor_position.unwrap();
                    self.new_bound.iter_mut().for_each(|bnd| {
                        bnd.a = origin_position.x;
                        bnd.b = pos.x;
                    });
                }
            } else if plot_ui.response().secondary_clicked() {
                self.remove_point(plot_ui);
            }
        }
        for bnd in self.new_bound.iter().chain(self.bounds.iter()) {
            for line in self.get_closest_straight_line(bnd.a, bnd.b) {
                line.map(|pts| {
                    let red = Color32::from_rgb(255, 0, 0);
                    let pts_to_draw =
                        PlotPoints::from(pts.iter().map(|pt| [pt.x, pt.y]).collect::<Vec<_>>());
                    plot_ui.points(Points::new(pts_to_draw));
                    let pts_to_draw =
                        PlotPoints::from(pts.iter().map(|pt| [pt.x, pt.y]).collect::<Vec<_>>());
                    plot_ui.line(Line::new(pts_to_draw).width(4.0).color(red));
                });
            }
        }
    }
}

// ---- MaskExtension -------------------------------------------------------

#[derive(Debug)]
pub struct MaskedPoint {
    pub frame: usize,
    pub pixel: usize,
}

/// Draw spline baseline that is subtracted from all scans.
#[derive(Debug)]
pub struct MaskExtensionGUI {
    pub mask_mode_enabled: bool,
    pub points: Vec<Pair<usize>>,
    pub dataset: Dataset,
}

impl MaskExtensionGUI {
    pub fn from_mask(mask: &[Pair<usize>], dataset: Dataset) -> Self {
        let points = mask.to_owned();
        Self {
            points,
            dataset,
            ..Default::default()
        }
    }
    fn add_point(&mut self, plot_ui: &mut PlotUi) {
        if let Some(masked_point) = self.neareast_index_to_cursor(&plot_ui) {
            self.points.push(masked_point);
        }
    }
    fn remove_point(&mut self, plot_ui: &mut PlotUi) {
        if let Some(Pair {
            a: frame_number,
            b: pixel_number,
        }) = self.neareast_index_to_cursor(&plot_ui)
        {
            if let Some(idx) = self
                .points
                .iter()
                .position(|p| p.a == frame_number && p.b == pixel_number)
            {
                self.points.remove(idx);
            }
        }
    }
    fn neareast_index_to_cursor(&mut self, plot_ui: &PlotUi) -> Option<Pair<usize>> {
        let mut previous_nearest = (
            1,                // frame number
            0,                // pixel index
            N64::max_value(), // distance
        );
        // iterate over wavenumber axis, spectral axis pairs (frames)
        for frame_number in 1..=self.dataset.data.ncols() / 2 {
            let x = self.dataset.data.column(2 * (frame_number - 1));
            let y = self.dataset.data.column(2 * frame_number - 1);
            // calcualte the smallest distance of all datapoints in the frame
            // to the mouse cursor
            for (pixel_idx, (xi, yi)) in x.iter().zip(y.iter()).enumerate() {
                match distance_cursor(&plot_ui, *xi, *yi) {
                    Some(distance) if distance < previous_nearest.2 => {
                        previous_nearest.0 = frame_number;
                        previous_nearest.1 = pixel_idx;
                        previous_nearest.2 = distance;
                    }
                    _ => continue,
                }
            }
        }
        Some(Pair {
            a: previous_nearest.0,
            b: previous_nearest.1 + 1,
        })
    }
}

impl Default for MaskExtensionGUI {
    fn default() -> Self {
        Self {
            mask_mode_enabled: true,
            points: vec![],
            dataset: Dataset::default(),
        }
    }
}

fn distance_cursor(plot_ui: &PlotUi, xi: f64, yi: f64) -> Option<N64> {
    if let Some(point) = plot_ui.pointer_coordinate() {
        let span = {
            let [xmin, ymin] = plot_ui.plot_bounds().min();
            let [xmax, ymax] = plot_ui.plot_bounds().max();
            (xmax - xmin, ymax - ymin)
        };
        // calculate distance in screen coordinates via rescaling to `span`
        // otherwise there can be suprising outcomes when selecing the
        // closest point
        if point.x - xi == 0.0 {
            dbg!(point.x, xi);
        }
        let dist = f64::sqrt(((point.x - xi) / span.0).powi(2) + ((point.y - yi) / span.1).powi(2));
        Some(N64::from_f64(dist))
    } else {
        None
    }
}

// ---- NormalizeExtension ----------------------------------------------------

pub struct NormalizeExtensionGUI {
    pub xi: f64,
    pub xj: Option<f64>,
    pub is_active: bool,
}

impl PlotExtensionGUI for NormalizeExtensionGUI {
    fn get_extension_result(&self) -> PlotExtensionResult {
        PlotExtensionResult::Normalize((self.xi, self.xj))
    }

    fn get_is_active_reference(&mut self) -> &mut bool {
        &mut self.is_active
    }

    fn extension_toggle_label(&self) -> String {
        "Add/Remove Marker".to_string()
    }

    fn modify_plot(&mut self, plot_ui: &mut PlotUi) {
        if self.is_active {
            let ctx = plot_ui.ctx();
            let primary_down = ctx.input(|i| i.pointer.primary_down());
            let secondary_down = ctx.input(|i| i.pointer.secondary_down());
            let hovered = plot_ui.response().hovered();
            if primary_down && hovered {
                if let Some(pts) = plot_ui.pointer_coordinate() {
                    self.xi = pts.x
                }
            }
            if secondary_down && hovered {
                if let Some(pts) = plot_ui.pointer_coordinate() {
                    self.xj = Some(pts.x)
                }
            }
            if ctx.input(|i| i.key_down(egui::Key::D)) {
                self.xj = None
            }
        }
        let red = Color32::from_rgb(255, 0, 0);
        plot_ui.vline(egui_plot::VLine::new(self.xi).color(red.clone()));
        if let Some(xj) = self.xj {
            plot_ui.vline(egui_plot::VLine::new(xj).color(red))
        }
    }
    fn is_pan_allowed(&self) -> bool {
        false
    }
}
