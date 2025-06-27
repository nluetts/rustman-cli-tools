use crate::common::{Dataset, Pair};
use crate::plot::{PlotTransform, SplineExtension};
use crate::transformations::Transformer;
use anyhow::Result;
use clap::Parser;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct DrawBaselineTransform {
    #[clap(short, long, help = "x,y points to draw spline baseline (optional).")]
    pub(crate) points: Option<Vec<Pair<f64>>>,
    #[clap(
        short,
        long,
        action,
        help = "If flag is set, add baseline to dataset instead of subtracting it."
    )]
    pub(crate) store: bool,
}

impl Transformer for DrawBaselineTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let (sender, receiver) = channel();
        let spline_ext = match &self.points {
            None => SplineExtension::new(vec![], sender),
            Some(points) => {
                let points = points.iter().map(|pt| [pt.a, pt.b]).collect();
                SplineExtension::new(points, sender)
            }
        };
        let spline_ext_arcmutex = Arc::new(Mutex::new(spline_ext));
        let mut plot_transform = PlotTransform {
            line_width: Some(1.0),
            extensions: vec![spline_ext_arcmutex.clone()],
            x_lim: None,
            y_lim: None,
            pixels: false,
        };
        // the actual work is done by plot transform + spline drawing extension
        _ = plot_transform.transform(dataset);

        let (points, spline) = receiver.recv().unwrap();
        if self.store {
            // store baseline as a new frame
            let x_p: Array1<f64> = dataset.data.column(0).to_owned();
            let y_p: Array1<f64> = x_p
                .iter()
                .map(|x| spline.sample(*x).unwrap_or(0.0))
                .collect();
            let baseline: Array2<f64> = ndarray::stack![Axis(1), x_p, y_p];
            dataset.data = ndarray::concatenate(Axis(1), &[dataset.data.view(), baseline.view()])?;
        } else {
            // subtract baseline
            for j in (0..dataset.data.ncols()).step_by(2) {
                for i in 0..dataset.data.nrows() {
                    dataset.data[[i, j + 1]] -= spline.sample(dataset.data[[i, j]]).unwrap_or(0.0);
                }
            }
        }

        // Store points to log them to yaml header.
        // TODO: This requires that ALL transformers are mutable, which is not very nice
        self.points = Some(
            points
                .iter()
                .map(|pt| Pair { a: pt[0], b: pt[1] })
                .collect(),
        );
        Ok(())
    }
}
