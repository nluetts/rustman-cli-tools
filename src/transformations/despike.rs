use super::Transformer;
use crate::common::Dataset;
use anyhow::{anyhow, Result};
use clap::Parser;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix2};
use noisy_float::types::N64;
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    ops::{Index, IndexMut},
};

#[derive(Debug, Parser, Serialize, Deserialize)]
#[serde(tag = "transformation")]
pub struct DespikeTransform {
    #[clap(help = "siglim")]
    pub siglim: f64,
    #[clap(help = "sigfrac?")]
    pub flim: f64,
}

impl Transformer for DespikeTransform {
    fn config_to_string(&self) -> Result<String> {
        serde_yaml::to_string(&self).map_err(anyhow::Error::msg)
    }
    fn transform(&mut self, dataset: &mut Dataset) -> Result<()> {
        let frames: Vec<_> = dataset
            .data
            .columns()
            .into_iter()
            .skip(1)
            .step_by(2)
            .collect();
        let frames = ndarray::stack(Axis(1), &frames)?;
        let db = DespikeBuffer::new(frames)?;
        let despiked_frames = despike(db, self.siglim, self.flim, 1.0, 6.0, 4);
        for i in 0..despiked_frames.nrows() {
            for j in 0..despiked_frames.ncols() {
                dataset.data[[i, j * 2 + 1]] = despiked_frames[[i, j]]
            }
        }
        Ok(())
    }
}

struct DespikeBuffer {
    _original_data: Array2<f64>,
    input_data: MirroredArray2,
    data_mask: Array2<bool>,
    laplacian: MirroredArray2,
    median_filtered_data: MirroredArray2,
    signal_to_noise_buffer: MirroredArray2,
    fine_structure_buffer: MirroredArray2,
    median_window_buffer: [N64; 50],
    ncols: usize,
    nrows: usize,
}

impl DespikeBuffer {
    fn new(original_data: Array2<f64>) -> Result<Self> {
        let nrows = original_data.nrows();
        let ncols = original_data.ncols();
        if nrows < 2 || ncols < 2 {
            return Err(anyhow!(format!(
                "spectral dataset must have at least 2 rows and 2 columns, got {} and {}",
                nrows, ncols
            )));
        };
        let input_data = MirroredArray2::new(original_data.clone());
        let laplacian = MirroredArray2::zeros((nrows, ncols));
        let median_filtered_data = MirroredArray2::zeros((nrows, ncols));
        let signal_to_noise_buffer = MirroredArray2::zeros((nrows, ncols));
        let fine_structure_buffer = MirroredArray2::zeros((nrows, ncols));
        let data_mask: Array2<bool> = ndarray::Array2::default((nrows, ncols));
        let median_window_buffer = [N64::default(); 50];
        let db = Self {
            _original_data: original_data,
            input_data,
            data_mask,
            laplacian,
            fine_structure_buffer,
            median_filtered_data,
            signal_to_noise_buffer,
            median_window_buffer,
            nrows,
            ncols,
        };
        Ok(db)
    }
}

// apply despike algorithm to input_data in `db`
fn despike(
    mut db: DespikeBuffer,
    siglim: f64,
    flim: f64,
    gain: f64,
    readnoise: f64,
    iter: usize,
) -> Array2<f64> {
    for _ in 0..iter {
        laplace_convolve(&mut db);
        let laplacian = &db.laplacian; // borrowing here to make sure not to accidentially mutate laplacian anymore

        // calculate S' by repeatedly modifying data in signal_to_noise_buffer
        median_filter(
            &db.input_data,
            &mut db.median_filtered_data,
            &mut db.median_window_buffer,
            5,
        );
        for i in 0..db.nrows {
            for j in 0..db.ncols {
                // equation 10 in van Dokkum 2001
                let noise = 1.0 / gain
                    * f64::sqrt(gain * db.median_filtered_data[[i, j]] + readnoise.powi(2));
                db.signal_to_noise_buffer[[i, j]] = laplacian[[i, j]] / (2.0 * noise);
            }
        } // signal_to_noise_buffer now holds S, equation 11 in van Dokkum 2001
        median_filter(
            &db.signal_to_noise_buffer,
            &mut db.median_filtered_data,
            &mut db.median_window_buffer,
            5,
        ); // median_filtered_data now holds 5x5 median filtered S
        for i in 0..db.nrows {
            for j in 0..db.ncols {
                db.signal_to_noise_buffer[[i, j]] =
                    db.signal_to_noise_buffer[[i, j]] - db.median_filtered_data[[i, j]];
            }
        } // signal_to_noise_buffer now holds S', equation 13 in van Dokkum 2001
        let laplacian_to_noise = &db.signal_to_noise_buffer;

        // calculate fine structure image
        //
        median_filter(
            &db.input_data,
            &mut db.median_filtered_data,
            &mut db.median_window_buffer,
            3,
        ); // median_filtered_data now holds 3x3 median filtered input data
        let median_filtered_image = &db.median_filtered_data;
        median_filter(
            &median_filtered_image,
            &mut db.fine_structure_buffer,
            &mut db.median_window_buffer,
            7,
        ); // fine_structure_buffer now holds 3x3 and then 7x7 median filtered input data
        for i in 0..db.nrows {
            for j in 0..db.ncols {
                // dbg!(
                //     db.median_filtered_data[[i, j]],
                //     db.fine_structure_buffer[[i, j]]
                // );
                db.fine_structure_buffer[[i, j]] =
                    db.median_filtered_data[[i, j]] - db.fine_structure_buffer[[i, j]];
            }
        } // fine_structure_buffer now holds fine structure image, equation 14 in van Dokkum 2001
        let fine_structure_image = &db.fine_structure_buffer;
        for i in 0..db.nrows {
            for j in 0..db.ncols {
                // dbg!(laplacian_to_noise[[i, j]]);
                // dbg!(laplacian[[i, j]] / fine_structure_image[[i, j]]);
                let is_cosmic_ray = laplacian_to_noise[[i, j]] > siglim
                    && laplacian[[i, j]] / fine_structure_image[[i, j]] > flim;
                if is_cosmic_ray {
                    db.data_mask[[i, j]] = is_cosmic_ray;
                    db.input_data[[i, j]] = median_filtered_image[[i, j]];
                }
            }
        }
    }
    db.input_data.data
}

/// perform laplace transformation on data in db.copied_input_data
///
/// data is upscaled in process of convolution
fn laplace_convolve(db: &mut DespikeBuffer) {
    // laplace kernel with used indices
    //
    //  0 -1  0
    // -1  4 -1
    //  0 -1  0
    //
    // it is applied such that the result is already upsampled by a factor of 2
    for i in 0..(db.nrows as i32) {
        for j in 0..(db.ncols as i32) {
            // get image elements
            let ij = db.input_data[[i, j]];
            let im1j = db.input_data[[i - 1, j]];
            let ijm1 = db.input_data[[i, j - 1]];
            let ip1j = db.input_data[[i + 1, j]];
            let ijp1 = db.input_data[[i, j + 1]];
            // upper left quadrant of supersampled pixel
            let subpixel_upper_left = 2.0 * ij - im1j - ijm1;
            // upper right quadrant
            let subpixel_upper_right = 2.0 * ij - im1j - ijp1;
            // lower left quadrant
            let subpixel_lower_left = 2.0 * ij - ip1j - ijm1;
            // lower right quadrant
            let subpixel_lower_right = 2.0 * ij - ip1j - ijp1;
            let convolution_elements = [
                subpixel_lower_right,
                subpixel_lower_left,
                subpixel_upper_right,
                subpixel_upper_left,
            ];
            db.laplacian[[i, j]] = convolution_elements.into_iter().filter(|x| *x > 0.0).sum();
        }
    }
}

fn store_pgm(arr2: &Array2<f64>) {
    let (min, max) = arr2.iter().fold((f64::MAX, f64::MIN), |(min, max), &next| {
        let nmin = if next < min { next } else { min };
        let nmax = if next > max { next } else { max };
        (nmin, nmax)
    });
    // map array values onto 0 to 255 grayscale
    let rescale = |val: &f64| f64::floor((val - min) / max * 255.) as u8;
    let mut fh = std::fs::File::create("dbg.pgm").unwrap();
    fh.write(format!("P2\n{} {}\n255\n", arr2.nrows(), arr2.ncols()).as_bytes())
        .unwrap();
    arr2.axis_iter(Axis(0)).for_each(|row| {
        let mut bytes: String = row
            .iter()
            .map(rescale)
            .map(|x| format!("{}", x) + " ")
            .collect();
        bytes.push('\n');
        fh.write(&bytes.as_bytes()).unwrap();
    });
}

// apply median filter to data in despike buffer
//
// choose where data comes from with `source` and where the median filtered
// data is stored with `target`
fn median_filter<const N: usize>(
    input: &MirroredArray2,
    output: &mut MirroredArray2,
    median_window_buffer: &mut [N64; N],
    window_size: usize,
) {
    for i in 0..(input.data.nrows() as i32) {
        for j in 0..(input.data.ncols() as i32) {
            let mut index_buffer = 0;
            for k in 0..window_size {
                for l in 0..window_size {
                    let k = k as i32 - (window_size as i32 / 2);
                    let l = l as i32 - (window_size as i32 / 2);
                    median_window_buffer[index_buffer] = N64::from_f64(input[[i + k, j + l]]);
                    // if j > 2 && i > 2 {
                    // dbg!(k, l, input[[i + k, j + l]], &median_window_buffer);
                    // }
                    index_buffer += 1;
                }
            }
            median_window_buffer[0..window_size * window_size].sort();
            output[[i, j]] = f64::from(median_window_buffer[window_size / 2]);
        }
    }
}

/// a custom 2D array that will mirror data on boundaries when accessed out of bounds
///
/// Negative indices are allowed, so indexing is done with i32. This struct
/// is used for image filters that need special behavior on the data
/// boundaries
struct MirroredArray2 {
    data: Array2<f64>,
}

impl MirroredArray2 {
    fn new<T>(data: ArrayBase<T, Ix2>) -> Self
    where
        T: Data<Elem = f64>,
    {
        Self {
            data: data.to_owned(),
        }
    }
    fn zeros(shape: (usize, usize)) -> Self {
        let data = Array2::zeros(shape);
        Self { data }
    }
    /// mirror the index at idx = 0 and idx = num_elem - 1
    fn mirror_index(idx: i32, num_elem: usize) -> usize {
        let num_elem = num_elem as i32;
        let i = if idx <= 0 {
            -(idx % num_elem)
        } else if idx >= num_elem {
            num_elem - 1 - (idx % num_elem)
        } else {
            idx
        };
        i as usize
    }
}

impl Index<[i32; 2]> for MirroredArray2 {
    type Output = f64;
    fn index(&self, index: [i32; 2]) -> &Self::Output {
        let m = self.data.nrows();
        let n = self.data.ncols();
        let i = MirroredArray2::mirror_index(index[0], m);
        let j = MirroredArray2::mirror_index(index[1], n);
        &self.data[[i, j]]
    }
}

impl IndexMut<[i32; 2]> for MirroredArray2 {
    fn index_mut(&mut self, index: [i32; 2]) -> &mut Self::Output {
        let m = self.data.nrows();
        let n = self.data.ncols();
        let i = MirroredArray2::mirror_index(index[0], m);
        let j = MirroredArray2::mirror_index(index[1], n);
        &mut self.data[[i, j]]
    }
}

impl Index<[usize; 2]> for MirroredArray2 {
    type Output = f64;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self[[index[0] as i32, index[1] as i32]]
    }
}

impl IndexMut<[usize; 2]> for MirroredArray2 {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self[[index[0] as i32, index[1] as i32]]
    }
}

#[cfg(test)]
mod tests {
    use super::{median_filter, MirroredArray2};
    use ndarray::array;
    use noisy_float::types::N64;
    #[test]
    fn test_median_filter() {
        let array2 = MirroredArray2::new(array![[1., 1., 1.], [1., 2., 1.], [1., 1., 1.]]);
        let mut median_filtered_array = MirroredArray2::zeros((3, 3));
        let mut median_buffer: [N64; 25] = [N64::from_f64(0.0); 25];
        median_filter(&array2, &mut median_filtered_array, &mut median_buffer, 3);
        assert_eq!(
            median_filtered_array.data,
            array![[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]
        );
        let mb: Vec<f64> = median_buffer.iter().map(|x| x.const_raw()).collect();
        assert_eq!(
            mb,
            vec![
                1., 1., 1., 1., 1., 1., 1., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0.
            ]
        )
    }
}
