use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    os::unix::fs::FileExt,
    path::Path,
};

use super::xml::XMLTag;

#[derive(Debug)]
pub struct SpeData {
    /// Number of frames measured
    frame_count: u64,
    /// Bytes per frame (all ROIs w/o metadata)
    frame_size_bytes: u64,
    /// Bytes per frame stride (all ROIs and metadata)
    frame_stride_bytes: u64,
    /// Exposure time in seconds
    exposure: f64,
    /// Center wavelength in nanometer
    center_wavelength: f64,
    /// Grating
    grating: String,
    /// Wavelength axis
    wavelength_axis: Vec<f64>,
    /// Intensity data ("frames")
    frames: Vec<Vec<u16>>,
    /// Filename of SPE file
    filename: String,
    /// Creation datetime of SPE file
    created: String,
}

impl SpeData {
    pub fn from_path(filepath: &Path) -> Result<SpeData, Box<dyn Error + 'static>> {
        let mut file = File::open(filepath)?;

        // Read XML footer
        //
        // Read start byte of footer
        let mut buf = [0u8; 8];
        file.read_at(&mut buf, 678)?;
        let xml_offset = u64::from_le_bytes(buf);
        // Read footer into bytes
        file.seek(SeekFrom::Start(xml_offset))?;
        let mut xml_footer = String::new();
        file.read_to_string(&mut xml_footer)?;
        // Parse footer bytes into XML
        let xml_document = XMLTag::from_str(&xml_footer)?;
        let xml_index = xml_document.build_index();
        let mut data = SpeData::empty_from_xml_index(xml_index)?;

        // Read data section (assumes full vertical binning, for now)
        file.seek(SeekFrom::Start(4100))?;
        let mut pos = 4100;
        let mut counts_buf = vec![0u8; data.frame_size_bytes as usize];
        while pos + data.frame_stride_bytes <= xml_offset {
            file.read_exact(&mut counts_buf)?;
            let frame: Vec<u16> = counts_buf
                .windows(2)
                .step_by(2)
                .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
                .collect();
            data.frames.push(frame);
            pos += data.frame_stride_bytes;
        }

        Ok(data)
    }

    pub fn write_csv<W: Write>(&self, wrt: &mut W) -> Result<(), Box<dyn Error>> {
        let meta_string = self.get_meta_data_string()?;
        writeln!(wrt, "{meta_string}")?;

        for frame in self.frames.iter() {
            for (cts, wn) in frame.iter().zip(self.wavelength_axis.iter()) {
                writeln!(wrt, "{wn},{cts}")?;
            }
        }

        wrt.flush()?;
        Ok(())
    }

    pub fn get_meta_data_string(&self) -> Result<String, Box<dyn Error>> {
        let mut wrt = BufWriter::new(Vec::<u8>::with_capacity(1000));
        writeln!(wrt, "# filename = {}", self.filename)?;
        writeln!(wrt, "# created = {}", self.created)?;
        writeln!(wrt, "# grating = {}", self.grating)?;
        writeln!(wrt, "# center wavelength = {}", self.center_wavelength)?;
        writeln!(wrt, "# exposure time = {}", self.exposure)?;
        writeln!(wrt, "# frame count = {}", self.frame_count)?;
        wrt.flush()?;

        Ok(String::from_utf8(wrt.into_inner()?)?)
    }

    pub fn get_wavelength(&self) -> &[f64] {
        &self.wavelength_axis
    }

    pub fn get_frames(&self) -> &[Vec<u16>] {
        &self.frames
    }

    fn empty_from_xml_index(index: HashMap<String, &XMLTag>) -> Result<Self, Box<dyn Error>> {
        let center_wavelength = index
            .get("SpeFormat/DataHistories/DataHistory/Origin/Experiment/Devices/Spectrometers/Spectrometer/Grating/CenterWavelength")
            .ok_or("center wavelength not found in XML footer")?
            .contents.parse::<f64>()?;
        let grating = index
            .get("SpeFormat/DataHistories/DataHistory/Origin/Experiment/Devices/Spectrometers/Spectrometer/Grating/Selected")
            .ok_or("grating selection not found in XML footer")?
            .contents.clone();
        let frame_count = index
            .get("SpeFormat/DataFormat/DataBlock")
            .ok_or("frame description not found in XML footer")?
            .parameters
            .get("count")
            .ok_or("frame count not found in XML footer")?
            .parse::<u64>()?;
        let frame_size_bytes = index
            .get("SpeFormat/DataFormat/DataBlock/DataBlock")
            .ok_or("region description not found in XML footer")?
            .parameters
            .get("size")
            .ok_or("region size not found in XML footer")?
            .parse::<u64>()?;
        let frame_stride_bytes = index
            .get("SpeFormat/DataFormat/DataBlock/DataBlock")
            .ok_or("stride length not found in XML footer")?
            .parameters
            .get("stride")
            .ok_or("stride length not found in XML footer")?
            .parse::<u64>()?;
        let exposure = index
            .get("SpeFormat/DataHistories/DataHistory/Origin/Experiment/Devices/Cameras/Camera/ShutterTiming/ExposureTime")
            .ok_or("exposure time not found in XML footer")?
            .contents
            .parse::<u64>()? as f64 / 1000.0;
        let wavelength_axis = index
            .get("SpeFormat/Calibrations/WavelengthMapping/Wavelength")
            .and_then(|&tag| convert_wavelength_string(&tag.contents).ok())
            .ok_or("Unable to extract wavelength axis")?;
        let filename = index
            .get("SpeFormat/DataHistories/DataHistory/Origin/Experiment/Devices/Cameras/Camera/Experiment/FileNameGeneration/BaseFileName")
            .ok_or("filename not found in XML footer")?
            .contents.clone();
        let created = index
            .get("SpeFormat/DataHistories/DataHistory/Origin")
            .ok_or("time data not found in XML footer")?
            .parameters
            .get("created")
            .ok_or("time data not found in XML footer")?
            .clone();

        Ok(Self {
            grating,
            center_wavelength,
            frame_count,
            exposure,
            frame_size_bytes,
            frame_stride_bytes,
            wavelength_axis,
            frames: Vec::new(),
            filename,
            created,
        })
    }
}

fn convert_wavelength_string(raw: &str) -> Result<Vec<f64>, ()> {
    let mut result: Vec<f64> = Vec::new();
    for substr in raw.split(",") {
        result.push(substr.parse().map_err(|_| ())?);
    }
    Ok(result)
}
