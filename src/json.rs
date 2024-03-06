// use serde_json::{Result, Value};

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;

use crate::error::ProcessorError;

#[derive(Serialize, Deserialize)]
struct JsonData {
    positions: Vec<Vec<(usize, usize)>>,
    stddevs: Vec<Vec<usize>>,
}

pub fn positions_to_json(
    positions: &[Vec<(usize, usize)>],
    stddevs: &[Vec<usize>],
    filename_ext: &(String, String),
) -> Result<(), ProcessorError> {
    let ext = "-data.json";
    let data = JsonData {
        positions: positions.to_vec(),
        stddevs: stddevs.to_vec(),
    };

    let mut file = File::create(format!("{}{ext}", filename_ext.0))?;
    file.write_all(serde_json::to_string(&data)?.as_bytes())?;

    Ok(())
}
