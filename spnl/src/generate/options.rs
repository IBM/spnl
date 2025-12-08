#[derive(Clone, Debug, clap::ValueEnum, serde::Serialize)]
pub enum WhatToTime {
    All,
    Gen,
    Gen1,
}

#[derive(Default)]
pub struct GenerateOptions {
    /// Prepare query?
    pub prepare: Option<bool>,

    /// Capture timing information
    pub time: Option<WhatToTime>,
}
