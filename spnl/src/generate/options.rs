#[derive(Default)]
pub struct GenerateOptions {
    /// Prepare query?
    pub prepare: Option<bool>,

    /// Capture timing information (TTFT and ITL)
    pub time: bool,
}
