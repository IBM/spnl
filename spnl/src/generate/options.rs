#[derive(Default)]
pub struct GenerateOptions {
    /// Prepare query?
    pub prepare: Option<bool>,

    /// Capture timing information (TTFT and ITL)
    pub time: bool,

    /// Completely silent mode - no stdout, no progress bars, no timing output
    /// Useful for benchmarks where you only want the result
    pub silent: bool,
}
