#[derive(Default)]
pub struct GenerateOptions {
    /// Capture timing information (TTFT and ITL)
    pub time: bool,

    /// Completely silent mode - no stdout, no progress bars, no timing output
    /// Useful for benchmarks where you only want the result
    pub silent: bool,
}
