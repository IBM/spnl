use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

// Custom progress handler for hf-hub downloads with delayed display
pub(crate) struct DownloadProgress {
    bar: Option<ProgressBar>,
    mp: MultiProgress,
    filename: String,
    start_time: std::time::Instant,
    visible: bool,
    delay_ms: u64,
    length: u64,
}

impl DownloadProgress {
    pub(crate) fn new(mp: &MultiProgress, filename: &str) -> Self {
        Self {
            bar: None,
            mp: mp.clone(),
            filename: filename.to_string(),
            start_time: std::time::Instant::now(),
            visible: false,
            delay_ms: 500, // Show progress bar only if download takes > 500ms
            length: 0,
        }
    }

    fn maybe_show(&mut self) {
        if !self.visible && self.start_time.elapsed().as_millis() > self.delay_ms as u128 {
            // Create and add the bar only when we actually need to show it
            let bar = self.mp.add(ProgressBar::new(self.length));
            bar.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}) ({eta})",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
            );
            bar.set_message(self.filename.clone());
            self.bar = Some(bar);
            self.visible = true;
        }
    }
}

impl hf_hub::api::Progress for DownloadProgress {
    fn init(&mut self, size: usize, _filename: &str) {
        self.length = size as u64;
        self.maybe_show();
        if let Some(bar) = &self.bar {
            bar.set_length(size as u64);
        }
    }

    fn update(&mut self, size: usize) {
        self.maybe_show();
        // update() provides incremental bytes, not cumulative
        if let Some(bar) = &self.bar {
            bar.inc(size as u64);
        }
    }

    fn finish(&mut self) {
        // Only finish and clear if the bar was actually created and shown
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        }
        // If bar is None, nothing was ever shown, so nothing to clean up
    }
}

// Made with Bob
