pub mod cpu;
pub mod gpu;
pub mod vram;
pub mod npu;
pub mod ram;

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkMetric {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResultPayload {
    pub component_type: String,
    pub score: Option<u64>,
    /// The score at each end of `median_ci_95`, on the same scale; set with
    /// `score` by [`crate::apply_score`]. `None` when there is no calibrated scale.
    pub score_ci_95: Option<[u64; 2]>,
    pub metrics: Vec<BenchmarkMetric>,
    pub run_count: usize,
    pub warmup_count: usize,
    /// Median of the primary metric's per-run values, in that metric's unit.
    pub median_raw: f64,
    /// Sample standard deviation (n - 1) of the same per-run values.
    pub std_dev: f64,
    /// `std_dev` as a percentage of the mean, comparable across units.
    pub cv_percent: f64,
    /// 95% interval for `median_raw` from order statistics (see
    /// [`Summary::median_ci_95`] for what it does and doesn't promise). It always
    /// contains `median_raw`; it is [min, max] when there were too few runs for a
    /// 95% interval, which `median_ci_valid` reports.
    pub median_ci_95: [f64; 2],
    pub median_ci_valid: bool,
    /// When the benchmark started (Unix ms) and how long it took. Set by
    /// [`timed`]; zero when the caller didn't time the run.
    pub started_at_unix_ms: u64,
    pub duration_ms: u64,
    pub error: Option<String>,
}

/// Run a benchmark and record when it started and how long it took.
pub fn timed<F: FnOnce() -> Result<BenchmarkResultPayload, String>>(f: F) -> Result<BenchmarkResultPayload, String> {
    let started_at_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let start = Instant::now();
    let mut result = f()?;
    result.started_at_unix_ms = started_at_unix_ms;
    result.duration_ms = start.elapsed().as_millis() as u64;
    Ok(result)
}

// Config types from the frontend
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuConfig {
    pub matrix_size: Option<usize>,
    pub thread_count: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuConfig {
    pub iterations: Option<u32>,
    pub workgroup_size: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VramConfig {
    pub buffer_size_mb: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct NpuConfig {
    pub iterations: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RamConfig {
    pub buffer_size_mb: Option<usize>,
    pub thread_count: Option<usize>,
}

/// Time-budgeted measurement: discard `warmup` iterations, then keep sampling
/// `f` until at least `min_runs` samples are collected AND `target_secs` of
/// wall-clock has elapsed (hard-capped at `max_runs`). Because `f` returns a
/// per-iteration *rate* (e.g. GFLOPS / GB/s), running more iterations does not
/// change the measured rate or the resulting score — it only makes each
/// benchmark run for a meaningful, perceptible duration and tightens the
/// confidence interval. Faster hardware simply collects more samples.
pub fn measure_timed<F: FnMut() -> f64>(
    mut f: F,
    warmup: usize,
    min_runs: usize,
    max_runs: usize,
    target_secs: f64,
) -> MeasurementStats {
    for _ in 0..warmup {
        let _ = f();
    }
    let start = Instant::now();
    let mut values: Vec<f64> = Vec::new();
    loop {
        values.push(f());
        let done = values.len() >= min_runs && start.elapsed().as_secs_f64() >= target_secs;
        if done || values.len() >= max_runs {
            break;
        }
    }
    stats_from(values)
}

/// Compute summary statistics from a set of per-iteration measurements.
fn stats_from(values: Vec<f64>) -> MeasurementStats {
    let summary = summarize(&values);
    MeasurementStats { summary, values }
}

#[derive(Debug, Clone)]
pub struct MeasurementStats {
    /// Statistics of the accepted (finite) measurements; `summary.n` counts them.
    pub summary: Summary,
    /// Every measurement taken, in order, including any non-finite ones the
    /// summary left out. Report `n`, not `values.len()`, as the run count.
    pub values: Vec<f64>,
}

impl std::ops::Deref for MeasurementStats {
    type Target = Summary;
    fn deref(&self) -> &Summary {
        &self.summary
    }
}

/// Summary statistics of one metric's per-run values, all in that metric's unit.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub n: usize,
    /// The middle value; the mean of the two middle values when `n` is even.
    pub median: f64,
    pub mean: f64,
    /// Sample standard deviation (n - 1); 0 for fewer than two values.
    pub std_dev: f64,
    /// `std_dev / mean` as a percentage; 0 when the mean is 0.
    pub cv_percent: f64,
    pub min: f64,
    pub max: f64,
    /// 95% interval for the median from order statistics. It always contains
    /// `median`, the median of these runs. It covers the true median with at
    /// least 95% probability when the runs are independent draws from one stable
    /// distribution, whatever its shape; drift, throttling or warm-up that changes
    /// over the run breaks that assumption, and so can the time-budgeted stopping
    /// rule. Separate runs can differ by more than one run's interval suggests.
    pub median_ci_95: [f64; 2],
    /// True when there were enough runs (n >= 6) to form a 95% interval; the
    /// interval is otherwise [min, max]. It doesn't check the assumptions above.
    pub ci_valid: bool,
}

/// Summarize a set of per-run measurements. Order doesn't matter. Non-finite
/// values (NaN, ±infinity) are not measurements: they're left out and not
/// counted in `n`.
pub fn summarize(values: &[f64]) -> Summary {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return Summary { n: 0, median: 0.0, mean: 0.0, std_dev: 0.0, cv_percent: 0.0, min: 0.0, max: 0.0, median_ci_95: [0.0, 0.0], ci_valid: false };
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    let median = if n % 2 == 1 { v[n / 2] } else { midpoint(v[n / 2 - 1], v[n / 2]) };
    // Welford's running mean and sum of squares, on values scaled into [-1, 1]
    // so neither can overflow while the result is still representable.
    let scale = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let (mut mean_s, mut m2_s) = (0.0_f64, 0.0_f64);
    for (i, x) in v.iter().enumerate() {
        let x = x / scale;
        let d = x - mean_s;
        mean_s += d / (i + 1) as f64;
        m2_s += d * (x - mean_s);
    }
    let mean = mean_s * scale;
    let sd_s = if n > 1 { (m2_s / (n - 1) as f64).sqrt() } else { 0.0 };
    let std_dev = sd_s * scale;
    // From the scaled values, so it stays finite whenever the ratio is.
    let cv_percent = if mean_s != 0.0 { 100.0 * (sd_s / mean_s.abs()) } else { 0.0 };
    let (median_ci_95, ci_valid) = match median_ci_ranks(n) {
        Some((lo, hi)) => ([v[lo - 1], v[hi - 1]], true),
        None => ([v[0], v[n - 1]], false),
    };
    Summary { n, median, mean, std_dev, cv_percent, min: v[0], max: v[n - 1], median_ci_95, ci_valid }
}

/// The midpoint of two finite values without overflow (two huge values) or
/// underflow (two subnormal ones).
fn midpoint(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if sum.is_finite() { sum / 2.0 } else { a / 2.0 + b / 2.0 }
}

/// 1-based ranks `(l, n - l + 1)` of the order statistics bounding a 95%
/// interval for the median: `l` is the largest rank with P(X <= l - 1) <= 0.025
/// for X ~ Binomial(n, 1/2), so the interval covers the median with probability
/// at least 95% whatever the distribution. `None` when no rank qualifies (n < 6).
/// The binomial terms are built in log space, so large `n` doesn't underflow.
fn median_ci_ranks(n: usize) -> Option<(usize, usize)> {
    let nf = n as f64;
    let mut log_pmf = nf * 0.5_f64.ln(); // log P(X = 0)
    let mut cdf = 0.0;
    let mut l = 0;
    for k in 0..n {
        cdf += log_pmf.exp(); // P(X <= k)
        if cdf > 0.025 {
            break;
        }
        l = k + 1;
        log_pmf += ((nf - k as f64) / (k as f64 + 1.0)).ln();
    }
    if l == 0 { None } else { Some((l, n - l + 1)) }
}

/// Map a raw measurement to the 0-10000 calibrated scale.
/// `anchors` are (raw_value, score) pairs sorted by raw_value ascending.
pub fn calibrate(raw: f64, anchors: &[(f64, f64)]) -> u64 {
    if anchors.is_empty() {
        return 0;
    }
    if raw <= anchors[0].0 {
        return (anchors[0].1 * (raw / anchors[0].0)).max(0.0) as u64;
    }
    if raw >= anchors[anchors.len() - 1].0 {
        return anchors[anchors.len() - 1].1 as u64;
    }
    for window in anchors.windows(2) {
        let (r0, s0) = window[0];
        let (r1, s1) = window[1];
        if raw >= r0 && raw <= r1 {
            let t = (raw - r0) / (r1 - r0);
            return (s0 + t * (s1 - s0)) as u64;
        }
    }
    0
}
