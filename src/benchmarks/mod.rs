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
    pub metrics: Vec<BenchmarkMetric>,
    pub run_count: usize,
    pub warmup_count: usize,
    pub median_raw: f64,
    pub variance: f64,
    pub confidence_interval: [f64; 2],
    pub error: Option<String>,
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
fn stats_from(mut values: Vec<f64>) -> MeasurementStats {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let median = values[values.len() / 2];

    MeasurementStats {
        median,
        mean,
        std_dev,
        min: values[0],
        max: values[values.len() - 1],
        values,
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MeasurementStats {
    pub median: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub values: Vec<f64>,
}

impl MeasurementStats {
    pub fn confidence_95(&self) -> [f64; 2] {
        let n = self.values.len() as f64;
        let margin = 1.96 * self.std_dev / n.sqrt();
        [self.mean - margin, self.mean + margin]
    }
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
