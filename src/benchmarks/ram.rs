use std::time::Instant;
use rayon::prelude::*;
use super::{RamConfig, BenchmarkResultPayload, BenchmarkMetric, measure_timed};

const WARMUP: usize = 3;
const MIN_RUNS: usize = 5;
const MAX_RUNS: usize = 100_000;
const TARGET_SECS: f64 = 1.0;

// Calibration anchors: (Triad GB/s, score)
pub const RAM_ANCHORS: &[(f64, f64)] = &[
    (15.0,   1000.0),   // DDR4-2133 single-channel
    (30.0,   3000.0),   // DDR4-3200 dual-channel
    (50.0,   5000.0),   // DDR5-4800 dual-channel
    (75.0,   7500.0),   // DDR5-6400 dual-channel
    (100.0, 10000.0),   // DDR5-8000 quad-channel
];

/// STREAM Copy: b[i] = a[i]
fn stream_copy(a: &[f64], b: &mut [f64]) {
    b.par_iter_mut()
        .zip(a.par_iter())
        .for_each(|(bi, ai)| {
            *bi = *ai;
        });
}

/// STREAM Scale: b[i] = scalar * a[i]
fn stream_scale(a: &[f64], b: &mut [f64], scalar: f64) {
    b.par_iter_mut()
        .zip(a.par_iter())
        .for_each(|(bi, ai)| {
            *bi = scalar * *ai;
        });
}

/// STREAM Add: c[i] = a[i] + b[i]
fn stream_add(a: &[f64], b: &[f64], c: &mut [f64]) {
    c.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|(ci, (ai, bi))| {
            *ci = *ai + *bi;
        });
}

/// STREAM Triad: a[i] = b[i] + scalar * c[i]
fn stream_triad(a: &mut [f64], b: &[f64], c: &[f64], scalar: f64) {
    a.par_iter_mut()
        .zip(b.par_iter().zip(c.par_iter()))
        .for_each(|(ai, (bi, ci))| {
            *ai = *bi + scalar * *ci;
        });
}

pub fn run(config: RamConfig) -> Result<BenchmarkResultPayload, String> {
    let buffer_mb = config.buffer_size_mb.unwrap_or(128);
    let n = buffer_mb * 1024 * 1024 / std::mem::size_of::<f64>();

    // Optionally configure thread pool
    let _pool = if let Some(tc) = config.thread_count {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(tc)
                .build()
                .map_err(|e| format!("Thread pool error: {e}"))?,
        )
    } else {
        None
    };

    // Allocate arrays
    let mut a: Vec<f64> = (0..n).map(|i| i as f64 * 1.0).collect();
    let mut b: Vec<f64> = (0..n).map(|i| i as f64 * 2.0).collect();
    let mut c: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let scalar = 3.0;

    let bytes_per_element = std::mem::size_of::<f64>() as f64;

    // Copy: reads a, writes b = 2 * n * 8 bytes
    let copy_stats = measure_timed(
        || {
            let start = Instant::now();
            stream_copy(&a, &mut b);
            std::hint::black_box(&b);
            let elapsed = start.elapsed().as_secs_f64();
            (2.0 * n as f64 * bytes_per_element) / elapsed / 1e9
        },
        WARMUP,
        MIN_RUNS,
        MAX_RUNS,
        TARGET_SECS,
    );

    // Scale: reads a, writes b = 2 * n * 8 bytes
    let scale_stats = measure_timed(
        || {
            let start = Instant::now();
            stream_scale(&a, &mut b, scalar);
            std::hint::black_box(&b);
            let elapsed = start.elapsed().as_secs_f64();
            (2.0 * n as f64 * bytes_per_element) / elapsed / 1e9
        },
        WARMUP,
        MIN_RUNS,
        MAX_RUNS,
        TARGET_SECS,
    );

    // Add: reads a, reads b, writes c = 3 * n * 8 bytes
    let add_stats = measure_timed(
        || {
            let start = Instant::now();
            stream_add(&a, &b, &mut c);
            std::hint::black_box(&c);
            let elapsed = start.elapsed().as_secs_f64();
            (3.0 * n as f64 * bytes_per_element) / elapsed / 1e9
        },
        WARMUP,
        MIN_RUNS,
        MAX_RUNS,
        TARGET_SECS,
    );

    // Triad: reads b, reads c, writes a = 3 * n * 8 bytes
    let triad_stats = measure_timed(
        || {
            let start = Instant::now();
            stream_triad(&mut a, &b, &c, scalar);
            std::hint::black_box(&a);
            let elapsed = start.elapsed().as_secs_f64();
            (3.0 * n as f64 * bytes_per_element) / elapsed / 1e9
        },
        WARMUP,
        MIN_RUNS,
        MAX_RUNS,
        TARGET_SECS,
    );

    Ok(BenchmarkResultPayload {
        component_type: "RAM".to_string(),
        score: None,
        score_ci_95: None,
        metrics: vec![
            BenchmarkMetric {
                name: "Copy".to_string(),
                value: round2(copy_stats.median),
                unit: "GB/s".to_string(),
                description: "b[i] = a[i]".to_string(),
            },
            BenchmarkMetric {
                name: "Scale".to_string(),
                value: round2(scale_stats.median),
                unit: "GB/s".to_string(),
                description: "b[i] = scalar * a[i]".to_string(),
            },
            BenchmarkMetric {
                name: "Add".to_string(),
                value: round2(add_stats.median),
                unit: "GB/s".to_string(),
                description: "c[i] = a[i] + b[i]".to_string(),
            },
            BenchmarkMetric {
                name: "Triad".to_string(),
                value: round2(triad_stats.median),
                unit: "GB/s".to_string(),
                description: "a[i] = b[i] + scalar * c[i]".to_string(),
            },
            BenchmarkMetric {
                name: "Buffer Size".to_string(),
                value: buffer_mb as f64,
                unit: "MB".to_string(),
                description: "Total test buffer per array".to_string(),
            },
        ],
        run_count: triad_stats.values.len(),
        warmup_count: WARMUP,
        median_raw: round2(triad_stats.median),
        std_dev: round2(triad_stats.std_dev),
        cv_percent: (triad_stats.cv_percent * 10.0).round() / 10.0,
        median_ci_95: [round2(triad_stats.median_ci_95[0]), round2(triad_stats.median_ci_95[1])],
        median_ci_valid: triad_stats.ci_valid,
        started_at_unix_ms: 0,
        duration_ms: 0,
        error: None,
    })
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
