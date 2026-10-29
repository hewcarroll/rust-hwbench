//! # rust-hwbench
//!
//! A small, dependency-light hardware benchmark library covering CPU, GPU,
//! system RAM, and GPU VRAM.
//!
//! - **CPU** — FP32/FP64/INT32 single- and multi-core GEMM throughput,
//!   multi-thread scaling, and L1/L2/L3/main-memory random-access latency.
//! - **GPU** — FP32/FP16/INT32 compute throughput via `wgpu` compute shaders,
//!   with the FP16 path hardened against driver hangs (own device + non-blocking
//!   poll + wall-clock timeout).
//! - **RAM** — STREAM Copy/Scale/Add/Triad memory bandwidth.
//! - **VRAM** — internal, host→device, and device→host GPU memory bandwidth.
//!
//! ## Raw-first, scoring opt-in
//!
//! Each `run` returns the measured throughput/bandwidth plus summary statistics
//! of the primary metric's per-run values ([`Summary`]): the median, sample
//! standard deviation, coefficient of variation and a 95% interval for the
//! median from order statistics (see [`Summary::median_ci_95`] for its assumptions). The [`BenchmarkResultPayload::score`] field is `None`
//! by default. Mapping a result onto an opinionated 0–10000 scale is opt-in via
//! [`score`] or [`apply_score`] (or call [`calibrate`] directly against the exported anchor tables,
//! e.g. [`benchmarks::cpu::CPU_ANCHORS`]).
//!
//! ## Example
//!
//! ```no_run
//! use rust_hwbench::{benchmarks::cpu, CpuConfig, score};
//!
//! let mut result = cpu::run(CpuConfig::default()).expect("cpu benchmark");
//! // Opt into the calibrated score:
//! result.score = score(&result);
//! println!("{} score: {:?}", result.component_type, result.score);
//! for m in &result.metrics {
//!     println!("  {:<18} {} {}", m.name, m.value, m.unit);
//! }
//! ```

pub mod benchmarks;
pub mod system;

pub use benchmarks::{
    calibrate, measure_timed, summarize, timed, BenchmarkMetric, BenchmarkResultPayload, CpuConfig,
    GpuConfig, MeasurementStats, NpuConfig, RamConfig, Summary, VramConfig,
};
pub use system::{detect as detect_system, SystemInfo};

/// Opt-in scoring: map a benchmark result's primary raw metric (`median_raw`)
/// onto the 0–10000 calibrated scale using this crate's built-in anchor tables.
///
/// Returns `None` for component types without a calibrated scale (e.g. `"NPU"`).
/// The anchor tables are public ([`benchmarks::cpu::CPU_ANCHORS`] etc.) so you
/// can supply your own and call [`calibrate`] instead.
pub fn score(result: &BenchmarkResultPayload) -> Option<u64> {
    Some(calibrate(result.median_raw, anchors_for(&result.component_type)?))
}

/// The score's own 95% interval: each end of `median_ci_95` mapped through the
/// same calibration as [`score`]. Calibration never decreases, so the interval
/// contains the score. `None` for component types without a calibrated scale.
pub fn score_interval(result: &BenchmarkResultPayload) -> Option<[u64; 2]> {
    let anchors = anchors_for(&result.component_type)?;
    Some([calibrate(result.median_ci_95[0], anchors), calibrate(result.median_ci_95[1], anchors)])
}

/// Attach the calibrated score and its interval to a result.
pub fn apply_score(result: &mut BenchmarkResultPayload) {
    result.score = score(result);
    result.score_ci_95 = score_interval(result);
}

fn anchors_for(component_type: &str) -> Option<&'static [(f64, f64)]> {
    match component_type {
        "CPU" => Some(benchmarks::cpu::CPU_ANCHORS),
        "GPU" => Some(benchmarks::gpu::GPU_ANCHORS),
        "RAM" => Some(benchmarks::ram::RAM_ANCHORS),
        "GPU_MEMORY" => Some(benchmarks::vram::VRAM_ANCHORS),
        _ => None,
    }
}
