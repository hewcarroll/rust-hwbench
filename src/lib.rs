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
//! standard deviation, coefficient of variation and a distribution-free 95%
//! interval for the median. The [`BenchmarkResultPayload::score`] field is `None`
//! by default. Mapping a result onto an opinionated 0–10000 scale is opt-in via
//! [`score`] (or call [`calibrate`] directly against the exported anchor tables,
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
    let anchors: &[(f64, f64)] = match result.component_type.as_str() {
        "CPU" => benchmarks::cpu::CPU_ANCHORS,
        "GPU" => benchmarks::gpu::GPU_ANCHORS,
        "RAM" => benchmarks::ram::RAM_ANCHORS,
        "GPU_MEMORY" => benchmarks::vram::VRAM_ANCHORS,
        _ => return None,
    };
    Some(calibrate(result.median_raw, anchors))
}