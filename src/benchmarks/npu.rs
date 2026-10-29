use super::{NpuConfig, BenchmarkResultPayload, BenchmarkMetric};

/// NPU benchmark — detects a dedicated AI accelerator.
///
/// Without the `ort` crate (ONNX Runtime), this module honestly reports
/// that no NPU was detected rather than faking it with GPU compute.
/// To enable real NPU benchmarking, add `ort` to Cargo.toml and
/// implement inference-based measurement.
pub fn run(_config: NpuConfig) -> Result<BenchmarkResultPayload, String> {
    // Phase 1: Honest detection — no faking GPU as NPU
    //
    // TODO: When `ort` is added as a dependency:
    // 1. Try creating an ONNX session with DirectML (Windows) or CoreML (macOS)
    // 2. If the NPU execution provider is available, run a small model
    // 3. Measure inference throughput in TOPS
    //
    // For now, report that no dedicated NPU is detected.

    let npu_detected = false; // Will be replaced by actual detection

    if npu_detected {
        // Future: actual NPU benchmark results
        unreachable!()
    } else {
        Ok(BenchmarkResultPayload {
            component_type: "NPU".to_string(),
            score: None,
            metrics: vec![
                BenchmarkMetric {
                    name: "Status".to_string(),
                    value: 0.0,
                    unit: "".to_string(),
                    description: "No dedicated AI accelerator detected".to_string(),
                },
                BenchmarkMetric {
                    name: "Note".to_string(),
                    value: 0.0,
                    unit: "".to_string(),
                    description: "NPU support requires ONNX Runtime with DirectML/CoreML provider".to_string(),
                },
            ],
            run_count: 0,
            warmup_count: 0,
            median_raw: 0.0,
            std_dev: 0.0,
            cv_percent: 0.0,
            median_ci_95: [0.0, 0.0],
            median_ci_valid: false,
            started_at_unix_ms: 0,
            duration_ms: 0,
            error: Some("No dedicated AI accelerator detected on this system.".to_string()),
        })
    }
}
