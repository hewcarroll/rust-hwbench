//! Smoke tests for the CPU/RAM paths (these run anywhere, including headless
//! CI). The GPU/VRAM paths require a real adapter and are exercised by the CLI
//! rather than in CI.

use rust_hwbench::{apply_score, benchmarks::cpu, benchmarks::ram, calibrate, score, CpuConfig, RamConfig};

#[test]
fn cpu_runs_and_reports_positive_throughput() {
    let r = cpu::run(CpuConfig {
        matrix_size: Some(128),
        thread_count: Some(2),
    })
    .expect("cpu benchmark should succeed");
    assert_eq!(r.component_type, "CPU");
    assert!(r.score.is_none(), "score is opt-in and must default to None");
    assert!(
        r.metrics.iter().any(|m| m.value > 0.0),
        "expected at least one positive metric"
    );
}

#[test]
fn ram_runs_and_reports_bandwidth() {
    let r = ram::run(RamConfig {
        buffer_size_mb: Some(8),
        thread_count: None,
    })
    .expect("ram benchmark should succeed");
    assert_eq!(r.component_type, "RAM");
    assert!(r.metrics.iter().any(|m| m.unit == "GB/s" && m.value > 0.0));
}

#[test]
fn scoring_is_opt_in_and_monotonic() {
    let r = cpu::run(CpuConfig {
        matrix_size: Some(128),
        thread_count: Some(1),
    })
    .unwrap();
    // Opt-in scoring maps onto the 0..=10000 scale.
    let s = score(&r).expect("CPU has a calibrated scale");
    assert!(s <= 10_000);

    // calibrate is monotonic between anchors.
    let anchors = [(10.0_f64, 1000.0_f64), (20.0, 2000.0)];
    assert!(calibrate(15.0, &anchors) > calibrate(11.0, &anchors));
}
#[test]
fn score_interval_contains_the_score() {
    let mut r = cpu::run(CpuConfig {
        matrix_size: Some(128),
        thread_count: Some(1),
    })
    .unwrap();
    apply_score(&mut r);
    let (s, [lo, hi]) = (r.score.unwrap(), r.score_ci_95.unwrap());
    assert!(lo <= s && s <= hi, "score {s} outside [{lo}, {hi}]");
    assert!(r.median_ci_95[0] <= r.median_raw && r.median_raw <= r.median_ci_95[1]);
}

#[test]
fn json_uses_the_0_2_field_names_and_timed_fills_timing() {
    let r = rust_hwbench::timed(|| {
        ram::run(RamConfig {
            buffer_size_mb: Some(8),
            thread_count: None,
        })
    })
    .unwrap();
    assert!(r.started_at_unix_ms > 1_700_000_000_000, "start time should be set");
    assert!(r.duration_ms > 0, "duration should be set");
    let json = serde_json::to_value(&r).unwrap();
    for key in ["medianRaw", "stdDev", "cvPercent", "medianCi95", "medianCiValid", "scoreCi95", "startedAtUnixMs", "durationMs"] {
        assert!(json.get(key).is_some(), "missing JSON key {key}");
    }
    for old in ["variance", "confidenceInterval"] {
        assert!(json.get(old).is_none(), "0.1.0 key {old} should be gone");
    }
}
