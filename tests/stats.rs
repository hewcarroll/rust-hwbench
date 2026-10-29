//! Statistics: the interval a result reports must describe the median it reports.
//! These need no hardware, so they run anywhere.

use rust_hwbench::{benchmarks::vram, summarize};

// A small deterministic PRNG, so the property test needs no extra crates.
struct XorShift(u64);

impl XorShift {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Runs shaped like the Oct 28 kala CPU run: most iterations near 20.3 GFLOPS,
/// a tail of slower ones (throttling, scheduler noise).
fn skewed_runs(n: usize, seed: u64) -> Vec<f64> {
    let mut r = XorShift(seed.max(1));
    (0..n)
        .map(|_| {
            if r.next_f64() < 0.2 {
                18.5 + 1.5 * r.next_f64()
            } else {
                20.3 + 0.1 * r.next_f64()
            }
        })
        .collect()
}

#[test]
fn mean_based_interval_misses_the_median_on_skewed_runs() {
    // Negative control: 0.1.0 reported mean ± 1.96·sd/√n next to the median.
    let v = skewed_runs(64, 7);
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    let upper = mean + 1.96 * sd / n.sqrt();
    let s = summarize(&v);
    assert!(s.median > upper, "the old interval should miss the median: median {} > upper {}", s.median, upper);
}

#[test]
fn median_interval_always_contains_the_median() {
    for seed in 1..=200u64 {
        for &n in &[6usize, 7, 10, 31, 64, 100, 935, 2682] {
            let v = skewed_runs(n, seed * 31 + n as u64);
            let s = summarize(&v);
            assert!(s.ci_valid, "n={n} should give a valid interval");
            assert!(
                s.median_ci_95[0] <= s.median && s.median <= s.median_ci_95[1],
                "n={n} seed={seed}: {s:?}"
            );
        }
    }
}

#[test]
fn known_answers() {
    // Even n: the median averages the two middle values.
    assert_eq!(summarize(&[4.0, 1.0, 3.0, 2.0]).median, 2.5);

    // Sample standard deviation (n - 1): sqrt(32 / 7).
    let s = summarize(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert!((s.std_dev - 2.138_089_935).abs() < 1e-6, "{}", s.std_dev);
    assert!((s.cv_percent - 42.761_798_7).abs() < 1e-4, "{}", s.cv_percent);

    // n = 100: the exact binomial interval is the 40th to 61st values.
    let v: Vec<f64> = (1..=100).map(f64::from).collect();
    assert_eq!(summarize(&v).median_ci_95, [40.0, 61.0]);
}

#[test]
fn too_few_runs_flag_the_interval() {
    // With 5 runs no order-statistic interval reaches 95% coverage.
    let s = summarize(&[3.0, 1.0, 2.0, 5.0, 4.0]);
    assert!(!s.ci_valid);
    assert_eq!(s.median_ci_95, [1.0, 5.0]);

    // With 6 the min–max range covers the median with probability 1 - 2/64.
    let s = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(s.ci_valid);
    assert_eq!(s.median_ci_95, [1.0, 6.0]);
}

#[test]
fn empty_input_is_all_zero() {
    let s = summarize(&[]);
    assert_eq!((s.n, s.median, s.std_dev, s.ci_valid), (0, 0.0, 0.0, false));
}

#[test]
fn vram_bandwidth_is_summarized_in_gb_per_s() {
    // 0.1.0 took VRAM's spread over durations in seconds, which rounded to 0.00.
    let bytes_moved = 2.0 * 256.0 * 1024.0 * 1024.0;
    let durations: Vec<f64> = (0..50).map(|i| 0.0014 + 0.00002 * (i % 7) as f64).collect();
    let s = vram::bandwidth_summary(bytes_moved, &durations);
    assert!(s.std_dev > 0.5, "spread should be in GB/s, got {}", s.std_dev);
    assert!(s.mean > 300.0 && s.mean < 400.0, "mean {}", s.mean);
    assert!(s.median_ci_95[0] <= s.median && s.median <= s.median_ci_95[1], "{s:?}");
}

// ---- Round 1 review (Codex): reproduce each finding before fixing it ----

#[test]
fn large_n_interval_has_exact_ranks() {
    // Exact binomial ranks for n = 2682 (checked independently): 1290th and 1393rd.
    let v: Vec<f64> = (1..=2682).map(f64::from).collect();
    assert_eq!(summarize(&v).median_ci_95, [1290.0, 1393.0]);
}

#[test]
fn singleton_pair_and_all_equal() {
    let s = summarize(&[7.0]);
    assert_eq!((s.n, s.median, s.std_dev, s.ci_valid), (1, 7.0, 0.0, false));
    let s = summarize(&[1.0, 3.0]);
    assert_eq!((s.median, s.ci_valid, s.median_ci_95), (2.0, false, [1.0, 3.0]));
    let s = summarize(&[5.0; 10]);
    assert_eq!((s.median, s.std_dev, s.cv_percent, s.median_ci_95), (5.0, 0.0, 0.0, [5.0, 5.0]));
}

#[test]
fn non_finite_values_are_left_out() {
    // NaN and infinities are not measurements: they're dropped and counted out of n.
    let s = summarize(&[f64::NAN; 6]);
    assert_eq!((s.n, s.median, s.ci_valid), (0, 0.0, false));
    let s = summarize(&[1.0, 2.0, f64::NAN, 3.0, f64::INFINITY, 4.0, 5.0, f64::NEG_INFINITY, 6.0]);
    assert_eq!((s.n, s.median, s.median_ci_95, s.ci_valid), (6, 3.5, [1.0, 6.0], true));
    assert!(s.std_dev.is_finite() && s.cv_percent.is_finite());
}

#[test]
fn huge_finite_values_do_not_overflow() {
    let s = summarize(&[1e308; 6]);
    assert_eq!((s.median, s.mean, s.std_dev, s.cv_percent), (1e308, 1e308, 0.0, 0.0));
    let s = summarize(&[1e308, 1.7e308]);
    assert!(s.median.is_finite() && s.mean.is_finite() && s.std_dev.is_finite(), "{s:?}");
}

#[test]
fn vram_median_is_the_median_of_per_run_rates() {
    // 1 GB in 1 s and in 2 s: rates 1.0 and 0.5 GB/s, median 0.75 (0.1.0 reported 0.5).
    assert_eq!(vram::bandwidth_summary(1e9, &[1.0, 2.0]).median, 0.75);
    // Infinite, zero, negative and NaN durations are not runs.
    let s = vram::bandwidth_summary(1e9, &[f64::INFINITY, 0.0, -1.0, f64::NAN, 1.0]);
    assert_eq!((s.n, s.median), (1, 1.0));
}
