use std::time::Instant;
use rayon::prelude::*;
use super::{CpuConfig, BenchmarkResultPayload, BenchmarkMetric, measure_timed};

const WARMUP: usize = 3;
const MIN_RUNS: usize = 3;
const MAX_RUNS: usize = 100_000;
// Per-test wall-clock budget. The CPU battery runs six compute tests, so the
// whole thing lands around ~6s + cache probing.
const TARGET_SECS: f64 = 1.0;

// Calibration anchors: (single-core FP32 GFLOPS, score). The score stays pinned
// to single-core FP32 so it remains comparable across runs and app versions.
pub const CPU_ANCHORS: &[(f64, f64)] = &[
    (15.0,   1000.0),   // i3-12100 / Ryzen 3 class
    (30.0,   3000.0),   // i5-12400 / Ryzen 5 class
    (50.0,   5000.0),   // i7-12700K class
    (70.0,   7500.0),   // i9-13900K class
    (100.0, 10000.0),   // i9-14900KS class
];

// ---------------------------------------------------------------------------
// Single-core GEMM kernels (allocation happens before the timed region)
// ---------------------------------------------------------------------------
fn matmul_f32_single(n: usize) -> f64 {
    let a: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.001) % 1.0).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i + 7) as f32 * 0.0013) % 1.0).collect();
    let mut c = vec![0.0f32; n * n];

    let start = Instant::now();
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(&c);
    2.0 * (n as f64).powi(3) / elapsed / 1e9 // GFLOPS
}

fn matmul_f64_single(n: usize) -> f64 {
    let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.001) % 1.0).collect();
    let b: Vec<f64> = (0..n * n).map(|i| ((i + 7) as f64 * 0.0013) % 1.0).collect();
    let mut c = vec![0.0f64; n * n];

    let start = Instant::now();
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(&c);
    2.0 * (n as f64).powi(3) / elapsed / 1e9 // GFLOPS
}

fn matmul_i32_single(n: usize) -> f64 {
    let a: Vec<i32> = (0..n * n).map(|i| (i as i32 % 7) + 1).collect();
    let b: Vec<i32> = (0..n * n).map(|i| (i as i32 % 13) + 1).collect();
    let mut c = vec![0i32; n * n];

    let start = Instant::now();
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] = c[i * n + j].wrapping_add(a_ik.wrapping_mul(b[k * n + j]));
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(&c);
    2.0 * (n as f64).powi(3) / elapsed / 1e9 // GIOPS (2 int ops per MAC)
}

// ---------------------------------------------------------------------------
// Multi-core GEMM kernels — aggregate throughput.
// Each thread runs its OWN independent matmul (no shared operand), so the test
// measures total compute the CPU can sustain and scales with core count,
// instead of parallelizing one memory-bandwidth-bound matrix (which barely
// scales for fast FP32). Returns aggregate GFLOPS/GIOPS across all tasks.
// ---------------------------------------------------------------------------
fn matmul_f32_multi(n: usize, tasks: usize) -> f64 {
    let mut work: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..tasks)
        .map(|_| {
            let a: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.001) % 1.0).collect();
            let b: Vec<f32> = (0..n * n).map(|i| ((i + 7) as f32 * 0.0013) % 1.0).collect();
            (a, b, vec![0.0f32; n * n])
        })
        .collect();

    let start = Instant::now();
    work.par_iter_mut().for_each(|(a, b, c)| {
        for i in 0..n {
            for k in 0..n {
                let a_ik = a[i * n + k];
                for j in 0..n {
                    c[i * n + j] += a_ik * b[k * n + j];
                }
            }
        }
    });
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(&work[0].2);
    (tasks as f64) * 2.0 * (n as f64).powi(3) / elapsed / 1e9
}

fn matmul_f64_multi(n: usize, tasks: usize) -> f64 {
    let mut work: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..tasks)
        .map(|_| {
            let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.001) % 1.0).collect();
            let b: Vec<f64> = (0..n * n).map(|i| ((i + 7) as f64 * 0.0013) % 1.0).collect();
            (a, b, vec![0.0f64; n * n])
        })
        .collect();

    let start = Instant::now();
    work.par_iter_mut().for_each(|(a, b, c)| {
        for i in 0..n {
            for k in 0..n {
                let a_ik = a[i * n + k];
                for j in 0..n {
                    c[i * n + j] += a_ik * b[k * n + j];
                }
            }
        }
    });
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(&work[0].2);
    (tasks as f64) * 2.0 * (n as f64).powi(3) / elapsed / 1e9
}

fn matmul_i32_multi(n: usize, tasks: usize) -> f64 {
    let mut work: Vec<(Vec<i32>, Vec<i32>, Vec<i32>)> = (0..tasks)
        .map(|_| {
            let a: Vec<i32> = (0..n * n).map(|i| (i as i32 % 7) + 1).collect();
            let b: Vec<i32> = (0..n * n).map(|i| (i as i32 % 13) + 1).collect();
            (a, b, vec![0i32; n * n])
        })
        .collect();

    let start = Instant::now();
    work.par_iter_mut().for_each(|(a, b, c)| {
        for i in 0..n {
            for k in 0..n {
                let a_ik = a[i * n + k];
                for j in 0..n {
                    c[i * n + j] = c[i * n + j].wrapping_add(a_ik.wrapping_mul(b[k * n + j]));
                }
            }
        }
    });
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(&work[0].2);
    (tasks as f64) * 2.0 * (n as f64).powi(3) / elapsed / 1e9
}

// Run a closure on a capped thread pool if one is configured, else the global pool.
fn pooled<F: Fn() -> f64 + Send + Sync>(pool: &Option<rayon::ThreadPool>, f: F) -> f64 {
    match pool {
        Some(p) => p.install(|| f()),
        None => f(),
    }
}

// ---------------------------------------------------------------------------
// Cache / memory random-access latency (pointer chase)
// ---------------------------------------------------------------------------
fn cache_latency_ns(buffer_size: usize) -> f64 {
    let count = buffer_size / std::mem::size_of::<usize>();
    if count < 2 {
        return 0.0;
    }

    let mut indices: Vec<usize> = (0..count).collect();
    let mut seed: u64 = 0xDEADBEEF;
    for i in (1..count).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (seed >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }

    let mut chain = vec![0usize; count];
    for i in 0..count {
        chain[i] = indices[(i + 1) % count];
    }

    let iterations = 1_000_000.min(count * 100);
    let mut idx = 0usize;

    let start = Instant::now();
    for _ in 0..iterations {
        idx = chain[idx];
    }
    let elapsed = start.elapsed().as_nanos() as f64;
    std::hint::black_box(idx);

    elapsed / iterations as f64
}

pub fn run(config: CpuConfig) -> Result<BenchmarkResultPayload, String> {
    let n = config.matrix_size.unwrap_or(512);

    // Task count for multi-core aggregate tests (default: all logical cores).
    let tasks = config
        .thread_count
        .filter(|&t| t > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1));

    // Optional capped pool (honors a user-set thread count); None = global pool.
    let pool = match config.thread_count {
        Some(tc) if tc > 0 => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(tc)
                .build()
                .map_err(|e| format!("Thread pool error: {e}"))?,
        ),
        _ => None,
    };

    // Compute battery (each runs for a fixed time budget)
    let fp32_single = measure_timed(|| matmul_f32_single(n), WARMUP, MIN_RUNS, MAX_RUNS, TARGET_SECS);
    let fp32_multi = measure_timed(|| pooled(&pool, || matmul_f32_multi(n, tasks)), WARMUP, MIN_RUNS, MAX_RUNS, TARGET_SECS);
    let fp64_single = measure_timed(|| matmul_f64_single(n), WARMUP, MIN_RUNS, MAX_RUNS, TARGET_SECS);
    let fp64_multi = measure_timed(|| pooled(&pool, || matmul_f64_multi(n, tasks)), WARMUP, MIN_RUNS, MAX_RUNS, TARGET_SECS);
    let int_single = measure_timed(|| matmul_i32_single(n), WARMUP, MIN_RUNS, MAX_RUNS, TARGET_SECS);
    let int_multi = measure_timed(|| pooled(&pool, || matmul_i32_multi(n, tasks)), WARMUP, MIN_RUNS, MAX_RUNS, TARGET_SECS);

    // Cache / memory latency hierarchy
    let l1_latency = cache_latency_ns(32 * 1024); // 32 KB
    let l2_latency = cache_latency_ns(256 * 1024); // 256 KB
    let l3_latency = cache_latency_ns(8 * 1024 * 1024); // 8 MB
    let mem_latency = cache_latency_ns(64 * 1024 * 1024); // 64 MB

    let mt_scaling = if fp32_single.median > 0.0 {
        fp32_multi.median / fp32_single.median
    } else {
        0.0
    };

    // Score stays pinned to single-core FP32 for comparability.
    let ci = fp32_single.confidence_95();

    Ok(BenchmarkResultPayload {
        component_type: "CPU".to_string(),
        score: None,
        metrics: vec![
            metric("FP32 Single-Core", round2(fp32_single.median), "GFLOPS", "Single-threaded FP32 matrix multiply"),
            metric("FP32 Multi-Core", round2(fp32_multi.median), "GFLOPS", "Aggregate FP32 throughput across all cores"),
            metric("FP64 Single-Core", round2(fp64_single.median), "GFLOPS", "Single-threaded FP64 matrix multiply"),
            metric("FP64 Multi-Core", round2(fp64_multi.median), "GFLOPS", "Aggregate FP64 throughput across all cores"),
            metric("INT32 Single-Core", round2(int_single.median), "GIOPS", "Single-threaded INT32 multiply-add"),
            metric("INT32 Multi-Core", round2(int_multi.median), "GIOPS", "Aggregate INT32 throughput across all cores"),
            metric("MT Scaling", round2(mt_scaling), "x", "Aggregate multi-core / single-core FP32 speedup"),
            metric("L1 R-Lat", round2(l1_latency), "ns", "L1 cache random access latency"),
            metric("L2 R-Lat", round2(l2_latency), "ns", "L2 cache random access latency"),
            metric("L3 R-Lat", round2(l3_latency), "ns", "L3 cache random access latency"),
            metric("Memory R-Lat", round2(mem_latency), "ns", "Main memory random access latency"),
        ],
        run_count: fp32_single.values.len(),
        warmup_count: WARMUP,
        median_raw: round2(fp32_single.median),
        variance: round2(fp32_single.std_dev),
        confidence_interval: [round2(ci[0]), round2(ci[1])],
        error: None,
    })
}

fn metric(name: &str, value: f64, unit: &str, description: &str) -> BenchmarkMetric {
    BenchmarkMetric {
        name: name.to_string(),
        value,
        unit: unit.to_string(),
        description: description.to_string(),
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
