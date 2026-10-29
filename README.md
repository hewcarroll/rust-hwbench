# rust-hwbench

A small, dependency-light hardware benchmark **library + CLI** for Rust, covering
CPU, GPU, system RAM, and GPU VRAM. Built on [`wgpu`](https://crates.io/crates/wgpu)
(cross-vendor, cross-platform GPU compute) and [`rayon`](https://crates.io/crates/rayon).

It returns **raw, comparable measurements** (GFLOPS, GB/s, ns) with summary
statistics. An opinionated 0–10000 score is available but **opt-in**.

## What it measures

| Component | Metrics |
|-----------|---------|
| **CPU**   | FP32 / FP64 / INT32 single- & multi-core GEMM throughput, multi-thread scaling, L1/L2/L3/main-memory random-access latency |
| **GPU**   | FP32 / FP16 / INT32 compute throughput (compute shaders); FP16 path is hardened against driver hangs |
| **RAM**   | STREAM Copy / Scale / Add / Triad bandwidth |
| **VRAM**  | internal (GPU↔GPU), upload (host→device), download (device→host) bandwidth |
| **NPU**   | honest detection stub (reports "no NPU" rather than faking it with GPU compute) |

Plus `system::detect()` for CPU model/cores/frequency, total RAM, best-GPU
selection (discrete > integrated > virtual > software), and OS.

## CLI

```sh
cargo install --path .          # or: cargo run --release --
rust-hwbench                    # system info + full battery, table output
rust-hwbench cpu ram            # only the CPU and RAM benchmarks
rust-hwbench --score            # attach the opt-in 0–10000 calibrated score
rust-hwbench --json             # machine-readable JSON
```

GPU/VRAM gracefully report "skipped" when no adapter is available (e.g. headless
CI), so the CPU/RAM paths still run.

## Library

```rust
use rust_hwbench::{benchmarks::cpu, CpuConfig, score};

let mut result = cpu::run(CpuConfig::default())?;
result.score = score(&result);          // opt-in scoring
for m in &result.metrics {
    println!("{:<18} {} {}", m.name, m.value, m.unit);
}
# Ok::<(), String>(())
```

The GPU/VRAM benchmarks are `async` (they drive `wgpu`); run them with any
executor, e.g. `pollster::block_on(gpu::run(GpuConfig::default()))`.

### Scoring is opt-in

Every `run()` leaves `BenchmarkResultPayload.score == None`. To map a result onto
the 0–10000 scale, call [`score`] (uses the built-in anchor tables) or call
`calibrate(raw, &anchors)` with your own anchors. The anchor tables
(`benchmarks::cpu::CPU_ANCHORS`, etc.) are public.

### Statistics

Each result summarizes the per-run values of its primary metric (single-core FP32
for the CPU, FP32 compute for the GPU, Triad for RAM, internal bandwidth for VRAM),
in that metric's own unit:

- `medianRaw`: the median, averaging the two middle runs when the count is even
- `stdDev` and `cvPercent`: the sample standard deviation (n − 1), and the same
  thing as a percentage of the mean
- `medianCi95`: a 95% confidence interval **for the median**, from order
  statistics. It always contains the median of the runs. It covers the true
  median at least 95% of the time when the runs are independent draws from one
  stable distribution, whatever its shape; it does not account for drift,
  throttling that changes during the run, or differences between separate runs,
  which on a real machine can be larger than one run's interval. With fewer than
  6 runs no such interval exists; it is then the min–max range and
  `medianCiValid` is `false` (`true` means enough runs, not that the assumptions
  hold).
- Non-finite measurements (NaN, infinities) are left out and not counted.
- `startedAtUnixMs` and `durationMs`: filled in when the run is wrapped in `timed`
  (the CLI does this)

0.1.0 reported the interval of the *mean* next to the median, called the standard
deviation `variance`, and took VRAM's spread over durations in seconds. 0.2.0
fixes all three, and renames the fields accordingly.

## Notes & caveats

- Throughput figures are **relative** measurements for comparing machines and
  configurations, not vendor-spec peak numbers.
- The calibration anchors reference public hardware tiers (e.g. i3…i9,
  RTX 3050…4090, DDR4/DDR5) and encode an opinion about scoring — treat the
  score as a convenience, the raw metrics as the data.
- Requires a Rust toolchain with `wgpu` 24's platform prerequisites.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Unless you explicitly state otherwise,
any contribution intentionally submitted for inclusion in this crate by you, as
defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.