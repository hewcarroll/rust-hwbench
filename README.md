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