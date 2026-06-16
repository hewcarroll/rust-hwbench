//! rust-hwbench CLI — run the hardware benchmark battery from the terminal.
//!
//! Usage:
//!   rust-hwbench [TARGETS...] [--score] [--json]
//!
//! TARGETS: any of `system cpu gpu ram vram npu all` (default: `all`, which
//! runs system detection plus every benchmark). `--score` attaches the opt-in
//! 0–10000 calibrated score; `--json` emits machine-readable JSON.

use rust_hwbench::{
    benchmarks::{cpu, gpu, npu, ram, vram},
    score, BenchmarkResultPayload, CpuConfig, GpuConfig, NpuConfig, RamConfig, VramConfig,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    let do_score = args.iter().any(|a| a == "--score");

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    let mut targets: Vec<String> = args
        .iter()
        .filter(|a| !a.starts_with('-'))
        .cloned()
        .collect();
    if targets.is_empty() || targets.iter().any(|t| t == "all") {
        targets = vec!["system", "cpu", "gpu", "ram", "vram", "npu"]
            .into_iter()
            .map(String::from)
            .collect();
    }

    let want = |name: &str| targets.iter().any(|t| t == name);

    let system = if want("system") {
        pollster::block_on(rust_hwbench::detect_system()).ok()
    } else {
        None
    };

    let mut results: Vec<BenchmarkResultPayload> = Vec::new();
    let mut run = |label: &str, res: Result<BenchmarkResultPayload, String>| match res {
        Ok(mut r) => {
            if do_score {
                r.score = score(&r);
            }
            results.push(r);
        }
        Err(e) => eprintln!("[{label}] skipped: {e}"),
    };

    if want("cpu") {
        run("cpu", cpu::run(CpuConfig::default()));
    }
    if want("gpu") {
        run("gpu", pollster::block_on(gpu::run(GpuConfig::default())));
    }
    if want("ram") {
        run("ram", ram::run(RamConfig::default()));
    }
    if want("vram") {
        run("vram", pollster::block_on(vram::run(VramConfig::default())));
    }
    if want("npu") {
        run("npu", npu::run(NpuConfig::default()));
    }

    if json {
        print_json(system.as_ref(), &results);
    } else {
        print_table(system.as_ref(), &results);
    }
}

fn print_help() {
    println!(
        "rust-hwbench — CPU/GPU/RAM/VRAM hardware benchmarks\n\n\
         USAGE:\n  rust-hwbench [TARGETS...] [--score] [--json]\n\n\
         TARGETS:\n  system cpu gpu ram vram npu all   (default: all)\n\n\
         FLAGS:\n  --score   attach the opt-in 0-10000 calibrated score\n  \
         --json    emit JSON instead of a table\n  -h, --help"
    );
}

fn print_table(system: Option<&rust_hwbench::SystemInfo>, results: &[BenchmarkResultPayload]) {
    if let Some(s) = system {
        println!("System");
        println!("  CPU   {} ({}c/{}t @ {} MHz)", s.cpu_model, s.cpu_cores_physical, s.cpu_cores_logical, s.cpu_frequency_mhz);
        println!("  RAM   {:.1} GB", s.ram_total_gb);
        println!("  GPU   {} [{}]{}", s.gpu_name, s.gpu_backend, if s.gpu_is_discrete { " (discrete)" } else { "" });
        println!("  OS    {}", s.os);
        println!();
    }
    for r in results {
        match r.score {
            Some(sc) => println!("{}  (score {})", r.component_type, sc),
            None => println!("{}", r.component_type),
        }
        for m in &r.metrics {
            println!("  {:<20} {:>10} {}", m.name, m.value, m.unit);
        }
        if r.run_count > 0 {
            println!(
                "  [{} runs, median {:.2}, ±{:.2}, 95% CI {:.2}..{:.2}]",
                r.run_count, r.median_raw, r.variance, r.confidence_interval[0], r.confidence_interval[1]
            );
        }
        if let Some(err) = &r.error {
            println!("  note: {err}");
        }
        println!();
    }
}

fn print_json(system: Option<&rust_hwbench::SystemInfo>, results: &[BenchmarkResultPayload]) {
    let out = serde_json::json!({
        "system": system,
        "results": results,
    });
    match serde_json::to_string_pretty(&out) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("json error: {e}"),
    }
}