use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use super::{summarize, BenchmarkMetric, BenchmarkResultPayload, GpuConfig, Summary};

const WARMUP: usize = 2;
const MIN_RUNS: usize = 5;
const MAX_RUNS: usize = 100_000;
const TARGET_SECS: f64 = 2.0; // per kernel; the battery runs two kernels

// Calibration anchors: (FP32 TFLOPS, score)
pub const GPU_ANCHORS: &[(f64, f64)] = &[
    (0.5,   1000.0),   // basic iGPU
    (5.0,   3000.0),   // RTX 3050 class
    (15.0,  5000.0),   // RTX 3070 / Arc A770 class
    (35.0,  7500.0),   // RTX 4080 class
    (80.0, 10000.0),   // RTX 4090 class
];

// Each shader loads its two operands once and does `iterations` multiply-adds
// (2 ops each) in a tight ALU loop, so the test is compute-bound.
const SHADER_F32: &str = r#"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
struct Params { n: u32, iterations: u32 }
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    let av = a[idx];
    let bv = b[idx];
    var acc = 0.0;
    for (var it = 0u; it < params.iterations; it++) {
        acc = acc + av * bv;
    }
    c[idx] = acc;
}
"#;

const SHADER_I32: &str = r#"
@group(0) @binding(0) var<storage, read> a : array<i32>;
@group(0) @binding(1) var<storage, read> b : array<i32>;
@group(0) @binding(2) var<storage, read_write> c : array<i32>;
struct Params { n: u32, iterations: u32 }
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    let av = a[idx];
    let bv = b[idx];
    var acc : i32 = 0;
    for (var it = 0u; it < params.iterations; it++) {
        acc = acc + av * bv;
    }
    c[idx] = acc;
}
"#;

const SHADER_F16: &str = r#"
enable f16;
@group(0) @binding(0) var<storage, read> a : array<f16>;
@group(0) @binding(1) var<storage, read> b : array<f16>;
@group(0) @binding(2) var<storage, read_write> c : array<f16>;
struct Params { n: u32, iterations: u32 }
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    let av = a[idx];
    let bv = b[idx];
    var acc : f16 = f16(0.0);
    for (var it = 0u; it < params.iterations; it++) {
        acc = acc + av * bv;
    }
    c[idx] = acc;
}
"#;

type KernelStats = Summary;

pub async fn run(config: GpuConfig) -> Result<BenchmarkResultPayload, String> {
    let iterations = config.iterations.unwrap_or(4096);
    let workgroup_size: u32 = config.workgroup_size.unwrap_or(256);
    let n: u32 = 1024 * 1024; // 1M elements (proven stable)

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .ok_or("No GPU adapter found")?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rust-hwbench GPU Benchmark"),
                ..Default::default()
            },
            None,
        )
        .await
        .map_err(|e| format!("Failed to create device: {e}"))?;

    let a_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) % 1.0 + 0.5).collect();
    let b_f32: Vec<f32> = (0..n).map(|i| ((i + 7) as f32 * 0.0013) % 1.0 + 0.5).collect();
    let a_i32: Vec<i32> = (0..n).map(|i| (i as i32 % 7) + 1).collect();
    let b_i32: Vec<i32> = (0..n).map(|i| (i as i32 % 13) + 1).collect();

    let fp32 = run_kernel(&device, &queue, SHADER_F32, n, iterations, workgroup_size, bytemuck::cast_slice(&a_f32), bytemuck::cast_slice(&b_f32), 4);
    let int32 = run_kernel(&device, &queue, SHADER_I32, n, iterations, workgroup_size, bytemuck::cast_slice(&a_i32), bytemuck::cast_slice(&b_i32), 4);

    // FP16 runs on its own device with a hard timeout, so a driver hang on the
    // f16 path can't freeze the app — it just reports "unavailable".
    let fp16 = run_f16(&adapter, n, iterations, workgroup_size).await;


    let mut metrics = vec![metric(
        "FP32 Compute",
        round2(fp32.median),
        "TFLOPS",
        "FP32 multiply-add throughput via compute shader",
    )];
    match &fp16 {
        Some(s) => {
            metrics.push(metric("FP16 Compute", round2(s.median), "TFLOPS", "Half-precision FP16 throughput"));
            if fp32.median > 0.0 {
                metrics.push(metric("FP16:FP32", round2(s.median / fp32.median), "x", "Half- vs single-precision speedup"));
            }
        }
        None => metrics.push(metric("FP16 Compute", 0.0, "n/a", "FP16 unavailable on this adapter")),
    }
    metrics.push(metric("INT32 Compute", round2(int32.median), "TOPS", "INT32 multiply-add throughput"));

    if fp32.n == 0 {
        return Err("no valid FP32 compute measurements".to_string());
    }

    Ok(BenchmarkResultPayload {
        component_type: "GPU".to_string(),
        score: None,
        score_ci_95: None,
        metrics,
        run_count: fp32.n,
        warmup_count: WARMUP,
        median_raw: round2(fp32.median),
        std_dev: round2(fp32.std_dev),
        cv_percent: (fp32.cv_percent * 10.0).round() / 10.0,
        median_ci_95: [round2(fp32.median_ci_95[0]), round2(fp32.median_ci_95[1])],
        median_ci_valid: fp32.ci_valid,
        started_at_unix_ms: 0,
        duration_ms: 0,
        error: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_kernel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader_src: &str,
    n: u32,
    iterations: u32,
    workgroup_size: u32,
    a_bytes: &[u8],
    b_bytes: &[u8],
    elem_bytes: u64,
) -> KernelStats {
    let buf_size = (n as u64) * elem_bytes;

    let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("A"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("B"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params_data = [n, iterations];
    let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buf_a, 0, a_bytes);
    queue.write_buffer(&buf_b, 0, b_bytes);
    queue.write_buffer(&buf_params, 0, bytemuck::cast_slice(&params_data[..]));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("GPU Bench Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            bgl_entry(0, true),
            bgl_entry(1, true),
            bgl_entry(2, false),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("GPU Benchmark Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let dispatch_count = (n + workgroup_size - 1) / workgroup_size;

    let mut durations: Vec<f64> = Vec::new();
    let bench_start = Instant::now();
    let mut run_idx = 0usize;

    loop {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Bench Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bench Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count, 1, 1);
        }

        let start = Instant::now();
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        let elapsed = start.elapsed().as_secs_f64();

        if run_idx >= WARMUP {
            durations.push(elapsed);
        }
        run_idx += 1;

        let done = durations.len() >= MIN_RUNS && bench_start.elapsed().as_secs_f64() >= TARGET_SECS;
        if done || durations.len() >= MAX_RUNS {
            break;
        }
    }

    // ops/sec in tera-ops (2 ops per multiply-add iteration, per element)
    let total_ops = (n as f64) * (iterations as f64) * 2.0;
    let tput: Vec<f64> = durations.iter().map(|d| total_ops / d / 1e12).collect();
    summarize(&tput)
}

/// FP16 compute test, fully hang-proofed:
/// - runs on its OWN device (the FP32/INT32 results already ran on a separate
///   device, so they're unaffected if this one wedges),
/// - drives completion with a non-blocking poll + wall-clock timeout instead of
///   the blocking `poll(Wait)` that can hang forever on a driver fault,
/// - treats the first batch as a canary: if it doesn't finish in time, FP16 is
///   abandoned and reported as unavailable.
async fn run_f16(
    adapter: &wgpu::Adapter,
    n: u32,
    iterations: u32,
    workgroup_size: u32,
) -> Option<KernelStats> {
    if !adapter.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rust-hwbench FP16"),
                required_features: wgpu::Features::SHADER_F16,
                ..Default::default()
            },
            None,
        )
        .await
        .ok()?;

    let buf_size = (n as u64) * 2; // f16 = 2 bytes
    let a_f16: Vec<u16> = vec![0x3C00u16; n as usize]; // 1.0
    let b_f16: Vec<u16> = vec![0x3800u16; n as usize]; // 0.5

    let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("A"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("B"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params_data = [n, iterations];
    let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&a_f16));
    queue.write_buffer(&buf_b, 0, bytemuck::cast_slice(&b_f16));
    queue.write_buffer(&buf_params, 0, bytemuck::cast_slice(&params_data[..]));

    // A device can advertise SHADER_F16 while the shader compiler still rejects
    // `enable f16;` (wgpu 24's Naga does). Catch that instead of panicking, and
    // report FP16 as unavailable.
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FP16 Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_F16.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            bgl_entry(0, true),
            bgl_entry(1, true),
            bgl_entry(2, false),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FP16 Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if device.pop_error_scope().await.is_some() {
        return None;
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let dispatch_count = (n + workgroup_size - 1) / workgroup_size;
    const BATCH: u32 = 16; // dispatches per submit — amortizes poll granularity

    let mut tput: Vec<f64> = Vec::new();
    let overall = Instant::now();
    let mut first = true;

    loop {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            for _ in 0..BATCH {
                pass.dispatch_workgroups(dispatch_count, 1, 1);
            }
        }

        let start = Instant::now();
        queue.submit(std::iter::once(encoder.finish()));
        // Hard timeout: if the GPU/driver wedges, bail instead of hanging.
        if !poll_until_done(&device, &queue, Duration::from_secs(4)) {
            return None;
        }
        let elapsed = start.elapsed().as_secs_f64();

        if first {
            first = false; // canary / warmup batch, discarded
        } else if elapsed > 0.0 {
            let ops = (BATCH as f64) * (n as f64) * (iterations as f64) * 2.0;
            tput.push(ops / elapsed / 1e12);
        }

        let enough = tput.len() >= MIN_RUNS && overall.elapsed().as_secs_f64() >= TARGET_SECS;
        if enough || overall.elapsed() > Duration::from_secs(8) || tput.len() >= MAX_RUNS {
            break;
        }
    }

    if tput.is_empty() {
        return None;
    }
    Some(summarize(&tput))
}

/// Non-blocking wait for the queue's submitted work with a hard timeout.
/// Returns false if the deadline passes (the GPU is hung / not responding).
fn poll_until_done(device: &wgpu::Device, queue: &wgpu::Queue, timeout: Duration) -> bool {
    let done = Arc::new(AtomicBool::new(false));
    let flag = done.clone();
    queue.on_submitted_work_done(move || flag.store(true, Ordering::Release));

    let start = Instant::now();
    loop {
        device.poll(wgpu::Maintain::Poll);
        if done.load(Ordering::Acquire) {
            return true;
        }
        if start.elapsed() > timeout {
            return false;
        }
        std::thread::sleep(Duration::from_micros(100));
    }
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


fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
