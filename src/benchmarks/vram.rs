use std::time::Instant;
use super::{VramConfig, BenchmarkResultPayload, BenchmarkMetric};

const WARMUP: usize = 2;
const RUNS: usize = 5;
const MIN_RUNS: usize = 5;
const MAX_RUNS: usize = 100_000;
const TARGET_SECS: f64 = 2.0;

// Calibration anchors: (GB/s internal bandwidth, score)
pub const VRAM_ANCHORS: &[(f64, f64)] = &[
    (50.0,   1000.0),   // DDR4 integrated GPU
    (200.0,  3000.0),   // GTX 1660 class
    (400.0,  5000.0),   // RTX 3070 class
    (700.0,  7500.0),   // RTX 4080 class
    (1000.0, 10000.0),  // RTX 4090 class
];

const COPY_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> src : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&src) {
        dst[idx] = src[idx];
    }
}
"#;

pub async fn run(config: VramConfig) -> Result<BenchmarkResultPayload, String> {
    let buffer_mb = config.buffer_size_mb.unwrap_or(128);
    let buffer_bytes = (buffer_mb * 1024 * 1024) as u64;

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
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("VRAM Benchmark"),
            ..Default::default()
        }, None)
        .await
        .map_err(|e| format!("Failed to create device: {e}"))?;

    // --- Internal GPU->GPU bandwidth ---
    let buf_src = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Src"),
        size: buffer_bytes,
        // COPY_SRC: the device->host test reads this buffer back.
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dst"),
        size: buffer_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Fill source with data
    let fill_data = vec![1.0f32; buffer_bytes as usize / 4];
    queue.write_buffer(&buf_src, 0, bytemuck::cast_slice(&fill_data));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("VRAM Copy Shader"),
        source: wgpu::ShaderSource::Wgsl(COPY_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("VRAM Copy Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_dst.as_entire_binding() },
        ],
    });

    // vec4<f32> = 16 bytes per element
    let element_count = buffer_bytes / 16;
    let dispatch_count = ((element_count + 255) / 256) as u32;

    let mut internal_durations: Vec<f64> = Vec::new();
    let internal_start = Instant::now();
    let mut run_idx = 0usize;

    loop {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count, 1, 1);
        }
        let start = Instant::now();
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        let elapsed = start.elapsed().as_secs_f64();

        if run_idx >= WARMUP {
            internal_durations.push(elapsed);
        }
        run_idx += 1;

        let done = internal_durations.len() >= MIN_RUNS && internal_start.elapsed().as_secs_f64() >= TARGET_SECS;
        if done || internal_durations.len() >= MAX_RUNS {
            break;
        }
    }

    internal_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let internal_median = internal_durations[internal_durations.len() / 2];
    // Read + Write = 2x buffer
    let internal_gbps = (buffer_bytes as f64 * 2.0) / internal_median / 1e9;

    // --- Upload bandwidth (CPU -> GPU) ---
    let upload_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Upload"),
        size: buffer_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let upload_data = vec![0u8; buffer_bytes as usize];
    let mut upload_durations: Vec<f64> = Vec::new();

    for run_idx in 0..(WARMUP + RUNS) {
        let start = Instant::now();
        queue.write_buffer(&upload_buf, 0, &upload_data);
        device.poll(wgpu::Maintain::Wait);
        let elapsed = start.elapsed().as_secs_f64();
        if run_idx >= WARMUP {
            upload_durations.push(elapsed);
        }
    }

    upload_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let upload_median = upload_durations[upload_durations.len() / 2];
    let upload_gbps = buffer_bytes as f64 / upload_median / 1e9;

    // --- Download bandwidth (GPU -> CPU) ---
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback"),
        size: buffer_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut download_durations: Vec<f64> = Vec::new();

    for run_idx in 0..(WARMUP + RUNS) {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&buf_src, 0, &readback_buf, 0, buffer_bytes);

        let start = Instant::now();
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| "Map channel closed")?
            .map_err(|e| format!("Map failed: {e}"))?;

        let elapsed = start.elapsed().as_secs_f64();
        readback_buf.unmap();

        if run_idx >= WARMUP {
            download_durations.push(elapsed);
        }
    }

    download_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let download_median = download_durations[download_durations.len() / 2];
    let download_gbps = buffer_bytes as f64 / download_median / 1e9;

    // Stats for internal bandwidth
    let mean = internal_durations.iter().sum::<f64>() / internal_durations.len() as f64;
    let variance = internal_durations.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / internal_durations.len() as f64;
    let std_dev = variance.sqrt();

    let gbps_values: Vec<f64> = internal_durations
        .iter()
        .map(|d| (buffer_bytes as f64 * 2.0) / d / 1e9)
        .collect();
    let mean_gbps = gbps_values.iter().sum::<f64>() / gbps_values.len() as f64;
    let gbps_std = (gbps_values.iter().map(|g| (g - mean_gbps).powi(2)).sum::<f64>() / gbps_values.len() as f64).sqrt();
    let margin = 1.96 * gbps_std / (gbps_values.len() as f64).sqrt();


    Ok(BenchmarkResultPayload {
        component_type: "GPU_MEMORY".to_string(),
        score: None,
        metrics: vec![
            BenchmarkMetric {
                name: "Internal Bandwidth".to_string(),
                value: round2(internal_gbps),
                unit: "GB/s".to_string(),
                description: "GPU-to-GPU memory copy via compute shader".to_string(),
            },
            BenchmarkMetric {
                name: "Upload Bandwidth".to_string(),
                value: round2(upload_gbps),
                unit: "GB/s".to_string(),
                description: "CPU-to-GPU write bandwidth".to_string(),
            },
            BenchmarkMetric {
                name: "Download Bandwidth".to_string(),
                value: round2(download_gbps),
                unit: "GB/s".to_string(),
                description: "GPU-to-CPU readback bandwidth".to_string(),
            },
            BenchmarkMetric {
                name: "Buffer Size".to_string(),
                value: buffer_mb as f64,
                unit: "MB".to_string(),
                description: "Test buffer size".to_string(),
            },
        ],
        run_count: internal_durations.len(),
        warmup_count: WARMUP,
        median_raw: round2(internal_gbps),
        variance: round2(std_dev),
        confidence_interval: [round2(mean_gbps - margin), round2(mean_gbps + margin)],
        error: None,
    })
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
