use serde::Serialize;
use sysinfo::System;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemInfo {
    pub cpu_model: String,
    pub cpu_cores_physical: usize,
    pub cpu_cores_logical: usize,
    pub cpu_frequency_mhz: u64,
    pub ram_total_gb: f64,
    pub gpu_name: String,
    pub gpu_backend: String,
    pub gpu_is_discrete: bool,
    pub os: String,
}

pub async fn detect() -> Result<SystemInfo, String> {
    let mut sys = System::new_all();
    sys.refresh_all();

    // CPU
    let cpu_model = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".to_string());
    let cpu_cores_physical = sys.physical_core_count().unwrap_or(0);
    let cpu_cores_logical = sys.cpus().len();
    let cpu_frequency_mhz = sys.cpus().first().map(|c| c.frequency()).unwrap_or(0);

    // RAM
    let ram_total_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

    // GPU via wgpu — pick the best *real* adapter. Prefer discrete > integrated >
    // virtual > other, and rank software/CPU fallbacks (e.g. llvmpipe) last so a
    // real GPU is reported when one exists.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let best = adapters.iter().max_by_key(|a| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 4u8,
        wgpu::DeviceType::IntegratedGpu => 3,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 1,
        wgpu::DeviceType::Cpu => 0,
    });

    let (gpu_name, gpu_backend, gpu_is_discrete) = if let Some(adapter) = best {
        let info = adapter.get_info();
        let name = if info.name.trim().is_empty() {
            "Unknown GPU".to_string()
        } else {
            info.name.trim().to_string()
        };
        (
            name,
            format!("{:?}", info.backend),
            info.device_type == wgpu::DeviceType::DiscreteGpu,
        )
    } else {
        ("No GPU detected".to_string(), "None".to_string(), false)
    };

    // OS
    let os = format!(
        "{} {}",
        System::name().unwrap_or_default(),
        System::os_version().unwrap_or_default()
    )
    .trim()
    .to_string();

    Ok(SystemInfo {
        cpu_model,
        cpu_cores_physical,
        cpu_cores_logical,
        cpu_frequency_mhz,
        ram_total_gb,
        gpu_name,
        gpu_backend,
        gpu_is_discrete,
        os,
    })
}
