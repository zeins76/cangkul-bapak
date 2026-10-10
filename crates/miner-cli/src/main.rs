use clap::{Parser, Subcommand};
use engine_cpu::{AtomicBoolCancelCheck, EngineRange, MinerEngine};
use miner_service::{run, ServiceConfig};
use primitive_types::U512;
use rand::RngCore;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// CLI defaults
const DEFAULT_GPU_BATCH_SIZE: u32 = 1_000_000;
const DEFAULT_CUDA_BATCH_SIZE: u32 = 32_000_000;
const DEFAULT_CPU_BATCH_SIZE: u64 = 10_000;

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the mining service
    Serve {
        /// Coin yang ditambang (saat ini hanya quantus)
        #[arg(long, env = "BAPAK_COIN", default_value = "quantus", value_parser = ["quantus"])]
        coin: String,

        /// Alamat ladang (pool) atau node.
        /// Pool stratum: stratum+tcp://qtc.kryptex.network:7049.
        /// Node Quantus (QUIC): 103.10.20.30:9833.
        /// Bisa ditulis --ladang, --pool, atau --node-addr.
        #[arg(
            long,
            visible_aliases = ["ladang", "pool"],
            env = "MINER_NODE_ADDR",
            default_value = "127.0.0.1:9833"
        )]
        node_addr: String,

        /// Alamat wallet QTC kamu, dipakai sebagai identitas di ladang (pool).
        /// Tidak bisa dipakai bersama --auth-token / --auth-token-file.
        #[arg(
            long,
            env = "BAPAK_WALLET",
            conflicts_with_all = ["auth_token", "auth_token_file"]
        )]
        wallet: Option<String>,

        /// Password pool stratum (biasanya x)
        #[arg(short = 'p', long = "pass", env = "BAPAK_PASS", default_value = "x")]
        pool_pass: String,

        /// Shared auth token from the node's `miner-auth-token` file
        /// (`<base-path>/chains/<chain>/miner-auth-token`). Prefer `--auth-token-file`.
        #[arg(long, env = "MINER_AUTH_TOKEN", conflicts_with = "auth_token_file")]
        auth_token: Option<String>,

        /// Path to the node's `miner-auth-token` file (trimmed). Preferred over
        /// `--auth-token` so the secret is not placed on the command line.
        #[arg(long, env = "MINER_AUTH_TOKEN_FILE", conflicts_with = "auth_token")]
        auth_token_file: Option<PathBuf>,

        /// SHA-256 fingerprint of the node's miner TLS certificate (64 hex chars).
        /// Prefer `--tls-cert-sha256-file` pointing at `miner-tls-cert-sha256`
        /// (the node also logs this fingerprint).
        #[arg(
            long,
            env = "MINER_TLS_CERT_SHA256",
            conflicts_with = "tls_cert_sha256_file"
        )]
        tls_cert_sha256: Option<String>,

        /// Path to the node's `miner-tls-cert-sha256` file.
        #[arg(
            long,
            env = "MINER_TLS_CERT_SHA256_FILE",
            conflicts_with = "tls_cert_sha256"
        )]
        tls_cert_sha256_file: Option<PathBuf>,

        /// Number of CPU worker threads to use for mining (default: auto-detect)
        #[arg(long = "cpu-workers", env = "MINER_CPU_WORKERS")]
        cpu_workers: Option<usize>,

        /// Number of GPU devices to use for mining (default: auto-detect)
        #[arg(long = "gpu-devices", env = "MINER_GPU_DEVICES")]
        gpu_devices: Option<usize>,

        /// GPU batch size in nonces - controls how often GPU checks for cancellation
        /// (default: 1000000, or 32000000 with --cuda-gpu)
        #[arg(long = "gpu-batch-size", env = "MINER_GPU_BATCH_SIZE", value_parser = clap::value_parser!(u32).range(1..))]
        gpu_batch_size: Option<u32>,

        /// CPU batch size in hashes - controls how often CPU checks for cancellation
        #[arg(long = "cpu-batch-size", env = "MINER_CPU_BATCH_SIZE", default_value_t = DEFAULT_CPU_BATCH_SIZE, value_parser = clap::value_parser!(u64).range(1..))]
        cpu_batch_size: u64,

        /// Port for Prometheus metrics HTTP endpoint (default: 9900)
        #[arg(
            long = "metrics-port",
            env = "MINER_METRICS_PORT",
            default_value_t = 9900
        )]
        metrics_port: u16,

        /// GPU throttle delay in milliseconds between batches (0 = no throttle)
        #[arg(
            long = "gpu-throttle-ms",
            env = "MINER_GPU_THROTTLE_MS",
            default_value_t = 0
        )]
        gpu_throttle_ms: u64,

        /// Allow integrated GPUs (APUs) even when discrete GPUs are available.
        /// By default, integrated GPUs are skipped when a discrete GPU is present
        /// to avoid resource contention and driver instability.
        #[arg(long = "allow-integrated", env = "MINER_ALLOW_INTEGRATED")]
        allow_integrated: bool,

        /// Use the native CUDA engine instead of wgpu/Vulkan (NVIDIA only)
        #[arg(long = "cuda-gpu", env = "MINER_CUDA_GPU")]
        cuda_gpu: bool,

        /// Enable verbose logging
        #[arg(short, long, env = "MINER_VERBOSE")]
        verbose: bool,

        /// Pakai pesan log teknis asli (bukan gaya bapak)
        #[arg(long = "log-normal", env = "BAPAK_LOG_NORMAL")]
        log_normal: bool,
    },

    /// Run a quick benchmark of the mining engines
    Benchmark {
        /// Number of CPU workers to use for benchmark
        #[arg(long = "cpu-workers", env = "MINER_CPU_WORKERS")]
        cpu_workers: Option<usize>,

        /// Number of GPU devices to use for benchmark
        #[arg(long = "gpu-devices", env = "MINER_GPU_DEVICES")]
        gpu_devices: Option<usize>,

        /// GPU batch size in nonces - controls how often GPU checks for cancellation
        /// (default: 1000000, or 32000000 with --cuda-gpu)
        #[arg(long = "gpu-batch-size", env = "MINER_GPU_BATCH_SIZE", value_parser = clap::value_parser!(u32).range(1..))]
        gpu_batch_size: Option<u32>,

        /// CPU batch size in hashes - controls how often CPU checks for cancellation
        #[arg(long = "cpu-batch-size", env = "MINER_CPU_BATCH_SIZE", default_value_t = DEFAULT_CPU_BATCH_SIZE, value_parser = clap::value_parser!(u64).range(1..))]
        cpu_batch_size: u64,

        /// Benchmark duration in seconds (default: 10)
        #[arg(short, long, default_value_t = 10)]
        duration: u64,

        /// Allow integrated GPUs (APUs) even when discrete GPUs are available
        #[arg(long = "allow-integrated", env = "MINER_ALLOW_INTEGRATED")]
        allow_integrated: bool,

        /// Use the native CUDA engine instead of wgpu/Vulkan (NVIDIA only)
        #[arg(long = "cuda-gpu", env = "MINER_CUDA_GPU")]
        cuda_gpu: bool,

        /// Enable verbose logging
        #[arg(short, long, env = "MINER_VERBOSE")]
        verbose: bool,

        /// Pakai pesan log teknis asli (bukan gaya bapak)
        #[arg(long = "log-normal", env = "BAPAK_LOG_NORMAL")]
        log_normal: bool,
    },
}

/// bapak: miner Quantus turunan quantus-miner (Apache-2.0).
///
/// Contoh: bapak --coin quantus --ladang stratum+tcp://qtc.kryptex.network:7049 --wallet <ALAMAT_QTC>.rig1
#[derive(Parser, Debug)]
#[command(name = "bapak", author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
}

#[tokio::main]
async fn main() {
    // `bapak --coin quantus --ladang ...` tanpa subcommand berarti `bapak serve ...`.
    let mut argv: Vec<String> = std::env::args().collect();
    if argv.len() > 1
        && argv[1].starts_with('-')
        && !matches!(argv[1].as_str(), "-h" | "--help" | "-V" | "--version")
    {
        argv.insert(1, "serve".to_string());
    }
    let args = Args::parse_from(argv);

    let Some(command) = args.command else {
        eprintln!("Error: perintah belum lengkap.");
        eprintln!(
            "Contoh: bapak --coin quantus --ladang stratum+tcp://qtc.kryptex.network:7049 \
             --wallet <ALAMAT_QTC>.rig1"
        );
        eprintln!("Uji kekuatan cangkul: bapak benchmark --gpu-devices 1 --cpu-workers 0");
        std::process::exit(1);
    };

    match command {
        Command::Serve {
            coin: _,
            node_addr,
            wallet,
            pool_pass,
            auth_token,
            auth_token_file,
            tls_cert_sha256,
            tls_cert_sha256_file,
            cpu_workers,
            gpu_devices,
            gpu_batch_size,
            cpu_batch_size,
            gpu_throttle_ms,
            metrics_port,
            allow_integrated,
            cuda_gpu,
            verbose,
            log_normal,
        } => {
            init_logger(verbose);
            miner_service::gaya::set_log_normal(log_normal);

            // stratum+tcp://... means a stratum pool (Kryptex, Suprnova, ...);
            // anything else is a Quantus node / QUIC pool address.
            let stratum = match miner_service::stratum::StratumConfig::parse_url(
                &node_addr,
                wallet.as_deref().unwrap_or(""),
                &pool_pass,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };

            let (node_socket, auth_token, tls_cert_sha256) = if stratum.is_some() {
                (
                    std::net::SocketAddr::from(([0, 0, 0, 0], 0)),
                    String::new(),
                    String::new(),
                )
            } else {
                let node_socket = match resolve_node_addr(&node_addr) {
                    Ok(addr) => addr,
                    Err(e) => {
                        eprintln!("Error: {e}");
                        std::process::exit(1);
                    }
                };
                let auth_token = match wallet {
                    Some(w) if !w.trim().is_empty() => Ok(w.trim().to_string()),
                    Some(_) => Err("--wallet kosong".to_string()),
                    None => resolve_auth_token(auth_token, auth_token_file),
                };
                let auth_token = match auth_token {
                    Ok(token) => token,
                    Err(e) => {
                        eprintln!("Error: {e}");
                        std::process::exit(1);
                    }
                };
                let tls_cert_sha256 =
                    match resolve_tls_cert_sha256(tls_cert_sha256, tls_cert_sha256_file) {
                        Ok(fp) => fp,
                        Err(e) => {
                            eprintln!("Error: {e}");
                            std::process::exit(1);
                        }
                    };
                // Fail closed on permanent misconfig before metrics/workers start.
                if let Err(e) = quic_transport::validate_auth_config(&auth_token, &tls_cert_sha256)
                {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
                (node_socket, auth_token, tls_cert_sha256)
            };

            if miner_service::gaya::log_normal() {
                log::info!("Starting external miner service...");
            } else {
                log::info!("🧑‍🌾 Bapak bangun, siap-siap ke ladang...");
            }

            // Start metrics HTTP server
            if let Err(e) = metrics::start_http_exporter(metrics_port).await {
                log::error!("Failed to start metrics exporter: {e:?}");
                std::process::exit(1);
            }
            log::info!(
                "Metrics available at http://0.0.0.0:{}/metrics",
                metrics_port
            );

            let config = ServiceConfig {
                stratum,
                node_addr: node_socket,
                auth_token,
                tls_cert_sha256,
                cpu_workers,
                gpu_devices,
                gpu_batch_size: resolve_gpu_batch_size(gpu_batch_size, cuda_gpu),
                cpu_batch_size,
                gpu_throttle_ms,
                allow_integrated,
                cuda_gpu,
            };

            if let Err(e) = run(config).await {
                log::error!("Miner service terminated with error: {e:?}");
                std::process::exit(1);
            }
        }

        Command::Benchmark {
            cpu_workers,
            gpu_devices,
            gpu_batch_size,
            cpu_batch_size,
            duration,
            allow_integrated,
            cuda_gpu,
            verbose,
            log_normal,
        } => {
            init_logger(verbose);
            miner_service::gaya::set_log_normal(log_normal);
            run_benchmark(
                cpu_workers,
                gpu_devices,
                resolve_gpu_batch_size(gpu_batch_size, cuda_gpu),
                cpu_batch_size,
                duration,
                allow_integrated,
                cuda_gpu,
            )
            .await;
        }
    }
}

fn resolve_gpu_batch_size(explicit: Option<u32>, cuda_gpu: bool) -> u32 {
    explicit.unwrap_or(if cuda_gpu {
        DEFAULT_CUDA_BATCH_SIZE
    } else {
        DEFAULT_GPU_BATCH_SIZE
    })
}

/// Resolve `host:port` (IP or hostname) to a socket address for QUIC.
fn resolve_node_addr(addr: &str) -> Result<std::net::SocketAddr, String> {
    use std::net::ToSocketAddrs;
    addr.to_socket_addrs()
        .map_err(|e| format!("alamat ladang/node tidak valid '{addr}': {e}"))?
        .next()
        .ok_or_else(|| format!("alamat ladang/node tidak ditemukan: {addr}"))
}

fn resolve_auth_token(
    auth_token: Option<String>,
    auth_token_file: Option<PathBuf>,
) -> Result<String, String> {
    resolve_required_secret(
        auth_token,
        auth_token_file,
        "--auth-token",
        "--auth-token-file",
        "isi --wallet <ALAMAT_QTC> untuk menambang di ladang (pool), atau \
         --auth-token-file <PATH> ke file miner-auth-token node untuk solo",
    )
}

fn resolve_tls_cert_sha256(value: Option<String>, file: Option<PathBuf>) -> Result<String, String> {
    resolve_required_secret(
        value,
        file,
        "--tls-cert-sha256",
        "--tls-cert-sha256-file",
        "TLS cert fingerprint required: pass --tls-cert-sha256-file <PATH> to the node's \
         miner-tls-cert-sha256 file (or --tls-cert-sha256 <HEX>; also printed in node logs)",
    )
}

fn resolve_required_secret(
    value: Option<String>,
    file: Option<PathBuf>,
    value_flag: &str,
    file_flag: &str,
    missing_msg: &str,
) -> Result<String, String> {
    if let Some(value) = value {
        let value = value.trim().to_string();
        if value.is_empty() {
            return Err(format!("{value_flag} is empty"));
        }
        return Ok(value);
    }
    if let Some(path) = file {
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read {file_flag} {}: {}", path.display(), e))?;
        let value = contents.trim().to_string();
        if value.is_empty() {
            return Err(format!("{file_flag} {} is empty", path.display()));
        }
        return Ok(value);
    }
    Err(missing_msg.into())
}

fn init_logger(verbose: bool) {
    if std::env::var("RUST_LOG").is_err() {
        // Filter out noisy wgpu/naga shader compilation logs
        let log_level = if verbose {
            "debug,miner=debug,gpu_engine=debug,cuda_engine=debug,engine_cpu=debug,wgpu=warn,wgpu_core=warn,wgpu_hal=warn,naga=warn"
        } else {
            "info,miner=info,gpu_engine=info,cuda_engine=info,wgpu=error,wgpu_core=error,wgpu_hal=error,naga=error"
        };
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();
}

async fn run_benchmark(
    cpu_workers: Option<usize>,
    gpu_devices: Option<usize>,
    gpu_batch_size: u32,
    cpu_batch_size: u64,
    duration: u64,
    allow_integrated: bool,
    cuda_gpu: bool,
) {
    let effective_cpu_workers = cpu_workers.unwrap_or_else(num_cpus::get);

    let (gpu_engine, effective_gpu_devices) = if cuda_gpu {
        match miner_service::resolve_cuda_configuration(gpu_devices, gpu_batch_size, 0) {
            Ok((engine, count)) => (engine, count),
            Err(e) => {
                eprintln!("❌ ERROR: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        match miner_service::resolve_gpu_configuration(
            gpu_devices,
            gpu_batch_size,
            0,
            allow_integrated,
        ) {
            Ok((engine, count)) => (engine, count),
            Err(e) => {
                eprintln!("❌ ERROR: {}", e);
                std::process::exit(1);
            }
        }
    };

    let total_workers = effective_cpu_workers + effective_gpu_devices;

    let bapak = !miner_service::gaya::log_normal();
    if bapak {
        println!("🧑‍🌾 Uji Kekuatan Cangkul Bapak");
        println!("==============================");
    } else {
        println!("🚀 Quantus Miner Benchmark");
        println!("==========================");
    }
    println!(
        "CPU Workers: {} (Available: {})",
        effective_cpu_workers,
        num_cpus::get()
    );
    println!("GPU Devices: {}", effective_gpu_devices);
    if effective_cpu_workers > 0 {
        println!("CPU batch size: {cpu_batch_size} hashes");
    }
    if effective_gpu_devices > 0 {
        println!("GPU batch size: {gpu_batch_size} nonces");
    }
    println!("Duration: {duration} seconds");
    println!();

    if total_workers == 0 {
        eprintln!("Error: No workers specified");
        std::process::exit(1);
    }

    // Create CPU engine
    let cpu_engine: Option<Arc<dyn MinerEngine>> = if effective_cpu_workers > 0 {
        Some(Arc::new(engine_cpu::FastCpuEngine::new(cpu_batch_size)))
    } else {
        None
    };

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let benchmark_start = Instant::now();

    // Random header hash for benchmark
    let mut header = [0u8; 32];
    rand::rng().fill_bytes(&mut header);
    let difficulty = U512::MAX; // High difficulty - no solutions expected

    let ref_engine = cpu_engine.as_ref().or(gpu_engine.as_ref()).unwrap();
    let ctx = ref_engine.prepare_context(header, difficulty);

    if bapak {
        println!("⛏️  Bapak mulai mencangkul...");
    } else {
        println!("⛏️  Starting benchmark...");
    }

    // Spawn worker threads
    let mut handles = Vec::new();
    let total_hashes = Arc::new(std::sync::Mutex::new(0u64));

    // Floor range widths at the old constants: engines still batch at the flag's
    // size internally, but tiny flags don't turn per-call harness overhead into
    // the measured quantity.
    let cpu_chunk = cpu_batch_size.max(10_000);
    let gpu_chunk = (gpu_batch_size as u64).max(1_000_000);

    for worker_id in 0..total_workers {
        let (engine, nonces_per_batch) = if worker_id < effective_cpu_workers {
            (cpu_engine.as_ref().unwrap().clone(), cpu_chunk)
        } else {
            (gpu_engine.as_ref().unwrap().clone(), gpu_chunk)
        };

        let ctx = ctx.clone();
        let cancel = cancel_flag.clone();
        let hashes = total_hashes.clone();
        let start = benchmark_start;

        let handle = thread::spawn(move || {
            let stride = U512::from(1_000_000_000_000u64);
            let mut current = U512::from(worker_id as u64).saturating_mul(stride);
            let step = U512::from(nonces_per_batch);

            loop {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }

                let worker_range = EngineRange {
                    start: current,
                    end: current
                        .saturating_add(step)
                        .saturating_sub(U512::from(1u64)),
                };

                let cancel_check = AtomicBoolCancelCheck(&cancel);
                let result = engine.search_range(&ctx, worker_range, &cancel_check);

                match result {
                    engine_cpu::EngineStatus::Found { hash_count, .. }
                    | engine_cpu::EngineStatus::Exhausted { hash_count }
                    | engine_cpu::EngineStatus::Cancelled { hash_count }
                    | engine_cpu::EngineStatus::DeviceLost { hash_count } => {
                        *hashes.lock().unwrap() += hash_count;
                    }
                    engine_cpu::EngineStatus::Running { .. } => {}
                }

                if matches!(result, engine_cpu::EngineStatus::DeviceLost { .. }) {
                    break;
                }

                current = current.saturating_add(step);

                if start.elapsed() >= Duration::from_secs(duration) {
                    break;
                }
            }

            engine_gpu::GpuEngine::clear_worker_resources();
            engine_cuda::CudaEngine::clear_worker_resources();
        });

        handles.push(handle);
    }

    // Progress updates
    let mut last_update = Instant::now();

    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;

        if benchmark_start.elapsed() >= Duration::from_secs(duration) {
            cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            break;
        }

        if last_update.elapsed() >= Duration::from_secs(1) {
            let current = *total_hashes.lock().unwrap();
            let elapsed = benchmark_start.elapsed().as_secs_f64();
            if current > 0 {
                let rate = current as f64 / elapsed;
                if bapak {
                    println!(
                        "⏱️  {:.1}s - kekuatan cangkul {} H/s",
                        elapsed,
                        format_hash_rate(rate)
                    );
                } else {
                    println!("⏱️  {:.1}s - {} H/s", elapsed, format_hash_rate(rate));
                }
            }
            last_update = Instant::now();
        }
    }

    // Wait for threads
    for handle in handles {
        let _ = handle.join();
    }

    let total_elapsed = benchmark_start.elapsed();
    let final_hashes = *total_hashes.lock().unwrap();
    let avg_rate = final_hashes as f64 / total_elapsed.as_secs_f64();

    println!();
    if bapak {
        println!("📊 Hasil Uji Cangkul");
    } else {
        println!("📊 Benchmark Results");
    }
    println!("===================");
    println!("Total time: {:.2}s", total_elapsed.as_secs_f64());
    println!("Total hashes: {}", final_hashes);
    if bapak {
        println!(
            "Kekuatan cangkul rata-rata: {} H/s",
            format_hash_rate(avg_rate)
        );
    } else {
        println!("Average rate: {} H/s", format_hash_rate(avg_rate));
    }

    if total_workers > 1 {
        let per_worker = avg_rate / total_workers as f64;
        println!("Per-worker: {} H/s", format_hash_rate(per_worker));
    }

    if bapak {
        println!("✅ Bapak selesai uji cangkul!");
    } else {
        println!("✅ Benchmark completed!");
    }
}

fn format_hash_rate(rate: f64) -> String {
    if rate >= 1_000_000.0 {
        format!("{:.2}M", rate / 1_000_000.0)
    } else if rate >= 1_000.0 {
        format!("{:.2}K", rate / 1_000.0)
    } else {
        format!("{:.0}", rate)
    }
}
