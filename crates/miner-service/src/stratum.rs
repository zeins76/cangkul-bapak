//! Stratum client for Quantus pools that use the "login / job / submit"
//! JSON-RPC dialect (Kryptex, Suprnova, LuckyPool, ...).
//!
//! Protocol as observed on qtc.kryptex.network:7049 and
//! quantus.suprnova.cc:7071 (October 2026):
//!
//! ```text
//! miner -> pool  {"id":1,"method":"login","params":{"login":"WALLET.rig","pass":"x","agent":"bapak/.."}}
//! pool  -> miner {"id":1,"error":null,"result":{"id":"<session>","job":{JOB},"status":"OK"}}
//! pool  -> miner {"method":"job","params":{"clean_jobs":true,"job":{JOB}}}
//! miner -> pool  {"id":2,"method":"submit","params":{"id":"<session>","job_id":"..","nonce":"<128 hex>"}}
//! pool  -> miner {"id":2,"error":null,"result":{"status":"OK"}}
//!
//! JOB = {"job_id":"..","mining_hash":"<32-byte hex>","difficulty":<number>,
//!        "target":"<64-byte hex>","extranonce":"<hex>"}
//! ```
//!
//! The submitted nonce is the full 64-byte big-endian nonce and must start
//! with the pool's extranonce bytes. The share target is `U512::MAX /
//! difficulty`, the same rule the mining engines already use.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use engine_cpu::MinerEngine;
use primitive_types::U512;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::tcp::OwnedWriteHalf;
use tokio::net::TcpStream;

use crate::{gaya, EngineType, JobStopReason, WorkerPool};
use pow_core::format_hashrate;

const AGENT: &str = concat!("bapak/", env!("CARGO_PKG_VERSION"));
const LOGIN_TIMEOUT: Duration = Duration::from_secs(30);
const KEEPALIVE_EVERY: Duration = Duration::from_secs(60);
const REPORT_EVERY: Duration = Duration::from_secs(30);

/// Where and how to log in to a stratum pool.
#[derive(Clone, Debug)]
pub struct StratumConfig {
    pub host: String,
    pub port: u16,
    /// Pool login, usually `WALLET` or `WALLET.worker`.
    pub login: String,
    /// Pool password, usually `x`.
    pub pass: String,
}

impl StratumConfig {
    /// Parse `stratum+tcp://host:port`.
    ///
    /// Returns `Ok(None)` when `url` has no `scheme://` part, so the caller
    /// can treat it as a plain node address instead.
    pub fn parse_url(url: &str, login: &str, pass: &str) -> anyhow::Result<Option<Self>> {
        let Some((scheme, rest)) = url.split_once("://") else {
            return Ok(None);
        };
        match scheme.to_ascii_lowercase().as_str() {
            "stratum+tcp" | "stratum" | "tcp" => {}
            "stratum+ssl" | "stratum+tls" | "ssl" | "tls" => {
                bail!("koneksi SSL/TLS ke pool belum didukung, pakai stratum+tcp://host:port")
            }
            other => bail!("skema '{other}' tidak dikenal, pakai stratum+tcp://host:port"),
        }
        let rest = rest.trim_end_matches('/');
        let (host, port) = rest
            .rsplit_once(':')
            .ok_or_else(|| anyhow!("port pool belum diisi: {url}"))?;
        if host.is_empty() {
            bail!("alamat pool kosong: {url}");
        }
        let port: u16 = port
            .parse()
            .with_context(|| format!("port pool tidak valid: {port}"))?;
        let login = login.trim();
        if login.is_empty() {
            bail!("--wallet wajib diisi untuk menambang di pool stratum");
        }
        Ok(Some(Self {
            host: host.to_string(),
            port,
            login: login.to_string(),
            pass: pass.to_string(),
        }))
    }
}

/// The pool rejected our login. Retrying will not help.
#[derive(Debug)]
struct LoginRejected(String);

impl std::fmt::Display for LoginRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pool menolak login: {}", self.0)
    }
}

impl std::error::Error for LoginRejected {}

/// One unit of work from the pool.
#[derive(Clone, Debug)]
struct PoolJob {
    job_id: String,
    header: [u8; 32],
    difficulty: U512,
    extranonce: Vec<u8>,
}

fn parse_job(v: &Value) -> anyhow::Result<PoolJob> {
    let job_id = v
        .get("job_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("job tanpa job_id"))?
        .to_string();

    let mining_hash = v
        .get("mining_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("job tanpa mining_hash"))?;
    let header_bytes = hex::decode(mining_hash).context("mining_hash bukan hex")?;
    let header: [u8; 32] = header_bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("mining_hash harus 32 byte, dapat {}", header_bytes.len()))?;

    let difficulty = match v.get("difficulty") {
        Some(Value::Number(n)) => n.as_u64().map(U512::from).or_else(|| {
            n.as_f64()
                .filter(|f| f.is_finite() && *f >= 1.0)
                .map(|f| U512::from(f as u64))
        }),
        Some(Value::String(s)) => U512::from_dec_str(s).ok(),
        _ => None,
    };
    let difficulty = match difficulty {
        Some(d) if !d.is_zero() => d,
        _ => {
            let target_hex = v
                .get("target")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("job tanpa difficulty dan target"))?;
            let target_bytes = hex::decode(target_hex).context("target bukan hex")?;
            if target_bytes.is_empty() || target_bytes.len() > 64 {
                bail!("panjang target tidak valid: {} byte", target_bytes.len());
            }
            let target = U512::from_big_endian(&target_bytes);
            if target.is_zero() {
                bail!("target nol");
            }
            U512::MAX / target
        }
    };

    let extranonce = match v.get("extranonce").and_then(Value::as_str) {
        Some(s) if !s.is_empty() => hex::decode(s).context("extranonce bukan hex")?,
        _ => Vec::new(),
    };
    if extranonce.len() > 32 {
        bail!("extranonce terlalu panjang: {} byte", extranonce.len());
    }

    Ok(PoolJob {
        job_id,
        header,
        difficulty,
        extranonce,
    })
}

fn error_text(err: &Value) -> String {
    err.get("message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| err.to_string())
}

async fn send(wr: &mut OwnedWriteHalf, msg: &Value) -> anyhow::Result<()> {
    let mut line = serde_json::to_string(msg)?;
    line.push('\n');
    wr.write_all(line.as_bytes())
        .await
        .context("gagal kirim pesan ke pool")?;
    Ok(())
}

fn start_pool_job(worker_pool: &WorkerPool, job: &PoolJob) -> u64 {
    metrics::set_active_jobs(1);
    worker_pool.start_job_with_prefix(job.header, job.difficulty, Some(job.extranonce.clone()))
}

/// Share counters that survive reconnects.
#[derive(Default)]
struct Stats {
    accepted: u64,
    rejected: u64,
}

/// Connect to a stratum pool and mine until a permanent error occurs.
/// Reconnects with backoff when the connection drops.
pub async fn connect_and_mine(
    config: StratumConfig,
    cpu_engine: Option<Arc<dyn MinerEngine>>,
    gpu_engine: Option<Arc<dyn MinerEngine>>,
    cpu_workers: usize,
    gpu_devices: usize,
) -> anyhow::Result<()> {
    let worker_pool = WorkerPool::new(cpu_engine, gpu_engine, cpu_workers, gpu_devices);
    metrics::set_effective_cpus(num_cpus::get() as i64);
    metrics::set_workers(worker_pool.worker_count() as i64);
    metrics::set_cpu_workers(worker_pool.cpu_worker_count() as i64);
    metrics::set_gpu_devices(worker_pool.gpu_worker_count() as i64);

    let mut stats = Stats::default();
    let mut delay = Duration::from_secs(2);
    const MAX_DELAY: Duration = Duration::from_secs(60);

    loop {
        let addr = format!("{}:{}", config.host, config.port);
        if gaya::log_normal() {
            log::info!("⛏️ Connecting to stratum pool {addr}...");
        } else {
            log::info!("🌾 Bapak berangkat ke ladang {addr}...");
        }

        let mut logged_in = false;
        let err = run_session(&config, &worker_pool, &mut stats, &mut logged_in)
            .await
            .err()
            .unwrap_or_else(|| anyhow!("sesi berakhir"));
        worker_pool.stop_current_job(JobStopReason::ConnectionLost);
        metrics::set_active_jobs(0);

        if err.downcast_ref::<LoginRejected>().is_some() {
            log::error!("⛔ {err}");
            return Err(err);
        }
        if gaya::log_normal() {
            log::warn!("⛏️ Pool connection lost: {err}");
        } else {
            log::warn!("⚠️ Jalan ke ladang putus: {err}");
        }

        if logged_in {
            delay = Duration::from_secs(2);
        }
        if gaya::log_normal() {
            log::info!("⛏️ Reconnecting in {}s...", delay.as_secs());
        } else {
            log::info!(
                "⏳ Bapak coba berangkat lagi {} detik lagi...",
                delay.as_secs()
            );
        }
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(MAX_DELAY);
    }
}

async fn run_session(
    config: &StratumConfig,
    worker_pool: &WorkerPool,
    stats: &mut Stats,
    logged_in: &mut bool,
) -> anyhow::Result<()> {
    let addr = format!("{}:{}", config.host, config.port);
    let stream = tokio::time::timeout(Duration::from_secs(20), TcpStream::connect(&addr))
        .await
        .map_err(|_| anyhow!("timeout konek ke {addr}"))?
        .with_context(|| format!("gagal konek ke {addr}"))?;
    stream.set_nodelay(true).ok();
    let (rd, mut wr) = stream.into_split();
    let mut lines = BufReader::new(rd).lines();

    if gaya::log_normal() {
        log::info!("⛏️ Connected to {addr}, logging in as {}", config.login);
    } else {
        log::info!("🌾 Bapak sampai di ladang {addr}, lapor diri...");
    }

    let mut next_id: u64 = 1;
    let login_id = next_id;
    next_id += 1;
    send(
        &mut wr,
        &json!({
            "id": login_id,
            "method": "login",
            "params": { "login": config.login, "pass": config.pass, "agent": AGENT }
        }),
    )
    .await?;

    let connected_at = Instant::now();
    let mut session: Option<String> = None;
    let mut current: Option<PoolJob> = None;
    let mut internal_job_id: u64 = 0;
    let mut pending: HashMap<u64, String> = HashMap::new();
    let mut last_send = Instant::now();
    let mut window_start = Instant::now();
    let mut window_hashes: u64 = 0;

    loop {
        if session.is_none() && connected_at.elapsed() > LOGIN_TIMEOUT {
            bail!(
                "pool tidak membalas login dalam {} detik",
                LOGIN_TIMEOUT.as_secs()
            );
        }

        // 1. Results from the GPU/CPU workers.
        let rx = worker_pool.result_receiver().clone();
        let polled =
            tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_millis(10))).await;
        if let Ok(Ok(result)) = polled {
            window_hashes += result.hash_count;
            match result.engine_type {
                EngineType::Cpu => metrics::record_cpu_hashes(result.hash_count),
                EngineType::Gpu => metrics::record_gpu_hashes(result.hash_count),
            }
            if result.job_id == internal_job_id {
                if let (Some(candidate), Some(job), Some(sid)) =
                    (result.candidate, current.as_ref(), session.as_ref())
                {
                    let nonce_hex = hex::encode(candidate.nonce.to_big_endian());
                    let id = next_id;
                    next_id += 1;
                    send(
                        &mut wr,
                        &json!({
                            "id": id,
                            "method": "submit",
                            "params": { "id": sid, "job_id": job.job_id, "nonce": nonce_hex }
                        }),
                    )
                    .await?;
                    last_send = Instant::now();
                    pending.insert(id, job.job_id.clone());
                    if gaya::log_normal() {
                        log::info!("📤 Share submitted for job {}", job.job_id);
                    } else {
                        log::info!("✨ Bapak nemu emas! Dikirim ke ladang (job {})", job.job_id);
                    }
                    // Keep mining the same job for the next share.
                    internal_job_id = start_pool_job(worker_pool, job);
                }
            }
        }

        // 2. Periodic hashrate and share report.
        if window_start.elapsed() >= REPORT_EVERY {
            let secs = window_start.elapsed().as_secs_f64().max(0.001);
            let rate = format_hashrate(window_hashes as f64 / secs);
            if gaya::log_normal() {
                log::info!(
                    "📊 Hashrate {rate} | accepted {} | rejected {}",
                    stats.accepted,
                    stats.rejected
                );
            } else {
                log::info!(
                    "⛏️ Kekuatan cangkul {rate} | emas diterima {} | ditolak {}",
                    stats.accepted,
                    stats.rejected
                );
            }
            window_start = Instant::now();
            window_hashes = 0;
        }

        // 3. Keepalive so the pool does not drop an idle connection.
        if let Some(sid) = session.as_ref() {
            if last_send.elapsed() >= KEEPALIVE_EVERY {
                let id = next_id;
                next_id += 1;
                send(
                    &mut wr,
                    &json!({ "id": id, "method": "keepalived", "params": { "id": sid } }),
                )
                .await?;
                last_send = Instant::now();
            }
        }

        // 4. Messages from the pool.
        tokio::select! {
            biased;

            line = lines.next_line() => {
                let line = line
                    .context("gagal membaca dari pool")?
                    .ok_or_else(|| anyhow!("pool menutup koneksi"))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let msg: Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(e) => {
                        log::warn!("Pesan pool tidak dikenali ({e}): {line}");
                        continue;
                    }
                };

                // Notifications from the pool.
                if let Some(method) = msg.get("method").and_then(Value::as_str) {
                    if method == "job" {
                        let job_value = msg
                            .get("params")
                            .and_then(|p| p.get("job").or(Some(p)));
                        match job_value.map(parse_job) {
                            Some(Ok(job)) => {
                                if gaya::log_normal() {
                                    log::info!("🧱 New job {} from pool", job.job_id);
                                } else {
                                    log::info!("📋 Bapak nerima job {}", job.job_id);
                                }
                                internal_job_id = start_pool_job(worker_pool, &job);
                                current = Some(job);
                            }
                            Some(Err(e)) => log::warn!("Job dari pool tidak valid: {e}"),
                            None => log::warn!("Pesan job tanpa isi: {line}"),
                        }
                    } else {
                        log::debug!("Pesan pool '{method}' diabaikan");
                    }
                    continue;
                }

                // Responses to our requests.
                let id = msg.get("id").and_then(Value::as_u64);
                let error = msg.get("error").filter(|e| !e.is_null());

                if id == Some(login_id) {
                    if let Some(err) = error {
                        return Err(LoginRejected(error_text(err)).into());
                    }
                    let result = msg
                        .get("result")
                        .ok_or_else(|| anyhow!("balasan login tanpa result"))?;
                    let sid = result
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| anyhow!("balasan login tanpa id sesi"))?
                        .to_string();
                    session = Some(sid);
                    *logged_in = true;
                    if gaya::log_normal() {
                        log::info!("✅ Logged in to pool");
                    } else {
                        log::info!("🧑‍🌾 Bapak diterima di ladang");
                    }
                    if let Some(job_value) = result.get("job") {
                        match parse_job(job_value) {
                            Ok(job) => {
                                if gaya::log_normal() {
                                    log::info!("🧱 New job {} from pool", job.job_id);
                                } else {
                                    log::info!("📋 Bapak nerima job {}", job.job_id);
                                }
                                internal_job_id = start_pool_job(worker_pool, &job);
                                current = Some(job);
                            }
                            Err(e) => log::warn!("Job dari pool tidak valid: {e}"),
                        }
                    }
                } else if let Some(id) = id {
                    if let Some(job_id) = pending.remove(&id) {
                        let status_ok = msg
                            .get("result")
                            .and_then(|r| r.get("status"))
                            .and_then(Value::as_str)
                            .map(|s| s.eq_ignore_ascii_case("ok"))
                            .unwrap_or(false);
                        if error.is_none() && status_ok {
                            stats.accepted += 1;
                            if gaya::log_normal() {
                                log::info!(
                                    "✅ Share accepted (job {job_id}) | accepted {} | rejected {}",
                                    stats.accepted,
                                    stats.rejected
                                );
                            } else {
                                log::info!(
                                    "💰 Emas diterima ladang! (diterima {}, ditolak {})",
                                    stats.accepted,
                                    stats.rejected
                                );
                            }
                        } else {
                            stats.rejected += 1;
                            let reason = error
                                .map(error_text)
                                .unwrap_or_else(|| msg.to_string());
                            if gaya::log_normal() {
                                log::warn!("❌ Share rejected (job {job_id}): {reason}");
                            } else {
                                log::warn!("❌ Emas ditolak ladang: {reason}");
                            }
                        }
                    }
                }
            }

            _ = tokio::time::sleep(Duration::from_millis(1)) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kryptex_job() {
        let v: Value = serde_json::from_str(
            r#"{"difficulty":18253611008,"extranonce":"16035d3d","job_id":"bdd8ea72_18253611008","mining_hash":"1ac483fa02743013ef5b2a3ddc32e970d0d522b9f1247bdd4e35421ddd586dc4","target":"000000003c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c"}"#,
        )
        .unwrap();
        let job = parse_job(&v).unwrap();
        assert_eq!(job.job_id, "bdd8ea72_18253611008");
        assert_eq!(job.difficulty, U512::from(18_253_611_008u64));
        assert_eq!(job.extranonce, vec![0x16, 0x03, 0x5d, 0x3d]);
        assert_eq!(job.header[0], 0x1a);

        // The pool target is MAX / difficulty, matching the engine's rule.
        let target_bytes = hex::decode(v["target"].as_str().unwrap()).unwrap();
        let pool_target = U512::from_big_endian(&target_bytes);
        let engine_target = U512::MAX / job.difficulty;
        let diff = if pool_target > engine_target {
            pool_target - engine_target
        } else {
            engine_target - pool_target
        };
        assert!(diff < (engine_target >> 32));
    }

    #[test]
    fn falls_back_to_target_without_difficulty() {
        let v = json!({
            "job_id": "1",
            "mining_hash": "00".repeat(32),
            "target": hex::encode((U512::MAX / U512::from(1000u64)).to_big_endian()),
        });
        let job = parse_job(&v).unwrap();
        assert_eq!(job.difficulty, U512::from(1000u64));
        assert!(job.extranonce.is_empty());
    }

    #[test]
    fn parses_stratum_url() {
        let c =
            StratumConfig::parse_url("stratum+tcp://qtc.kryptex.network:7049", "WALLET.rig1", "x")
                .unwrap()
                .unwrap();
        assert_eq!(c.host, "qtc.kryptex.network");
        assert_eq!(c.port, 7049);
        assert_eq!(c.login, "WALLET.rig1");

        assert!(StratumConfig::parse_url("127.0.0.1:9833", "", "x")
            .unwrap()
            .is_none());
        assert!(StratumConfig::parse_url("stratum+ssl://h:1", "w", "x").is_err());
        assert!(StratumConfig::parse_url("stratum+tcp://h:1", "", "x").is_err());
    }
}
