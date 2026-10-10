//! Gaya pesan log untuk `bapak`.
//!
//! Default-nya pesan bergaya "bapak tani" (GPU = cangkul, pool = ladang,
//! share = emas). Flag `--log-normal` mengembalikan pesan teknis asli dari
//! quantus-miner, berguna saat mencari solusi error di dokumentasi upstream.

use std::sync::atomic::{AtomicBool, Ordering};

static LOG_NORMAL: AtomicBool = AtomicBool::new(false);

/// Aktifkan (`true`) atau matikan pesan teknis asli.
pub fn set_log_normal(on: bool) {
    LOG_NORMAL.store(on, Ordering::Relaxed);
}

/// `true` kalau pengguna minta pesan teknis asli (`--log-normal`).
pub fn log_normal() -> bool {
    LOG_NORMAL.load(Ordering::Relaxed)
}

/// Nama alat untuk tipe worker: GPU = "Cangkul", CPU = "Arit".
pub fn nama_alat(is_gpu: bool) -> &'static str {
    if is_gpu {
        "Cangkul"
    } else {
        "Arit"
    }
}

/// Nama modul di log diganti nama gaya bapak, misalnya `miner_service`
/// menjadi `layanan_bapak`. Bagian setelah `::` yang tidak dikenal dibiarkan.
pub fn nama_target(target: &str) -> String {
    const GANTI: [(&str, &str); 7] = [
        ("miner_service::stratum", "layanan_bapak::ladang"),
        ("miner_service::quic", "layanan_bapak::ladang"),
        ("miner_service", "layanan_bapak"),
        ("cuda_engine", "mesin_cangkul"),
        ("gpu_engine", "mesin_cangkul"),
        ("engine_cpu", "mesin_arit"),
        ("miner_cli", "bapak"),
    ];
    for (asli, baru) in GANTI {
        if let Some(sisa) = target.strip_prefix(asli) {
            if sisa.is_empty() || sisa.starts_with("::") {
                return format!("{baru}{sisa}");
            }
        }
    }
    target.to_string()
}

#[cfg(test)]
mod tests {
    use super::nama_target;

    #[test]
    fn ganti_nama_target() {
        assert_eq!(nama_target("miner_service"), "layanan_bapak");
        assert_eq!(
            nama_target("miner_service::stratum"),
            "layanan_bapak::ladang"
        );
        assert_eq!(nama_target("miner_service::quic"), "layanan_bapak::ladang");
        assert_eq!(nama_target("cuda_engine"), "mesin_cangkul");
        assert_eq!(nama_target("gpu_engine"), "mesin_cangkul");
        assert_eq!(nama_target("engine_cpu::fast"), "mesin_arit::fast");
        assert_eq!(nama_target("miner_cli"), "bapak");
        assert_eq!(nama_target("miner_service_extra"), "miner_service_extra");
        assert_eq!(nama_target("wgpu_core"), "wgpu_core");
    }
}
