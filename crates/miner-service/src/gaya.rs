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
