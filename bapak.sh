#!/usr/bin/env bash
#
#  bapak — penambang Quantus (QTC) untuk GPU NVIDIA
#  Turunan quantus-miner (Apache-2.0). Repo: github.com/zeins76/cangkul-bapak
#
#  Cara pakai:
#     1. Edit WALLET dan LADANG di bawah.
#     2. Jalankan:  bash bapak.sh
#
set -euo pipefail

# ======================= KONFIGURASI =======================
WALLET="qzpAZqbNNAihYg888iEyoLGDtebKeQbeqxgSzNkw5Pqu2MZFV.rig1"
LADANG="stratum+tcp://eu.lproute.com:5660"     # LuckyPool. Kryptex: stratum+tcp://qtc.kryptex.network:7049
COIN="quantus"
AUTO_RESTART=1                                  # 1 = hidupkan lagi otomatis kalau berhenti, 0 = tidak
# ===========================================================

URL="https://github.com/zeins76/cangkul-bapak/releases/latest/download/bapak-linux-x86_64"
BIN="bapak"

garis() { printf '%s\n' "------------------------------------------------------------"; }

garis
echo "  bapak — penambang Quantus (QTC)"
echo "  coin   : $COIN"
echo "  ladang : $LADANG"
echo "  wallet : $WALLET"
garis

# --- Unduh binary kalau belum ada ---
if [ ! -x "./$BIN" ]; then
  echo "[*] Mengunduh bapak dari rilis terbaru..."
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress "$URL" -O "$BIN"
  else
    curl -L --progress-bar "$URL" -o "$BIN"
  fi
  chmod +x "$BIN"                               # beri izin eksekusi (otomatis, tidak perlu manual)
  echo "[*] bapak siap."
  garis
fi

# --- Jalankan ---
run_bapak() {
  ./"$BIN" --coin "$COIN" --cuda-gpu --ladang "$LADANG" --wallet "$WALLET"
}

if [ "$AUTO_RESTART" = "1" ]; then
  # Hidupkan lagi kalau berhenti (bukan karena Ctrl+C). Transparan: proses tetap bernama bapak.
  trap 'echo; echo "[*] Dihentikan pengguna."; exit 0' INT TERM
  while true; do
    run_bapak || true
    echo "[!] bapak berhenti. Mulai lagi dalam 5 detik... (Ctrl+C untuk keluar)"
    sleep 5
  done
else
  exec ./"$BIN" --coin "$COIN" --cuda-gpu --ladang "$LADANG" --wallet "$WALLET"
fi
