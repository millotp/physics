#!/usr/bin/env bash
# Sweep NB_THREAD via the `--bench` runner, restoring the original value on exit.
#
# Usage:
#   ./sweep_threads.sh                 # default values: 1 2 3 4 5 6 8 10 12 15 16 20
#   ./sweep_threads.sh "2 4 8 16"      # custom space-separated list
#
# Output: one CSV-ish summary per NB_THREAD on stdout (avg over 3 trials).

set -euo pipefail

VALUES_DEFAULT="1 2 3 4 5 6 8 10 12 15 16 20"
VALUES=${1:-$VALUES_DEFAULT}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHYS=src/physics.rs
ORIG=$(awk '/^pub const NB_THREAD: usize = / {gsub(/[^0-9]/,""); print; exit}' "$PHYS")
echo "original NB_THREAD = $ORIG (will restore on exit)"

restore() {
  sed -i.bak -E "s/^pub const NB_THREAD: usize = [0-9]+;/pub const NB_THREAD: usize = $ORIG;/" "$PHYS"
  rm -f "$PHYS.bak"
  echo
  echo "restored NB_THREAD = $ORIG"
}
trap restore EXIT

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
BIN="$TARGET_DIR/release/physics"

TRIALS=3
FRAMES=100
WARMUP=500
SEED=1

run_summary() {
  local n_thread="$1"
  sed -i.bak -E "s/^pub const NB_THREAD: usize = [0-9]+;/pub const NB_THREAD: usize = $n_thread;/" "$PHYS"
  rm -f "$PHYS.bak"
  if ! cargo build --release >/dev/null 2>&1; then
    echo "$n_thread BUILD_FAILED"
    return
  fi
  if [[ ! -x "$BIN" ]]; then
    BIN=$(ls -t "$TARGET_DIR"/release/deps/physics-* 2>/dev/null | grep -v '\.' | head -1)
  fi
  local OUT
  OUT=$(mktemp)
  for _ in $(seq 1 "$TRIALS"); do
    "$BIN" --bench "$FRAMES" "$WARMUP" "$SEED" 2>/dev/null >> "$OUT"
  done
  awk -v n="$n_thread" '
    /median=/ {
      match($0, /median=[0-9]+/); m = substr($0, RSTART+7, RLENGTH-7);
      match($0, /mean=[0-9]+/);   me = substr($0, RSTART+5, RLENGTH-5);
      match($0, /p95=[0-9]+/);    p = substr($0, RSTART+4, RLENGTH-4);
      match($0, /fps_avg=[0-9.]+/); f = substr($0, RSTART+8, RLENGTH-8);
      if (m+0 < best_m || best_m == 0) best_m = m+0;
      sum_m += m+0; sum_me += me+0; sum_p += p+0; sum_f += f+0; cnt++;
    }
    END {
      if (cnt == 0) { printf "%-3s  no_samples\n", n; exit }
      printf "%-3s  best_median=%-5d avg_median=%-5d avg_mean=%-5d avg_p95=%-5d avg_fps=%-6.1f\n",
             n, best_m, sum_m/cnt, sum_me/cnt, sum_p/cnt, sum_f/cnt;
    }
  ' "$OUT"
  rm -f "$OUT"
}

echo "== sweeping NB_THREAD: $VALUES =="
echo "    trials=$TRIALS frames=$FRAMES warmup=$WARMUP seed=$SEED"
echo
printf "%-3s  %-19s %-19s %-19s %-19s %s\n" "n" "best_median" "avg_median" "avg_mean" "avg_p95" "avg_fps"
for n in $VALUES; do
  run_summary "$n"
done
