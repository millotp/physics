#!/usr/bin/env bash
# Repeatable benchmark runner.
#
# Usage:
#   ./bench.sh                 # 3 trials × (warmup=500 frames=100 seed=1) ≈ 5s each
#   ./bench.sh 3 100 500 1     # trials frames warmup seed
#   ./bench.sh 3 100 500 1 label  # also writes bench-<label>.txt
#
# Compares with bench-baseline.txt if present.

set -euo pipefail

TRIALS=${1:-3}
FRAMES=${2:-100}
WARMUP=${3:-500}
SEED=${4:-1}
LABEL=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "== building release =="
cargo build --release >/dev/null

# Respect $CARGO_TARGET_DIR if set (common under sandboxed runners).
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
BIN="$TARGET_DIR/release/physics"
if [[ ! -x "$BIN" ]]; then
  # Fall back to the deps binary (no copy).
  BIN=$(ls -t "$TARGET_DIR"/release/deps/physics-* 2>/dev/null | grep -v '\.' | head -1)
fi
if [[ -z "${BIN:-}" || ! -x "$BIN" ]]; then
  echo "error: could not locate physics binary under $TARGET_DIR/release" >&2
  exit 1
fi

OUT=$(mktemp)
echo "== running $TRIALS trials: frames=$FRAMES warmup=$WARMUP seed=$SEED =="
echo "    bin=$BIN"
for i in $(seq 1 "$TRIALS"); do
  line=$("$BIN" --bench "$FRAMES" "$WARMUP" "$SEED" 2>/dev/null)
  echo "trial_$i $line"
  echo "trial_$i $line" >> "$OUT"
done

summary() {
  awk '
    /median=/ {
      match($0, /median=[0-9]+/); m = substr($0, RSTART+7, RLENGTH-7);
      match($0, /mean=[0-9]+/);   me = substr($0, RSTART+5, RLENGTH-5);
      match($0, /p95=[0-9]+/);    p = substr($0, RSTART+4, RLENGTH-4);
      match($0, /fps_avg=[0-9.]+/); f = substr($0, RSTART+8, RLENGTH-8);
      if (m+0 < best_m || best_m == 0) best_m = m+0;
      sum_m += m+0; sum_me += me+0; sum_p += p+0; sum_f += f+0; n++;
    }
    END {
      if (n == 0) { print "no samples"; exit }
      printf "best_median=%d avg_median=%.0f avg_mean=%.0f avg_p95=%.0f avg_fps=%.1f  (n=%d)\n",
             best_m, sum_m/n, sum_me/n, sum_p/n, sum_f/n, n;
    }
  ' "$1"
}

echo
echo "== summary =="
summary "$OUT"

if [[ -n "$LABEL" ]]; then
  cp "$OUT" "bench-$LABEL.txt"
  echo "saved trials -> bench-$LABEL.txt"
fi

if [[ -f bench-baseline.txt && "$LABEL" != "baseline" ]]; then
  echo
  echo "== vs baseline =="
  echo -n "baseline:  "; summary bench-baseline.txt
  echo -n "current :  "; summary "$OUT"
fi

rm -f "$OUT"
