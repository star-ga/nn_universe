#!/usr/bin/env bash
# V3.0 Tier-1 item 2: 20-seed robustness at large N.
#
# Runs 20 seeds each at widths 1024 and 4096 on whatever device this
# script is invoked on. Targets ~15 min per width on an A100 / H100.
# Writes per-seed JSONs via multiseed_remote.py.
#
# Usage (on the remote pod):
#   bash experiments/v1_2_scaling/run_20seed_remote.sh /workspace/v3_0_robustness
#
# Produces:
#   $OUT_DIR/width_1024_20seeds.json
#   $OUT_DIR/width_4096_20seeds.json
set -euo pipefail

OUT_DIR="${1:-/workspace/v3_0_robustness}"
mkdir -p "$OUT_DIR"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MULTI="$REPO/experiments/v1_2_scaling/multiseed_remote.py"

SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61"

echo "=== 20-seed @ width=1024 ==="
python3 "$MULTI" --width 1024 --seeds $SEEDS --steps 15000 --out "$OUT_DIR/width_1024_20seeds.json"

echo ""
echo "=== 20-seed @ width=4096 (bf16 + grad_ckpt) ==="
python3 "$MULTI" --width 4096 --seeds $SEEDS --steps 15000 --bf16 --grad-ckpt --out "$OUT_DIR/width_4096_20seeds.json"

echo ""
echo "Done. Results in $OUT_DIR"
ls -la "$OUT_DIR"
