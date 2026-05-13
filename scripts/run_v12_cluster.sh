#!/bin/bash
# run_v12_cluster.sh — V12 cluster follow-up orchestration
#
# Implements items 1–5 from docs/cluster_roadmap_v12.md.
# Drives either:
#   (a) 2× RTX 4070 on local network via SSH (embarrassingly parallel)
#   (b) 1× H100 80GB on Runpod (sequential)
#   (c) single local GPU (rtx 3080 — items 1, 3, 4, 5 only; item 2 needs bigger)
#
# Usage:
#   ./run_v12_cluster.sh local           # single-GPU here
#   ./run_v12_cluster.sh dual <host_b>   # 2 boxes, host_b is reachable via SSH
#   ./run_v12_cluster.sh h100            # rent + run on H100 (prints runpod cmd)
#
# Inputs:
#   $V12_OUT_DIR        default /data/checkpoints/v12_cluster_followup
#   $HOST_A_NAME        default $(hostname)
#   $HOST_B_NAME        passed as second arg in dual mode
#   $WORK_QUEUE_FILE    default $V12_OUT_DIR/work_queue.txt  (SSH-locked)
#
# Decision rules (see docs/cluster_roadmap_v12.md §"Decision rules per item"):
#   exit 0 = all items completed within their decision rules
#   exit 1 = at least one item falsified its prediction (interesting result)
#   exit 2 = infrastructure failure (network, OOM, missing checkpoint)
#
# The script is idempotent — re-running picks up where the work queue left off.

set -euo pipefail

MODE="${1:-local}"
HOST_B="${2:-}"

V12_OUT_DIR="${V12_OUT_DIR:-/data/checkpoints/v12_cluster_followup}"
HOST_A_NAME="${HOST_A_NAME:-$(hostname)}"
WORK_QUEUE_FILE="$V12_OUT_DIR/work_queue.txt"
LOCK_DIR="$V12_OUT_DIR/.lock"
LOG_DIR="$V12_OUT_DIR/logs"
RESULTS_DIR="$V12_OUT_DIR/results"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$V12_OUT_DIR" "$LOG_DIR" "$RESULTS_DIR"

# ------------------------------------------------------------ work manifest
# Each line:  <substrate_id> <seed> <item_number> <vram_class> <expected_minutes>
# vram_class: tiny (<2GB), small (<8GB), mid (<16GB), large (>16GB → H100 only)
generate_manifest() {
  cat > "$WORK_QUEUE_FILE" <<'MANIFEST'
# item 1: 13 substrates × 5 seeds with raw FIM diagonals retained
mlp_trained_w200 0 1 tiny 8
mlp_trained_w200 1 1 tiny 8
mlp_trained_w200 2 1 tiny 8
mlp_trained_w200 3 1 tiny 8
mlp_trained_w200 4 1 tiny 8
cnn_trained_w200 0 1 tiny 12
cnn_trained_w200 1 1 tiny 12
cnn_trained_w200 2 1 tiny 12
cnn_trained_w200 3 1 tiny 12
cnn_trained_w200 4 1 tiny 12
vit_trained_w200 0 1 small 18
vit_trained_w200 1 1 small 18
vit_trained_w200 2 1 small 18
vit_trained_w200 3 1 small 18
vit_trained_w200 4 1 small 18
mlp_untrained_w200 0 1 tiny 4
mlp_untrained_w200 1 1 tiny 4
mlp_untrained_w200 2 1 tiny 4
mlp_untrained_w200 3 1 tiny 4
mlp_untrained_w200 4 1 tiny 4
boolean_circuit 0 1 tiny 6
boolean_circuit 1 1 tiny 6
boolean_circuit 2 1 tiny 6
boolean_circuit 3 1 tiny 6
boolean_circuit 4 1 tiny 6
linear_regression 0 1 tiny 1
logistic_regression 0 1 tiny 1
kernel_ridge 0 1 tiny 2
gaussian_process 0 1 tiny 2
u1_lattice_L8 0 1 tiny 3
su2_lattice_L3 0 1 tiny 3
ising_chain_N256 0 1 tiny 2
harmonic_chain_N256 0 1 tiny 2
cellular_automaton_R110 0 1 tiny 2
random_matrix_GOE 0 1 tiny 2

# item 3: 5-seed at production scale
resnet50_imagenet1k 0 3 small 30
resnet50_imagenet1k 1 3 small 30
resnet50_imagenet1k 2 3 small 30
resnet50_imagenet1k 3 3 small 30
resnet50_imagenet1k 4 3 small 30
vit_l_16 0 3 small 45
vit_l_16 1 3 small 45
vit_l_16 2 3 small 45
vit_l_16 3 3 small 45
vit_l_16 4 3 small 45
gpt2_large 0 3 mid 60
gpt2_large 1 3 mid 60
gpt2_large 2 3 mid 60
gpt2_large 3 3 mid 60
gpt2_large 4 3 mid 60
mamba_790m_hf 0 3 small 30
mamba_790m_hf 1 3 small 30
mamba_790m_hf 2 3 small 30
mamba_790m_hf 3 3 small 30
mamba_790m_hf 4 3 small 30

# item 4: probe convergence sweep (single seed, multiple probe counts)
pythia28b_probe_n50 42 4 mid 30
pythia28b_probe_n100 42 4 mid 45
pythia28b_probe_n200 42 4 mid 60
pythia28b_probe_n400 42 4 mid 90
pythia28b_probe_n800 42 4 mid 150
pythia28b_probe_n1600 42 4 mid 240

# item 5: 300M-param RFF kernel-ridge control (parameter-matched to ViT-L/16)
rff_kernel_ridge_300m 0 5 small 30
rff_kernel_ridge_300m 1 5 small 30
rff_kernel_ridge_300m 2 5 small 30
rff_kernel_ridge_300m 3 5 small 30
rff_kernel_ridge_300m 4 5 small 30

# item 2: real-data LM-loss FIM (Pile validation)
pythia14b_pile_loss 0 2 small 45
pythia14b_pile_loss 1 2 small 45
pythia14b_pile_loss 2 2 small 45
pythia28b_pile_loss 0 2 mid 90
pythia28b_pile_loss 1 2 mid 90
pythia28b_pile_loss 2 2 mid 90
mamba_790m_pile_loss 0 2 small 30
mamba_790m_pile_loss 1 2 small 30
mamba_790m_pile_loss 2 2 small 30
olmoe_1b7b_pile_loss_int4 0 2 mid 120
olmoe_1b7b_pile_loss_int4 1 2 mid 120

# item 2 large: Pythia-6.9B FP16 → flagged for H100 only
pythia69b_pile_loss 0 2 large 180
pythia69b_pile_loss 1 2 large 180
MANIFEST
}

# ------------------------------------------------------------ work-queue lock helpers
acquire_lock() {
  local item="$1"
  mkdir -p "$LOCK_DIR"
  # atomic mkdir-based lock
  if mkdir "$LOCK_DIR/$item.lock" 2>/dev/null; then
    echo "$HOSTNAME:$$" > "$LOCK_DIR/$item.lock/owner"
    return 0
  fi
  return 1
}

release_lock() {
  rm -rf "$LOCK_DIR/$1.lock"
}

# ------------------------------------------------------------ runner per work-item
run_work_item() {
  local substrate="$1"
  local seed="$2"
  local item="$3"
  local vram_class="$4"
  local minutes="$5"
  local job_id="${substrate}_seed${seed}_item${item}"
  local result_file="$RESULTS_DIR/${job_id}.json"
  local log_file="$LOG_DIR/${job_id}.log"

  if [ -f "$result_file" ]; then
    echo "  [skip] $job_id already done"
    return 0
  fi

  echo "  [run]  $job_id (vram=$vram_class, ~${minutes}min)"
  case "$item" in
    1) script="experiments/v12_partition_invariant/run.py" ;;
    2) script="experiments/v12_lm_loss_fim/run.py" ;;
    3) script="experiments/v12_production_multiseed/run.py" ;;
    4) script="experiments/v6_0_mechanism/probe_convergence.py" ;;
    5) script="experiments/v12_nondeep_control/run.py" ;;
    *) echo "    unknown item $item" >&2; return 2 ;;
  esac

  cd "$REPO_ROOT"
  python3 "$script" \
    --substrate "$substrate" \
    --seed "$seed" \
    --out "$result_file" \
    --raw-fim-out "$RESULTS_DIR/${job_id}_raw_fim.npy" \
    > "$log_file" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "    FAIL $job_id rc=$rc — see $log_file"
    return $rc
  fi
  echo "    OK   $job_id"
}

# ------------------------------------------------------------ main loop
work_loop() {
  local skip_large="${1:-yes}"  # "yes" on 4070 (skip large items), "no" on H100
  local n_done=0 n_skipped=0 n_failed=0

  while read -r substrate seed item vram_class minutes; do
    [ -z "$substrate" ] && continue
    [[ "$substrate" =~ ^# ]] && continue

    if [ "$skip_large" = "yes" ] && [ "$vram_class" = "large" ]; then
      echo "  [defer-to-h100] $substrate seed=$seed item=$item"
      n_skipped=$((n_skipped + 1))
      continue
    fi

    job_key="${substrate}_seed${seed}_item${item}"
    if ! acquire_lock "$job_key"; then
      continue  # another worker has it
    fi
    if run_work_item "$substrate" "$seed" "$item" "$vram_class" "$minutes"; then
      n_done=$((n_done + 1))
    else
      n_failed=$((n_failed + 1))
    fi
    release_lock "$job_key"
  done < "$WORK_QUEUE_FILE"

  echo ""
  echo "=== work_loop summary ==="
  echo "  completed: $n_done"
  echo "  deferred:  $n_skipped"
  echo "  failed:    $n_failed"
  return 0
}

# ------------------------------------------------------------ aggregate + decision rules
aggregate() {
  python3 "$REPO_ROOT/experiments/v12_partition_invariant/aggregate.py" \
    --results-dir "$RESULTS_DIR" \
    --out "$V12_OUT_DIR/v12_aggregate.json"

  python3 "$REPO_ROOT/experiments/v12_partition_invariant/decision_rules.py" \
    --aggregate "$V12_OUT_DIR/v12_aggregate.json" \
    --out "$V12_OUT_DIR/v12_decision_verdict.json"

  local verdict=$(python3 -c "import json; print(json.load(open('$V12_OUT_DIR/v12_decision_verdict.json'))['overall_verdict'])")
  echo ""
  echo "=== V12 decision verdict: $verdict ==="
  case "$verdict" in
    PASS_ALL) return 0 ;;
    PARTIAL) return 1 ;;
    *)       return 2 ;;
  esac
}

# ------------------------------------------------------------ dispatch
case "$MODE" in
  local)
    [ -f "$WORK_QUEUE_FILE" ] || generate_manifest
    work_loop yes
    aggregate
    ;;

  dual)
    if [ -z "$HOST_B" ]; then
      echo "FAIL: dual mode requires second arg = HOST_B ssh-name"
      exit 2
    fi
    [ -f "$WORK_QUEUE_FILE" ] || generate_manifest
    # launch worker on remote box
    ssh "$HOST_B" "cd $REPO_ROOT && \
      V12_OUT_DIR=$V12_OUT_DIR HOST_A_NAME=$HOST_A_NAME \
      $REPO_ROOT/scripts/run_v12_cluster.sh local" &
    REMOTE_PID=$!
    # launch worker locally
    work_loop yes
    wait $REMOTE_PID
    aggregate
    ;;

  h100)
    [ -f "$WORK_QUEUE_FILE" ] || generate_manifest
    cat <<HINT

To run V12 on a rented H100:

  # Runpod community $1.99/hr template "PyTorch 2.5 + CUDA 12.4", 1× H100 80GB
  ssh root@<runpod-host>
  git clone https://github.com/<anonymous>/nn_universe.git
  cd nn_universe
  pip install torch numpy scipy transformers accelerate bitsandbytes
  V12_OUT_DIR=/workspace/v12 scripts/run_v12_cluster.sh local

  # Expected wall-clock: ~24 hours full FP16 across all items.
  # Expected cost: ~$48.

After completion, scp results back to $V12_OUT_DIR/results.

HINT
    ;;

  *)
    echo "Usage: $0 {local|dual <host_b>|h100}"
    exit 2
    ;;
esac
