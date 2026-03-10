#!/bin/bash
# deploy_tpu_v6e.sh — Deploy to TPU v6e in Europe (supports multihost)
#
# Usage:
#   chmod +x deploy_tpu_v6e.sh
#   ./deploy_tpu_v6e.sh                              # defaults: mathematica-v6e, v6e-8
#   ./deploy_tpu_v6e.sh my-tpu v6e-16                # custom name + multihost (2 hosts)
#   ./deploy_tpu_v6e.sh my-tpu v6e-8 --test          # 100-step validation run
#
# Multihost topology:
#   v6e-1   →  1 chip,  1 host
#   v6e-4   →  4 chips, 1 host
#   v6e-8   →  8 chips, 1 host
#   v6e-16  → 16 chips, 2 hosts (8 per host)
#   v6e-32  → 32 chips, 4 hosts (8 per host)
#   v6e-64  → 64 chips, 8 hosts (8 per host)

set -euo pipefail

# --- Configuration ---
TPU_NAME="${1:-mathematica-v6e}"
TPU_TYPE="${2:-v6e-8}"
TEST_MODE="${3:-}"
ZONE="europe-west4-b"
PROJECT="$(gcloud config get-value project 2>/dev/null)"
RUNTIME_VERSION="tpu-ubuntu2204-base"

echo "============================================"
echo "  TPU v6e Deployment (preemptible, Europe)"
echo "============================================"
echo "  Project:  $PROJECT"
echo "  TPU Name: $TPU_NAME"
echo "  Type:     $TPU_TYPE"
echo "  Zone:     $ZONE"
echo "  Runtime:  $RUNTIME_VERSION"
echo ""

# --- Step 1: Create TPU VM (preemptible) ---
echo "[1/4] Creating preemptible TPU VM..."
if gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" &>/dev/null; then
    echo "  TPU '$TPU_NAME' already exists. Reusing."
else
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --zone="$ZONE" \
        --accelerator-type="$TPU_TYPE" \
        --version="$RUNTIME_VERSION" \
        --preemptible \
        --quiet
    echo "  Created."
fi

# --- Step 2: Setup environment on ALL workers ---
echo "[2/4] Setting up environment on all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --worker=all \
    --command="$(cat <<'REMOTE_SETUP'
set -e

echo ">> [$(hostname)] Upgrading pip..."
pip install --upgrade pip -q

echo ">> [$(hostname)] Installing JAX for TPU..."
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q

echo ">> [$(hostname)] Installing dependencies..."
pip install flax optax numpy matplotlib -q

echo ">> [$(hostname)] Cloning / updating repo..."
if [ -d ~/vibe-code ]; then
    cd ~/vibe-code && git pull origin main
else
    git clone https://github.com/kathir-ks/vibe-code.git ~/vibe-code
fi

echo ">> [$(hostname)] Verifying JAX + TPU..."
python3 -c "
import jax
jax.distributed.initialize()
devs = jax.devices()
print(f'  Total JAX devices: {len(devs)} x {devs[0].device_kind}')
print(f'  Process {jax.process_index()}/{jax.process_count()}: {jax.local_device_count()} local devices')
for i, d in enumerate(jax.local_devices()):
    print(f'    [{i}] {d}')
"

echo ">> [$(hostname)] Setup complete."
REMOTE_SETUP
)"

# --- Step 3: Launch training on ALL workers ---
if [ "$TEST_MODE" = "--test" ]; then
    echo "[3/4] Running 100-step validation on all workers..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --worker=all \
        --command="$(cat <<'REMOTE_TEST'
set -e
cd ~/vibe-code/mathematica
python3 train.py --steps 100 --log-interval 10 --ckpt-interval 50 2>&1 | tee "test_run_$(hostname).log"
echo ""
echo ">> Test run complete on $(hostname)."
REMOTE_TEST
)"
else
    echo "[3/4] Starting full training on all workers..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --worker=all \
        --command="$(cat <<'REMOTE_TRAIN'
set -e
cd ~/vibe-code/mathematica

# Kill existing session if any
tmux kill-session -t train 2>/dev/null || true

# Start training in tmux (survives SSH disconnect)
tmux new-session -d -s train \
    "python3 train.py 2>&1 | tee training_$(hostname).log; echo 'TRAINING DONE'; exec bash"

echo ">> Training started in tmux session 'train' on $(hostname)"
REMOTE_TRAIN
)"
fi

# --- Step 4: Print useful commands ---
echo ""
echo "============================================"
echo "  Deployment Complete"
echo "============================================"
echo ""
echo "Attach to training (worker 0):"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command='tmux attach -t train'"
echo ""
echo "Tail logs (worker 0):"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command='tail -30 ~/vibe-code/mathematica/training_*.log'"
echo ""
echo "Check all workers:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command='pgrep -a python3'"
echo ""
echo "Run inference after training (worker 0 only):"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command='cd ~/vibe-code/mathematica && python3 inference.py'"
echo ""
echo "Delete TPU (stop billing):"
echo "  gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet"
echo ""
