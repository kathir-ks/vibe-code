#!/bin/bash
# deploy_tpu.sh — Create a preemptible TPU VM and run training
#
# Usage:
#   chmod +x deploy_tpu.sh
#   ./deploy_tpu.sh [TPU_NAME] [TPU_TYPE]
#
# Examples:
#   ./deploy_tpu.sh                          # defaults: mathematica-v1, v2-8
#   ./deploy_tpu.sh my-tpu v3-8             # custom name and type

set -euo pipefail

# --- Configuration ---
TPU_NAME="${1:-mathematica-v1}"
TPU_TYPE="${2:-v2-8}"
ZONE="us-central2-b"
PROJECT="$(gcloud config get-value project)"
RUNTIME_VERSION="tpu-ubuntu2204-base"
REPO_URL="git@github.com:kathir-ks/vibe-code.git"
WORK_DIR="~/vibe-code/mathematica"

echo "=== TPU Deployment ==="
echo "  Project:  $PROJECT"
echo "  TPU Name: $TPU_NAME"
echo "  TPU Type: $TPU_TYPE"
echo "  Zone:     $ZONE"
echo "  Runtime:  $RUNTIME_VERSION"
echo ""

# --- Step 1: Create TPU VM (preemptible) ---
echo "[1/4] Creating preemptible TPU VM..."
gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$ZONE" \
    --accelerator-type="$TPU_TYPE" \
    --version="$RUNTIME_VERSION" \
    --preemptible \
    --quiet || {
    echo "TPU VM may already exist. Trying to continue..."
}

# --- Step 2: Setup environment on TPU VM ---
echo "[2/4] Setting up environment on TPU VM..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --command="$(cat <<'REMOTE_SETUP'
set -e
echo ">> Installing dependencies..."
pip install --upgrade pip
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax numpy matplotlib

echo ">> Cloning repo..."
if [ -d ~/vibe-code ]; then
    cd ~/vibe-code && git pull origin main
else
    git clone git@github.com:kathir-ks/vibe-code.git ~/vibe-code
fi

echo ">> Verifying JAX sees TPU..."
python3 -c "import jax; print(f'JAX devices: {jax.device_count()} x {jax.devices()[0].device_kind}')"

echo ">> Setup complete."
REMOTE_SETUP
)"

# --- Step 3: Start training in a tmux session ---
echo "[3/4] Starting training in tmux session 'train'..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --command="$(cat <<'REMOTE_TRAIN'
set -e
cd ~/vibe-code/mathematica

# Kill existing session if any
tmux kill-session -t train 2>/dev/null || true

# Start training in tmux (survives SSH disconnect)
tmux new-session -d -s train "python3 train.py 2>&1 | tee training.log"

echo ">> Training started in tmux session 'train'"
echo ">> To monitor: gcloud compute tpus tpu-vm ssh TPU_NAME --zone=us-central2-b --command='tmux attach -t train'"
REMOTE_TRAIN
)"

# --- Step 4: Print monitoring commands ---
echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Monitor training:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command='tmux attach -t train'"
echo ""
echo "Check logs:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command='tail -50 ~/vibe-code/mathematica/training.log'"
echo ""
echo "Stop TPU (save cost):"
echo "  gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet"
echo ""
