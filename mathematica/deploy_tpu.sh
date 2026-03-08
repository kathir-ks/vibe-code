#!/bin/bash
# deploy_tpu.sh — Create a preemptible TPU v4-8 VM and run training
#
# Usage:
#   chmod +x deploy_tpu.sh
#   ./deploy_tpu.sh                     # defaults: mathematica-v1
#   ./deploy_tpu.sh my-tpu              # custom name
#   ./deploy_tpu.sh my-tpu --test       # 100-step validation run only

set -euo pipefail

# --- Configuration ---
TPU_NAME="${1:-mathematica-v1}"
TEST_MODE="${2:-}"
ZONE="us-central2-b"
TPU_TYPE="v4-8"
PROJECT="$(gcloud config get-value project 2>/dev/null)"
RUNTIME_VERSION="tpu-ubuntu2204-base"

echo "============================================"
echo "  TPU v4-8 Deployment (preemptible)"
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

# --- Step 2: Setup environment on TPU VM ---
echo "[2/4] Setting up environment..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --command="$(cat <<'REMOTE_SETUP'
set -e

echo ">> Upgrading pip..."
pip install --upgrade pip -q

echo ">> Installing JAX for TPU..."
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q

echo ">> Installing dependencies..."
pip install flax optax numpy matplotlib -q

echo ">> Cloning / updating repo..."
if [ -d ~/vibe-code ]; then
    cd ~/vibe-code && git pull origin main
else
    git clone https://github.com/kathir-ks/vibe-code.git ~/vibe-code
fi

echo ">> Verifying JAX + TPU..."
python3 -c "
import jax
devs = jax.devices()
print(f'  JAX devices: {len(devs)} x {devs[0].device_kind}')
for i, d in enumerate(devs):
    print(f'    [{i}] {d}')
"

echo ">> Setup complete."
REMOTE_SETUP
)"

# --- Step 3: Launch training ---
if [ "$TEST_MODE" = "--test" ]; then
    echo "[3/4] Running 100-step validation..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --command="$(cat <<'REMOTE_TEST'
set -e
cd ~/vibe-code/mathematica
python3 train.py --steps 100 --log-interval 10 --ckpt-interval 50 2>&1 | tee test_run.log
echo ""
echo ">> Test run complete. Check test_run.log for results."
REMOTE_TEST
)"
else
    echo "[3/4] Starting full training in tmux session 'train'..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --command="$(cat <<'REMOTE_TRAIN'
set -e
cd ~/vibe-code/mathematica

# Kill existing session if any
tmux kill-session -t train 2>/dev/null || true

# Start training in tmux (survives SSH disconnect)
tmux new-session -d -s train \
    "python3 train.py 2>&1 | tee training.log; echo 'TRAINING DONE'; exec bash"

echo ">> Training started in tmux session 'train'"
REMOTE_TRAIN
)"
fi

# --- Step 4: Print useful commands ---
echo ""
echo "============================================"
echo "  Deployment Complete"
echo "============================================"
echo ""
echo "Attach to training session:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command='tmux attach -t train'"
echo ""
echo "Tail logs:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command='tail -30 ~/vibe-code/mathematica/training.log'"
echo ""
echo "Run inference after training:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command='cd ~/vibe-code/mathematica && python3 inference.py'"
echo ""
echo "Delete TPU (stop billing):"
echo "  gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet"
echo ""
