#!/bin/bash
# test_run.sh — Quick 100-step validation before full training
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Quick Validation Run ==="
python3 -c "import jax; print(f'JAX devices: {jax.device_count()} x {jax.devices()[0].device_kind}')"
echo ""
python3 train.py --steps 100 --log-interval 10 --ckpt-interval 50
echo ""
echo "=== Validation passed ==="
