# train.py — Training loop for the emergent number line experiment
#
# Loss D: Self-reconstruction + Cross-area-matching + Latent binarization
#
#   SELF path (keeps decoders sharp):
#     C -> Enc_A -> Z_A -> Dec_A -> C'   loss = BCE(C', C)
#     T -> Enc_B -> Z_B -> Dec_B -> T'   loss = BCE(T', T)
#
#   CROSS path (forces count encoding via area proxy):
#     C -> Enc_A -> Z_A -> Dec_B -> T'   loss = (area(T') - area(C))^2
#     T -> Enc_B -> Z_B -> Dec_A -> C'   loss = (area(C') - area(T))^2
#
#   LATENT path (clean binary latent):
#     loss = mean(Z * (1 - Z))  where Z = sigmoid(logits)
#
# Supports single-device (JIT), multi-device (pmap), and multihost (pmap) training.
# Automatically selects pmap when jax.device_count() > 1 (e.g. TPU v4-8, v6e multihost).

import jax
import jax.numpy as jnp
import optax
from flax import serialization
import flax.jax_utils as flax_utils
from functools import partial
import os
import time
import argparse

import config as cfg
from data import generate_training_pair, generate_training_batch
from models import Encoder, Decoder, binarize_ste


# ---------------------------------------------------------------------------
# Loss function (pure, works under both jit and pmap)
# ---------------------------------------------------------------------------

def compute_loss(params_A, params_B,
                 circle_img, triangle_img,
                 enc_A, dec_A, enc_B, dec_B,
                 lambda_self, lambda_cross, lambda_bin):
    """Compute the combined loss for one training step."""
    # --- Encode ---
    logits_A = enc_A.apply({'params': params_A['encoder']}, circle_img)
    logits_B = enc_B.apply({'params': params_B['encoder']}, triangle_img)

    z_A = binarize_ste(logits_A)
    z_B = binarize_ste(logits_B)

    # --- Self-reconstruction ---
    recon_C_logits = dec_A.apply({'params': params_A['decoder']}, z_A)
    recon_T_logits = dec_B.apply({'params': params_B['decoder']}, z_B)

    loss_self_A = optax.sigmoid_binary_cross_entropy(recon_C_logits, circle_img).mean()
    loss_self_B = optax.sigmoid_binary_cross_entropy(recon_T_logits, triangle_img).mean()
    loss_self = loss_self_A + loss_self_B

    # --- Cross-communication (area-matching) ---
    cross_T_logits = dec_B.apply({'params': params_B['decoder']}, z_A)
    cross_C_logits = dec_A.apply({'params': params_A['decoder']}, z_B)

    cross_T_probs = jax.nn.sigmoid(cross_T_logits)
    cross_C_probs = jax.nn.sigmoid(cross_C_logits)

    area_circle_input = jnp.sum(circle_img)
    area_triangle_input = jnp.sum(triangle_img)
    area_cross_T = jnp.sum(cross_T_probs)
    area_cross_C = jnp.sum(cross_C_probs)

    num_pixels = cfg.IMG_SIZE * cfg.IMG_SIZE
    loss_cross_AB = ((area_cross_T - area_circle_input) / num_pixels) ** 2
    loss_cross_BA = ((area_cross_C - area_triangle_input) / num_pixels) ** 2
    loss_cross = loss_cross_AB + loss_cross_BA

    # --- Latent binarization ---
    probs_A = jax.nn.sigmoid(logits_A)
    probs_B = jax.nn.sigmoid(logits_B)
    loss_bin = jnp.mean(probs_A * (1 - probs_A)) + jnp.mean(probs_B * (1 - probs_B))

    # --- Total ---
    total_loss = (lambda_self * loss_self
                  + lambda_cross * loss_cross
                  + lambda_bin * loss_bin)

    metrics = {
        'total_loss': total_loss,
        'loss_self': loss_self,
        'loss_self_A': loss_self_A,
        'loss_self_B': loss_self_B,
        'loss_cross': loss_cross,
        'loss_bin': loss_bin,
        'area_circle_in': area_circle_input / num_pixels,
        'area_cross_T': area_cross_T / num_pixels,
        'area_triangle_in': area_triangle_input / num_pixels,
        'area_cross_C': area_cross_C / num_pixels,
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Single-device JIT path (fallback for 1 device)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def train_step_jit(params_A, params_B, opt_state_A, opt_state_B,
                   enc_A, dec_A, enc_B, dec_B, optimizer_def,
                   circle_img, triangle_img,
                   lambda_self, lambda_cross, lambda_bin):
    """Single JIT-compiled training step (1 device)."""
    grad_fn = jax.value_and_grad(compute_loss, argnums=(0, 1), has_aux=True)
    (loss, metrics), (grads_A, grads_B) = grad_fn(
        params_A, params_B,
        circle_img, triangle_img,
        enc_A, dec_A, enc_B, dec_B,
        lambda_self, lambda_cross, lambda_bin,
    )

    updates_A, opt_state_A = optimizer_def.update(grads_A, opt_state_A, params_A)
    params_A = optax.apply_updates(params_A, updates_A)

    updates_B, opt_state_B = optimizer_def.update(grads_B, opt_state_B, params_B)
    params_B = optax.apply_updates(params_B, updates_B)

    return params_A, params_B, opt_state_A, opt_state_B, metrics


# ---------------------------------------------------------------------------
# Multi-device pmap path (TPU v4-8: 4 devices)
# ---------------------------------------------------------------------------

@partial(jax.pmap,
         axis_name=cfg.PMAP_AXIS_NAME,
         # 4-8: model instances + optimizer, 11-13: loss weight scalars
         # All static_broadcasted args are broadcast (not sharded).
         # Remaining args (0-3: params/opt_state, 9-10: images) default to in_axes=0.
         static_broadcasted_argnums=(4, 5, 6, 7, 8, 11, 12, 13))
def train_step_pmap(params_A, params_B, opt_state_A, opt_state_B,
                    enc_A, dec_A, enc_B, dec_B, optimizer_def,
                    circle_img, triangle_img,
                    lambda_self, lambda_cross, lambda_bin):
    """Data-parallel training step with gradient averaging across devices."""
    grad_fn = jax.value_and_grad(compute_loss, argnums=(0, 1), has_aux=True)
    (loss, metrics), (grads_A, grads_B) = grad_fn(
        params_A, params_B,
        circle_img, triangle_img,
        enc_A, dec_A, enc_B, dec_B,
        lambda_self, lambda_cross, lambda_bin,
    )

    # Average gradients across all devices
    grads_A = jax.lax.pmean(grads_A, axis_name=cfg.PMAP_AXIS_NAME)
    grads_B = jax.lax.pmean(grads_B, axis_name=cfg.PMAP_AXIS_NAME)

    # Average metrics for consistent logging
    metrics = jax.lax.pmean(metrics, axis_name=cfg.PMAP_AXIS_NAME)

    updates_A, opt_state_A = optimizer_def.update(grads_A, opt_state_A, params_A)
    params_A = optax.apply_updates(params_A, updates_A)

    updates_B, opt_state_B = optimizer_def.update(grads_B, opt_state_B, params_B)
    params_B = optax.apply_updates(params_B, updates_B)

    return params_A, params_B, opt_state_A, opt_state_B, metrics


# ---------------------------------------------------------------------------
# Model init, checkpointing
# ---------------------------------------------------------------------------

def init_model_params(rng_key):
    """Initialize both models and return params + module instances."""
    enc_A, dec_A = Encoder(), Decoder()
    enc_B, dec_B = Encoder(), Decoder()

    dummy_img = jnp.zeros((1, cfg.IMG_SIZE, cfg.IMG_SIZE, 1), dtype=cfg.DTYPE)
    dummy_lat = jnp.zeros((1, cfg.LATENT_SIZE, cfg.LATENT_SIZE, 1), dtype=cfg.DTYPE)

    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    params_A = {
        'encoder': enc_A.init(k1, dummy_img)['params'],
        'decoder': dec_A.init(k2, dummy_lat)['params'],
    }
    params_B = {
        'encoder': enc_B.init(k3, dummy_img)['params'],
        'decoder': dec_B.init(k4, dummy_lat)['params'],
    }
    return params_A, params_B, enc_A, dec_A, enc_B, dec_B


def save_checkpoint(params_A, params_B, opt_state_A, opt_state_B, step, path):
    data = {
        'params_A': params_A,
        'params_B': params_B,
        'opt_state_A': opt_state_A,
        'opt_state_B': opt_state_B,
        'step': step,
    }
    with open(path, 'wb') as f:
        f.write(serialization.to_bytes(data))
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, params_A, params_B, opt_state_A, opt_state_B):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = serialization.from_bytes(
            {'params_A': params_A, 'params_B': params_B,
             'opt_state_A': opt_state_A, 'opt_state_B': opt_state_B,
             'step': 0},
            f.read()
        )
    print(f"  Resumed from step {data['step']}")
    return data


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(num_steps=None, log_interval=None, ckpt_interval=None,
         lr=None, lambda_self=None, lambda_cross=None, lambda_bin=None):
    """Main training loop. Args override config.py defaults when provided."""
    # Initialize distributed runtime (required for multihost TPUs, no-op on single host)
    jax.distributed.initialize()

    num_steps = num_steps or cfg.NUM_TRAINING_STEPS
    log_interval = log_interval or cfg.LOG_INTERVAL
    ckpt_interval = ckpt_interval or cfg.CHECKPOINT_INTERVAL
    lr = lr or cfg.LEARNING_RATE
    ls = lambda_self if lambda_self is not None else cfg.LAMBDA_SELF
    lc = lambda_cross if lambda_cross is not None else cfg.LAMBDA_CROSS
    lb = lambda_bin if lambda_bin is not None else cfg.LAMBDA_BINARIZATION

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # --- Device setup ---
    num_devices = jax.device_count()          # total across all hosts
    num_local_devices = jax.local_device_count()  # devices on this host
    num_processes = jax.process_count()       # number of hosts
    process_idx = jax.process_index()         # this host's index
    is_main = process_idx == 0
    use_pmap = num_devices > 1

    rng = jax.random.PRNGKey(42)
    rng, init_key = jax.random.split(rng)

    # --- Init models ---
    params_A, params_B, enc_A, dec_A, enc_B, dec_B = init_model_params(init_key)

    optimizer = optax.adam(lr)
    opt_state_A = optimizer.init(params_A)
    opt_state_B = optimizer.init(params_B)

    # --- Restore checkpoint (always in unreplicated form) ---
    start_step = 0
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, 'latest.msgpack')
    restored = load_checkpoint(ckpt_path, params_A, params_B, opt_state_A, opt_state_B)
    if restored:
        params_A = restored['params_A']
        params_B = restored['params_B']
        opt_state_A = restored['opt_state_A']
        opt_state_B = restored['opt_state_B']
        start_step = restored['step']

    # --- Print info (main process only) ---
    if is_main:
        param_count_A = sum(x.size for x in jax.tree.leaves(params_A))
        param_count_B = sum(x.size for x in jax.tree.leaves(params_B))
        print(f"Model A params: {param_count_A:,}")
        print(f"Model B params: {param_count_B:,}")
        print(f"Training from step {start_step} to {num_steps}")
        print(f"Loss weights: self={ls}, cross={lc}, bin={lb}")
        print(f"JAX devices: {num_devices} x {jax.devices()[0].device_kind}")
        if num_processes > 1:
            print(f"Mode: MULTIHOST DATA-PARALLEL (pmap over {num_devices} devices "
                  f"across {num_processes} hosts, {num_local_devices} local)")
        elif use_pmap:
            print(f"Mode: DATA-PARALLEL (pmap over {num_devices} devices, effective batch={num_devices})")
        else:
            print(f"Mode: SINGLE-DEVICE (jit)")

    # --- Replicate for pmap (replicates to LOCAL devices in multihost) ---
    if use_pmap:
        params_A = flax_utils.replicate(params_A)
        params_B = flax_utils.replicate(params_B)
        opt_state_A = flax_utils.replicate(opt_state_A)
        opt_state_B = flax_utils.replicate(opt_state_B)

    # Loss weight scalars
    # pmap path: plain floats (static_broadcasted args must be hashable)
    # jit path: jnp scalars for tracing
    ls_jnp = jnp.float32(ls)
    lc_jnp = jnp.float32(lc)
    lb_jnp = jnp.float32(lb)

    # --- Training loop ---
    t0 = time.time()
    if is_main:
        print("Compiling... (first step will be slow)")

    for step in range(start_step, num_steps):
        rng, data_key = jax.random.split(rng)

        if use_pmap:
            # Each host generates data for its LOCAL devices only
            # fold_in with process_idx ensures unique data across hosts
            local_data_key = jax.random.fold_in(data_key, process_idx)
            circle_batch, triangle_batch, counts = generate_training_batch(
                local_data_key, num_local_devices)
            count_n = counts[0]

            params_A, params_B, opt_state_A, opt_state_B, metrics = train_step_pmap(
                params_A, params_B, opt_state_A, opt_state_B,
                enc_A, dec_A, enc_B, dec_B, optimizer,
                circle_batch, triangle_batch,
                float(ls), float(lc), float(lb),
            )
        else:
            circle_img, triangle_img, count_n = generate_training_pair(data_key)
            circle_batch = circle_img[None, ...]
            triangle_batch = triangle_img[None, ...]

            params_A, params_B, opt_state_A, opt_state_B, metrics = train_step_jit(
                params_A, params_B, opt_state_A, opt_state_B,
                enc_A, dec_A, enc_B, dec_B, optimizer,
                circle_batch, triangle_batch,
                ls_jnp, lc_jnp, lb_jnp,
            )

        # --- Logging (main process only) ---
        if is_main and (step + 1) % log_interval == 0:
            # Unreplicate metrics if pmap (all replicas identical due to pmean)
            m = flax_utils.unreplicate(metrics) if use_pmap else metrics

            elapsed = time.time() - t0
            steps_done = step + 1 - start_step
            sps = steps_done / elapsed if elapsed > 0 else 0
            print(
                f"Step {step+1:>6d} | "
                f"loss={m['total_loss']:.4f} "
                f"self={m['loss_self']:.4f} "
                f"cross={m['loss_cross']:.6f} "
                f"bin={m['loss_bin']:.4f} | "
                f"N={count_n:>2d} "
                f"area_c={m['area_circle_in']:.4f} "
                f"area_xT={m['area_cross_T']:.4f} | "
                f"{sps:.1f} steps/s"
            )

        # --- Checkpointing (main process only, always save unreplicated) ---
        if is_main and (step + 1) % ckpt_interval == 0:
            if use_pmap:
                save_checkpoint(
                    flax_utils.unreplicate(params_A),
                    flax_utils.unreplicate(params_B),
                    flax_utils.unreplicate(opt_state_A),
                    flax_utils.unreplicate(opt_state_B),
                    step + 1, ckpt_path)
            else:
                save_checkpoint(params_A, params_B, opt_state_A, opt_state_B,
                                step + 1, ckpt_path)

    # --- Final save (main process only) ---
    if is_main:
        if use_pmap:
            save_checkpoint(
                flax_utils.unreplicate(params_A),
                flax_utils.unreplicate(params_B),
                flax_utils.unreplicate(opt_state_A),
                flax_utils.unreplicate(opt_state_B),
                num_steps, ckpt_path)
        else:
            save_checkpoint(params_A, params_B, opt_state_A, opt_state_B,
                            num_steps, ckpt_path)
        print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emergent number line models')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=None)
    parser.add_argument('--ckpt-interval', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lambda-self', type=float, default=None)
    parser.add_argument('--lambda-cross', type=float, default=None)
    parser.add_argument('--lambda-bin', type=float, default=None)
    args = parser.parse_args()

    main(
        num_steps=args.steps,
        log_interval=args.log_interval,
        ckpt_interval=args.ckpt_interval,
        lr=args.lr,
        lambda_self=args.lambda_self,
        lambda_cross=args.lambda_cross,
        lambda_bin=args.lambda_bin,
    )
