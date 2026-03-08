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

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from functools import partial
import os
import time

from config import (
    IMG_SIZE, LATENT_SIZE, LEARNING_RATE, NUM_TRAINING_STEPS,
    LOG_INTERVAL, CHECKPOINT_INTERVAL, CHECKPOINT_DIR,
    LAMBDA_SELF, LAMBDA_CROSS, LAMBDA_BINARIZATION, DTYPE,
)
from data import generate_training_pair
from models import Encoder, Decoder, binarize_ste


def compute_loss(params_A, params_B,
                 circle_img, triangle_img,
                 enc_A, dec_A, enc_B, dec_B):
    """Compute the combined loss for one training step.

    Args:
        params_A: {'encoder': ..., 'decoder': ...} for circle model
        params_B: {'encoder': ..., 'decoder': ...} for triangle model
        circle_img: [1, H, W, 1] circle input
        triangle_img: [1, H, W, 1] triangle input
        enc_A, dec_A, enc_B, dec_B: Flax module instances (static)
    """
    # --- Encode ---
    logits_A = enc_A.apply({'params': params_A['encoder']}, circle_img)
    logits_B = enc_B.apply({'params': params_B['encoder']}, triangle_img)

    z_A = binarize_ste(logits_A)  # binary latent from circles
    z_B = binarize_ste(logits_B)  # binary latent from triangles

    # --- Self-reconstruction (each model decodes its own latent) ---
    recon_C_logits = dec_A.apply({'params': params_A['decoder']}, z_A)  # circles -> circles
    recon_T_logits = dec_B.apply({'params': params_B['decoder']}, z_B)  # triangles -> triangles

    loss_self_A = optax.sigmoid_binary_cross_entropy(recon_C_logits, circle_img).mean()
    loss_self_B = optax.sigmoid_binary_cross_entropy(recon_T_logits, triangle_img).mean()
    loss_self = loss_self_A + loss_self_B

    # --- Cross-communication (area-matching proxy for count) ---
    cross_T_logits = dec_B.apply({'params': params_B['decoder']}, z_A)  # circles latent -> triangles
    cross_C_logits = dec_A.apply({'params': params_A['decoder']}, z_B)  # triangles latent -> circles

    cross_T_probs = jax.nn.sigmoid(cross_T_logits)
    cross_C_probs = jax.nn.sigmoid(cross_C_logits)

    area_circle_input = jnp.sum(circle_img)
    area_triangle_input = jnp.sum(triangle_img)
    area_cross_T = jnp.sum(cross_T_probs)
    area_cross_C = jnp.sum(cross_C_probs)

    # Normalize by image size to keep loss scale reasonable
    num_pixels = IMG_SIZE * IMG_SIZE
    loss_cross_AB = ((area_cross_T - area_circle_input) / num_pixels) ** 2
    loss_cross_BA = ((area_cross_C - area_triangle_input) / num_pixels) ** 2
    loss_cross = loss_cross_AB + loss_cross_BA

    # --- Latent binarization ---
    probs_A = jax.nn.sigmoid(logits_A)
    probs_B = jax.nn.sigmoid(logits_B)
    loss_bin = jnp.mean(probs_A * (1 - probs_A)) + jnp.mean(probs_B * (1 - probs_B))

    # --- Total ---
    total_loss = (LAMBDA_SELF * loss_self
                  + LAMBDA_CROSS * loss_cross
                  + LAMBDA_BINARIZATION * loss_bin)

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


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def train_step(params_A, params_B, opt_state_A, opt_state_B,
               enc_A, dec_A, enc_B, dec_B, optimizer_def,
               circle_img, triangle_img):
    """Single JIT-compiled training step."""

    grad_fn = jax.value_and_grad(compute_loss, argnums=(0, 1), has_aux=True)
    (loss, metrics), (grads_A, grads_B) = grad_fn(
        params_A, params_B,
        circle_img, triangle_img,
        enc_A, dec_A, enc_B, dec_B,
    )

    updates_A, opt_state_A = optimizer_def.update(grads_A, opt_state_A, params_A)
    params_A = optax.apply_updates(params_A, updates_A)

    updates_B, opt_state_B = optimizer_def.update(grads_B, opt_state_B, params_B)
    params_B = optax.apply_updates(params_B, updates_B)

    return params_A, params_B, opt_state_A, opt_state_B, metrics


def init_model_params(rng_key):
    """Initialize both models and return params + module instances."""
    enc_A, dec_A = Encoder(), Decoder()
    enc_B, dec_B = Encoder(), Decoder()

    dummy_img = jnp.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=DTYPE)
    dummy_lat = jnp.zeros((1, LATENT_SIZE, LATENT_SIZE, 1), dtype=DTYPE)

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


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    rng = jax.random.PRNGKey(42)
    rng, init_key = jax.random.split(rng)

    # Init models
    params_A, params_B, enc_A, dec_A, enc_B, dec_B = init_model_params(init_key)

    optimizer = optax.adam(LEARNING_RATE)
    opt_state_A = optimizer.init(params_A)
    opt_state_B = optimizer.init(params_B)

    start_step = 0

    # Try to resume
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'latest.msgpack')
    restored = load_checkpoint(ckpt_path, params_A, params_B, opt_state_A, opt_state_B)
    if restored:
        params_A = restored['params_A']
        params_B = restored['params_B']
        opt_state_A = restored['opt_state_A']
        opt_state_B = restored['opt_state_B']
        start_step = restored['step']

    param_count_A = sum(x.size for x in jax.tree.leaves(params_A))
    param_count_B = sum(x.size for x in jax.tree.leaves(params_B))
    print(f"Model A params: {param_count_A:,}")
    print(f"Model B params: {param_count_B:,}")
    print(f"Training from step {start_step} to {NUM_TRAINING_STEPS}")
    print(f"Loss weights: self={LAMBDA_SELF}, cross={LAMBDA_CROSS}, bin={LAMBDA_BINARIZATION}")

    t0 = time.time()

    for step in range(start_step, NUM_TRAINING_STEPS):
        rng, data_key = jax.random.split(rng)
        circle_img, triangle_img, count_n = generate_training_pair(data_key)

        # Add batch dim
        circle_batch = circle_img[None, ...]
        triangle_batch = triangle_img[None, ...]

        params_A, params_B, opt_state_A, opt_state_B, metrics = train_step(
            params_A, params_B, opt_state_A, opt_state_B,
            enc_A, dec_A, enc_B, dec_B, optimizer,
            circle_batch, triangle_batch,
        )

        if (step + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            steps_done = step + 1 - start_step
            sps = steps_done / elapsed if elapsed > 0 else 0
            print(
                f"Step {step+1:>6d} | "
                f"loss={metrics['total_loss']:.4f} "
                f"self={metrics['loss_self']:.4f} "
                f"cross={metrics['loss_cross']:.6f} "
                f"bin={metrics['loss_bin']:.4f} | "
                f"N={count_n:>2d} "
                f"area_c={metrics['area_circle_in']:.4f} "
                f"area_xT={metrics['area_cross_T']:.4f} | "
                f"{sps:.1f} steps/s"
            )

        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(params_A, params_B, opt_state_A, opt_state_B,
                            step + 1, ckpt_path)

    # Final save
    save_checkpoint(params_A, params_B, opt_state_A, opt_state_B,
                    NUM_TRAINING_STEPS, ckpt_path)
    print("Training complete.")


if __name__ == '__main__':
    main()
