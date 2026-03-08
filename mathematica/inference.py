# inference.py — Run inference on trained models and visualize results

import jax
import jax.numpy as jnp
import numpy as np
import os
import argparse
from flax import serialization

from config import IMG_SIZE, LATENT_SIZE, CHECKPOINT_DIR, DTYPE
from data import generate_circle_image, generate_triangle_image
from models import Encoder, Decoder, binarize_ste


def load_params(ckpt_path):
    """Load model parameters from a checkpoint."""
    enc_A, dec_A = Encoder(), Decoder()
    enc_B, dec_B = Encoder(), Decoder()

    dummy_img = jnp.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=DTYPE)
    dummy_lat = jnp.zeros((1, LATENT_SIZE, LATENT_SIZE, 1), dtype=DTYPE)

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    params_A = {
        'encoder': enc_A.init(k1, dummy_img)['params'],
        'decoder': dec_A.init(k2, dummy_lat)['params'],
    }
    params_B = {
        'encoder': enc_B.init(k3, dummy_img)['params'],
        'decoder': dec_B.init(k4, dummy_lat)['params'],
    }

    # Only need params, not optimizer state, but we restore the full dict
    import optax
    optimizer = optax.adam(1e-4)
    template = {
        'params_A': params_A, 'params_B': params_B,
        'opt_state_A': optimizer.init(params_A),
        'opt_state_B': optimizer.init(params_B),
        'step': 0,
    }

    with open(ckpt_path, 'rb') as f:
        data = serialization.from_bytes(template, f.read())

    return data['params_A'], data['params_B'], enc_A, dec_A, enc_B, dec_B, data['step']


def run_inference(ckpt_path, seed=42):
    """Load models, generate test images, and show results."""
    params_A, params_B, enc_A, dec_A, enc_B, dec_B, step = load_params(ckpt_path)
    print(f"Loaded checkpoint from step {step}")

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    # Generate test images
    circle_img, n_circles = generate_circle_image(k1)
    triangle_img, n_triangles = generate_triangle_image(k2)

    print(f"Generated circle image with {n_circles} circles")
    print(f"Generated triangle image with {n_triangles} triangles")

    # Self-reconstruction: circles -> enc_A -> dec_A -> circles'
    circle_batch = circle_img[None, ...]
    logits_A = enc_A.apply({'params': params_A['encoder']}, circle_batch)
    z_A = binarize_ste(logits_A)
    recon_C = jax.nn.sigmoid(dec_A.apply({'params': params_A['decoder']}, z_A))

    # Self-reconstruction: triangles -> enc_B -> dec_B -> triangles'
    triangle_batch = triangle_img[None, ...]
    logits_B = enc_B.apply({'params': params_B['encoder']}, triangle_batch)
    z_B = binarize_ste(logits_B)
    recon_T = jax.nn.sigmoid(dec_B.apply({'params': params_B['decoder']}, z_B))

    # Cross-communication: circles -> enc_A -> dec_B -> triangles'
    cross_T = jax.nn.sigmoid(dec_B.apply({'params': params_B['decoder']}, z_A))

    # Cross-communication: triangles -> enc_B -> dec_A -> circles'
    cross_C = jax.nn.sigmoid(dec_A.apply({'params': params_A['decoder']}, z_B))

    # Print area comparisons
    print(f"\nArea comparisons (normalized):")
    print(f"  Circle input area:        {np.sum(np.array(circle_img)) / (IMG_SIZE**2):.4f}")
    print(f"  Self-recon circle area:   {np.sum(np.array(recon_C)) / (IMG_SIZE**2):.4f}")
    print(f"  Cross triangle area:      {np.sum(np.array(cross_T)) / (IMG_SIZE**2):.4f}")
    print(f"  Triangle input area:      {np.sum(np.array(triangle_img)) / (IMG_SIZE**2):.4f}")
    print(f"  Self-recon triangle area: {np.sum(np.array(recon_T)) / (IMG_SIZE**2):.4f}")
    print(f"  Cross circle area:        {np.sum(np.array(cross_C)) / (IMG_SIZE**2):.4f}")

    # Latent space stats
    z_A_np = np.array(z_A[0, :, :, 0])
    z_B_np = np.array(z_B[0, :, :, 0])
    print(f"\nLatent space stats:")
    print(f"  Z_A: {np.sum(z_A_np == 1)} ones, {np.sum(z_A_np == 0)} zeros out of {z_A_np.size}")
    print(f"  Z_B: {np.sum(z_B_np == 1)} ones, {np.sum(z_B_np == 0)} zeros out of {z_B_np.size}")

    # Try to save images if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        axes[0, 0].imshow(np.array(circle_img[:, :, 0]), cmap='gray')
        axes[0, 0].set_title(f'Input Circles (N={n_circles})')
        axes[0, 1].imshow(z_A_np, cmap='gray')
        axes[0, 1].set_title('Latent Z_A')
        axes[0, 2].imshow(np.array(recon_C[0, :, :, 0]), cmap='gray')
        axes[0, 2].set_title('Self-Recon Circles')
        axes[0, 3].imshow(np.array(cross_T[0, :, :, 0]), cmap='gray')
        axes[0, 3].set_title('Cross → Triangles')

        axes[1, 0].imshow(np.array(triangle_img[:, :, 0]), cmap='gray')
        axes[1, 0].set_title(f'Input Triangles (N={n_triangles})')
        axes[1, 1].imshow(z_B_np, cmap='gray')
        axes[1, 1].set_title('Latent Z_B')
        axes[1, 2].imshow(np.array(recon_T[0, :, :, 0]), cmap='gray')
        axes[1, 2].set_title('Self-Recon Triangles')
        axes[1, 3].imshow(np.array(cross_C[0, :, :, 0]), cmap='gray')
        axes[1, 3].set_title('Cross → Circles')

        for ax in axes.flat:
            ax.axis('off')

        plt.suptitle(f'Emergent Number Line — Step {step}', fontsize=16)
        plt.tight_layout()

        out_path = os.path.join(CHECKPOINT_DIR, f'inference_step_{step}.png')
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved visualization to {out_path}")
        plt.close()
    except ImportError:
        print("\nmatplotlib not available — skipping image save")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=os.path.join(CHECKPOINT_DIR, 'latest.msgpack'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_inference(args.ckpt, args.seed)
