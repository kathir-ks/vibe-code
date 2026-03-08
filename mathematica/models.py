# models.py — Encoder and Decoder for the emergent number line experiment
#
# Architecture:
#   Encoder: 1080x1080x1 -> 32x32x1 (binary latent via STE)
#   Decoder: 32x32x1 -> 1080x1080x1 (logits, sigmoid applied in loss)
#
# Downsampling path: 1080 -> 540 -> 270 -> 135 -> 68 -> 34 --(VALID 3x3)--> 32
# Upsampling path:   32 --(VALID 3x3)--> 34 -> 68 -> 135* -> 270 -> 540 -> 1080
#   * crop from 136 to 135

import flax.linen as nn
import jax
import jax.numpy as jnp
from config import IMG_SIZE, LATENT_SIZE


class Encoder(nn.Module):
    """Compresses IMG_SIZE x IMG_SIZE x 1 -> LATENT_SIZE x LATENT_SIZE x 1."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64,  (5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(128, (5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(256, (5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(512, (5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(512, (5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        # 34x34 -> 32x32 via VALID padding
        x = nn.Conv(1, (3, 3), strides=(1, 1), padding='VALID')(x)
        return x  # raw logits; binarization applied externally


class Decoder(nn.Module):
    """Reconstructs LATENT_SIZE x LATENT_SIZE x 1 -> IMG_SIZE x IMG_SIZE x 1."""

    @nn.compact
    def __call__(self, z):
        # 32 -> 34
        x = nn.ConvTranspose(512, (3, 3), strides=(1, 1), padding='VALID')(z)
        x = nn.relu(x)
        # 34 -> 68
        x = nn.ConvTranspose(512, (5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        # 68 -> 136, crop to 135
        x = nn.ConvTranspose(256, (5, 5), strides=(2, 2), padding='SAME')(x)
        x = x[:, :135, :135, :]
        x = nn.relu(x)
        # 135 -> 270
        x = nn.ConvTranspose(128, (5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        # 270 -> 540
        x = nn.ConvTranspose(64, (5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        # 540 -> 1080
        x = nn.ConvTranspose(1, (5, 5), strides=(2, 2), padding='SAME')(x)
        return x  # raw logits; sigmoid applied in loss for numerical stability


def binarize_ste(logits):
    """Binarize latent space using Straight-Through Estimator.

    Forward: sigmoid -> round to {0, 1}
    Backward: gradients flow through sigmoid (as if round didn't happen)
    """
    probs = jax.nn.sigmoid(logits)
    binary = jnp.round(probs)
    return probs + jax.lax.stop_gradient(binary - probs)
