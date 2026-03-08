# inference.py

import jax
import jax.numpy as jnp
from flax import serialization
import matplotlib.pyplot as plt
import numpy as np
import os

from config import IMG_SIZE, LATENT_SIZE, LAST_CHECKPOINT_PATH, DTYPE
from data_generator import generate_paired_images
from models import Encoder, Decoder

def load_models_and_params():
    """Initializes models and loads parameters from the latest checkpoint."""
    circle_encoder = Encoder(latent_dim=LATENT_SIZE)
    circle_decoder = Decoder(output_dim=IMG_SIZE)
    square_encoder = Encoder(latent_dim=LATENT_SIZE)
    square_decoder = Decoder(output_dim=IMG_SIZE)

    dummy_input = jnp.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=DTYPE)
    dummy_latent = jnp.zeros((1, LATENT_SIZE, LATENT_SIZE, 1), dtype=DTYPE)
    
    key = jax.random.key(0)
    params_circle_template = {'encoder': circle_encoder.init(key, dummy_input)['params'],
                              'decoder': circle_decoder.init(key, dummy_latent)['params']}
    params_square_template = {'encoder': square_encoder.init(key, dummy_input)['params'],
                              'decoder': square_decoder.init(key, dummy_latent)['params']}

    if not os.path.exists(LAST_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {LAST_CHECKPOINT_PATH}. Please run train.py first.")

    with open(LAST_CHECKPOINT_PATH, "rb") as f:
        encoded_bytes = f.read()
    
    template = {'params_circle': params_circle_template, 'opt_state_circle': None,
                'params_square': params_square_template, 'opt_state_square': None, 'step': 0}
    
    decoded = serialization.from_bytes(template, encoded_bytes)
    params_circle = decoded['params_circle']
    params_square = decoded['params_square']
    
    print(f"Model weights loaded successfully from step {decoded['step']}.")
    
    models = (circle_encoder, circle_decoder, square_encoder, square_decoder)
    params = (params_circle, params_square)
    return models, params

def run_inference():
    try:
        models, params = load_models_and_params()
    except FileNotFoundError as e:
        print(e)
        return

    (circle_encoder, circle_decoder, square_encoder, square_decoder) = models
    (params_circle, params_square) = params

    # Generate a new, unseen pair of images for inference
    rng_key = jax.random.key(42)
    gt_circles_img, gt_squares_img = generate_paired_images(rng_key)

    # Prepare inputs
    circle_batch = jnp.expand_dims(gt_circles_img, (0, -1))
    square_batch = jnp.expand_dims(gt_squares_img, (0, -1))

    # --- Path 1: Circles -> Latent -> Squares ---
    latent_c = circle_encoder.apply({'params': params_circle['encoder']}, circle_batch)
    decoded_s_logits = square_decoder.apply({'params': params_square['decoder']}, latent_c)
    decoded_s_img = np.array(jax.nn.sigmoid(decoded_s_logits)[0, ..., 0])
    
    # --- Path 2: Squares -> Latent -> Circles ---
    latent_s = square_encoder.apply({'params': params_square['encoder']}, square_batch)
    decoded_c_logits = circle_decoder.apply({'params': params_circle['decoder']}, latent_s)
    decoded_c_img = np.array(jax.nn.sigmoid(decoded_c_logits)[0, ..., 0])

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Model Inference: Shape Translation", fontsize=16)

    axes[0, 0].imshow(gt_circles_img, cmap='gray')
    axes[0, 0].set_title("Input Circles (Ground Truth)")
    axes[0, 1].imshow(np.array(latent_c)[0, ..., 0], cmap='gray')
    axes[0, 1].set_title("Latent Representation")
    axes[0, 2].imshow(decoded_s_img, cmap='gray')
    axes[0, 2].set_title("Decoded Squares")

    axes[1, 0].imshow(gt_squares_img, cmap='gray')
    axes[1, 0].set_title("Input Squares (Ground Truth)")
    axes[1, 1].imshow(np.array(latent_s)[0, ..., 0], cmap='gray')
    axes[1, 1].set_title("Latent Representation")
    axes[1, 2].imshow(decoded_c_img, cmap='gray')
    axes[1, 2].set_title("Decoded Circles")
    
    for ax in axes.flatten():
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    run_inference()
