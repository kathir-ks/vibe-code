# inference.py

import jax
import jax.numpy as jnp
from flax import serialization
import matplotlib.pyplot as plt
import numpy as np
import os
import msgpack

# Import modules from our project
from config import IMG_SIZE, LATENT_SIZE, CHECKPOINT_DIR, LAST_CHECKPOINT_PATH, DTYPE
from data_generator import generate_non_overlapping_circles, generate_non_overlapping_squares
from models import Encoder, Decoder
from utils import count_objects # Use the raw numpy count for visualization/verification

def initialize_empty_models():
    """Initializes model structures with dummy parameters for loading."""
    # Return individual instances here too
    circle_encoder = Encoder(latent_dim=LATENT_SIZE)
    circle_decoder = Decoder(output_dim=IMG_SIZE)
    square_encoder = Encoder(latent_dim=LATENT_SIZE)
    square_decoder = Decoder(output_dim=IMG_SIZE)

    dummy_input = jnp.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=DTYPE)
    dummy_latent = jnp.zeros((1, LATENT_SIZE, LATENT_SIZE, 1), dtype=DTYPE)

    dummy_key = jax.random.key(0) # Use a dummy key for init to get structure
    params_circle_encoder_template = circle_encoder.init(dummy_key, dummy_input)['params']
    params_circle_decoder_template = circle_decoder.init(dummy_key, dummy_latent)['params']
    params_circle_template = {'encoder': params_circle_encoder_template, 'decoder': params_circle_decoder_template}

    params_square_encoder_template = square_encoder.init(dummy_key, dummy_input)['params']
    params_square_decoder_template = square_decoder.init(dummy_key, dummy_latent)['params']
    params_square_template = {'encoder': params_square_encoder_template, 'decoder': params_square_decoder_template}

    # Return individual model instances alongside templates
    return params_circle_template, params_square_template, \
           circle_encoder, circle_decoder, square_encoder, square_decoder

def load_inference_weights(filepath, params_circle_template, params_square_template):
    """Loads only model parameters from a checkpoint file for inference."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}. Please train the model first.")

    with open(filepath, "rb") as f:
        encoded = f.read()
    decoded = msgpack.unpackb(encoded, object_hook=serialization.from_bytes, raw=False)

    params_circle = serialization.from_state_dict(params_circle_template, decoded['params_circle'])
    params_square = serialization.from_state_dict(params_square_template, decoded['params_square'])
    
    print(f"Model weights loaded successfully from {filepath}")
    return params_circle, params_square

def run_inference():
    print("Initializing models for inference...")
    params_circle_template, params_square_template, \
    circle_encoder, circle_decoder, square_encoder, square_decoder = initialize_empty_models()

    print(f"Attempting to load weights from: {LAST_CHECKPOINT_PATH}")
    try:
        params_circle, params_square = load_inference_weights(LAST_CHECKPOINT_PATH, params_circle_template, params_square_template)
    except FileNotFoundError as e:
        print(e)
        print("Please run `train.py` (or `test_suite.py`) first to generate a checkpoint.")
        return

    rng_key = jax.random.key(12345) # Use a new key for inference

    # Path 1: Circles -> Latent (Circle Encoder) -> Squares (Square Decoder)
    print("\nRunning inference for Circle -> Squares path:")
    rng_key, gen_key = jax.random.split(rng_key)
    input_circles_img, num_input_circles = generate_non_overlapping_circles(gen_key)
    input_circles_img_batch = jnp.expand_dims(input_circles_img, (0, -1))

    latent_from_circles = circle_encoder.apply({'params': params_circle['encoder']}, input_circles_img_batch)
    decoded_squares = square_decoder.apply({'params': params_square['decoder']}, latent_from_circles)

    num_decoded_squares_actual = count_objects(decoded_squares[0, ..., 0])

    print(f"Input Circles: {num_input_circles}")
    print(f"Decoded Squares (Target count: {num_input_circles}): {num_decoded_squares_actual}")

    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    axes1[0].imshow(np.array(input_circles_img), cmap='gray')
    axes1[0].set_title(f"Input Circles: {num_input_circles}")
    axes1[0].axis('off')

    axes1[1].imshow(np.array(latent_from_circles[0, ..., 0]), cmap='gray')
    axes1[1].set_title("Latent Space (from Circles)")
    axes1[1].axis('off')

    axes1[2].imshow(np.array(decoded_squares[0, ..., 0]), cmap='gray')
    axes1[2].set_title(f"Decoded Squares: {num_decoded_squares_actual}")
    axes1[2].axis('off')
    plt.suptitle("Inference: Circles Encoded by Circle Model, Decoded by Square Model")
    plt.tight_layout()
    plt.show()

    # Path 2: Squares -> Latent (Square Encoder) -> Circles (Circle Decoder)
    print("\nRunning inference for Square -> Circles path:")
    rng_key, gen_key = jax.random.split(rng_key)
    input_squares_img, num_input_squares = generate_non_overlapping_squares(gen_key)
    input_squares_img_batch = jnp.expand_dims(input_squares_img, (0, -1))

    latent_from_squares = square_encoder.apply({'params': params_square['encoder']}, input_squares_img_batch)
    decoded_circles = circle_decoder.apply({'params': params_circle['decoder']}, latent_from_squares)

    num_decoded_circles_actual = count_objects(decoded_circles[0, ..., 0])

    print(f"Input Squares: {num_input_squares}")
    print(f"Decoded Circles (Target count: {num_input_squares}): {num_decoded_circles_actual}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    axes2[0].imshow(np.array(input_squares_img), cmap='gray')
    axes2[0].set_title(f"Input Squares: {num_input_squares}")
    axes2[0].axis('off')

    axes2[1].imshow(np.array(latent_from_squares[0, ..., 0]), cmap='gray')
    axes2[1].set_title("Latent Space (from Squares)")
    axes2[1].axis('off')

    axes2[2].imshow(np.array(decoded_circles[0, ..., 0]), cmap='gray')
    axes2[2].set_title(f"Decoded Circles: {num_decoded_circles_actual}")
    axes2[2].axis('off')
    plt.suptitle("Inference: Squares Encoded by Square Model, Decoded by Circle Model")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_inference()
