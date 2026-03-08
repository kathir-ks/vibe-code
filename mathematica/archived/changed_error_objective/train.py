# train.py

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from functools import partial
import os
import msgpack

from config import (IMG_SIZE, LATENT_SIZE, LEARNING_RATE, NUM_TRAINING_STEPS, LOG_INTERVAL,
                    LATENT_BINARIZATION_LOSS_WEIGHT, CHECKPOINT_DIR, CHECKPOINT_INTERVAL,
                    RESUME_FROM_CHECKPOINT, LAST_CHECKPOINT_PATH, DTYPE)
from data_generator import generate_paired_images
from models import Encoder, Decoder

def custom_loss(params_circle, params_square, input_circles_img, input_squares_img,
                circle_encoder, circle_decoder, square_encoder, square_decoder):
    """Calculates loss based on pixel-wise reconstruction of paired images."""
    latent_from_circles = circle_encoder.apply({'params': params_circle['encoder']}, input_circles_img)
    latent_from_squares = square_encoder.apply({'params': params_square['encoder']}, input_squares_img)
    decoded_squares_logits = square_decoder.apply({'params': params_square['decoder']}, latent_from_circles)
    decoded_circles_logits = circle_decoder.apply({'params': params_circle['decoder']}, latent_from_squares)

    loss_reconstruction_c2s = optax.sigmoid_binary_cross_entropy(
        logits=decoded_squares_logits, labels=input_squares_img).mean()
    loss_reconstruction_s2c = optax.sigmoid_binary_cross_entropy(
        logits=decoded_circles_logits, labels=input_circles_img).mean()

    latent_loss_circle = jnp.mean(jnp.abs(latent_from_circles - jnp.round(latent_from_circles)))
    latent_loss_square = jnp.mean(jnp.abs(latent_from_squares - jnp.round(latent_from_squares)))
    latent_binarization_loss = latent_loss_circle + latent_loss_square
    
    total_loss = loss_reconstruction_c2s + loss_reconstruction_s2c + \
                 LATENT_BINARIZATION_LOSS_WEIGHT * latent_binarization_loss
    
    metrics = {
        'total_loss': total_loss,
        'loss_reconstruction': loss_reconstruction_c2s + loss_reconstruction_s2c,
        'latent_binarization_loss': latent_binarization_loss
    }
    return total_loss, metrics

@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def train_step(params_circle, params_square, opt_state_circle, opt_state_square,
               circle_encoder, circle_decoder, square_encoder, square_decoder,
               optimizer_def, input_circles_img, input_squares_img):
    
    grad_fn = jax.value_and_grad(custom_loss, argnums=(0, 1), has_aux=True)
    (loss, metrics), (grads_circle, grads_square) = grad_fn(
        params_circle, params_square, input_circles_img, input_squares_img,
        circle_encoder, circle_decoder, square_encoder, square_decoder)
    
    updates_circle, opt_state_circle = optimizer_def.update(grads_circle, opt_state_circle, params_circle)
    params_circle = optax.apply_updates(params_circle, updates_circle)

    updates_square, opt_state_square = optimizer_def.update(grads_square, opt_state_square, params_square)
    params_square = optax.apply_updates(params_square, updates_square)

    return params_circle, params_square, opt_state_circle, opt_state_square, metrics

def initialize_models_and_state(rng_key):
    """Initializes models, parameters, and optimizer states."""
    circle_encoder = Encoder(latent_dim=LATENT_SIZE)
    circle_decoder = Decoder(output_dim=IMG_SIZE)
    square_encoder = Encoder(latent_dim=LATENT_SIZE)
    square_decoder = Decoder(output_dim=IMG_SIZE)

    dummy_input = jnp.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=DTYPE)
    dummy_latent = jnp.zeros((1, LATENT_SIZE, LATENT_SIZE, 1), dtype=DTYPE)

    key_c, key_s = jax.random.split(rng_key)
    params_circle = {'encoder': circle_encoder.init(key_c, dummy_input)['params'],
                     'decoder': circle_decoder.init(key_c, dummy_latent)['params']}
    params_square = {'encoder': square_encoder.init(key_s, dummy_input)['params'],
                     'decoder': square_decoder.init(key_s, dummy_latent)['params']}
    
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state_circle = optimizer.init(params_circle)
    opt_state_square = optimizer.init(params_square)
    
    models = (circle_encoder, circle_decoder, square_encoder, square_decoder)
    params = (params_circle, params_square)
    opt_states = (opt_state_circle, opt_state_square)
    
    return models, params, opt_states, optimizer

def save_checkpoint(params, opt_states, step, filepath):
    """Saves model parameters and optimizer states."""
    (params_circle, params_square) = params
    (opt_state_circle, opt_state_square) = opt_states
    data = {'params_circle': params_circle, 'opt_state_circle': opt_state_circle,
            'params_square': params_square, 'opt_state_square': opt_state_square, 'step': step}
    serialized_data = serialization.to_bytes(data)
    with open(filepath, "wb") as f:
        f.write(serialized_data)
    print(f"Checkpoint saved to {filepath}")

def main():
    rng_key = jax.random.key(0)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    models, params, opt_states, optimizer = initialize_models_and_state(rng_key)
    (circle_encoder, circle_decoder, square_encoder, square_decoder) = models
    (params_circle, params_square) = params
    (opt_state_circle, opt_state_square) = opt_states
    
    start_step = 0 # Resume logic can be added here if needed

    num_steps = NUM_TRAINING_STEPS
    print(f"Starting training for {num_steps} steps...")

    for step in range(start_step, num_steps):
        rng_key, data_key = jax.random.split(rng_key)
        
        circle_img, square_img = generate_paired_images(data_key)
        circle_batch = jnp.expand_dims(circle_img, (0, -1))
        square_batch = jnp.expand_dims(square_img, (0, -1))

        params_circle, params_square, opt_state_circle, opt_state_square, metrics = train_step(
            params_circle, params_square, opt_state_circle, opt_state_square,
            circle_encoder, circle_decoder, square_encoder, square_decoder,
            optimizer, circle_batch, square_batch
        )

        if (step + 1) % LOG_INTERVAL == 0:
            print(f"Step {step + 1}/{num_steps}: "
                  f"Total Loss: {metrics['total_loss']:.4f}, "
                  f"Reconstruction Loss: {metrics['loss_reconstruction']:.4f}, "
                  f"Latent Loss: {metrics['latent_binarization_loss']:.4f}")

        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{step}_checkpoint.msgpack")
            save_checkpoint((params_circle, params_square), (opt_state_circle, opt_state_square), step + 1, checkpoint_path)

if __name__ == '__main__':
    main()
