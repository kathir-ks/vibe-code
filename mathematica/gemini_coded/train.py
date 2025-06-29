# train.py

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from flax import serialization
import matplotlib.pyplot as plt 
import numpy as np 
from functools import partial
import os
import msgpack

# Import modules from our project
from config import IMG_SIZE, LATENT_SIZE, LEARNING_RATE, NUM_TRAINING_STEPS, LOG_INTERVAL, \
    LATENT_BINARIZATION_LOSS_WEIGHT, CHECKPOINT_DIR, CHECKPOINT_INTERVAL, \
    RESUME_FROM_CHECKPOINT, LAST_CHECKPOINT_PATH, DTYPE, TEST_MODE
from data_generator import generate_non_overlapping_circles, generate_non_overlapping_squares
from models import Encoder, Decoder
from utils import jax_count_objects_stopped_gradient # Ensure this imports the updated utility

class TrainState(train_state.TrainState):
    pass

def custom_loss(params_circle, params_square, 
                input_circles_img, num_input_circles,
                input_squares_img, num_input_squares,
                circle_encoder, circle_decoder, square_encoder, square_decoder):
    """
    Custom loss function for the two models.
    Model 1: Circles -> Latent -> Squares (decoded by Model 2's decoder)
    Model 2: Squares -> Latent -> Circles (decoded by Model 1's decoder)
    """
    # Encoder applications
    latent_from_circles = circle_encoder.apply({'params': params_circle['encoder']}, input_circles_img)
    latent_from_squares = square_encoder.apply({'params': params_square['encoder']}, input_squares_img)

    # Decoder applications
    decoded_squares = square_decoder.apply({'params': params_square['decoder']}, latent_from_circles)
    decoded_circles = circle_decoder.apply({'params': params_circle['decoder']}, latent_from_squares)

    # --- CRITICAL FIX: Apply stop_gradient before passing to the counting function ---
    # This prevents JAX from trying to differentiate through the non-differentiable
    # `jax_count_objects_stopped_gradient` function in a way that causes Tracer errors.
    # It ensures that `jax_count_objects_stopped_gradient` receives a primal (concrete) value.
    stopped_decoded_squares_2d = jax.lax.stop_gradient(decoded_squares[0, ..., 0])
    stopped_decoded_circles_2d = jax.lax.stop_gradient(decoded_circles[0, ..., 0])

    # Object counting (non-differentiable part)
    num_decoded_squares = jax_count_objects_stopped_gradient(stopped_decoded_squares_2d)
    num_decoded_circles = jax_count_objects_stopped_gradient(stopped_decoded_circles_2d)

    # Count matching loss
    loss_model1_count = jnp.abs(num_input_circles - num_decoded_squares)
    loss_model2_count = jnp.abs(num_input_squares - num_decoded_circles)

    # Latent space binarization loss
    latent_loss_circle = jnp.mean(jnp.abs(latent_from_circles - jnp.round(latent_from_circles)))
    latent_loss_square = jnp.mean(jnp.abs(latent_from_squares - jnp.round(latent_from_squares)))
    latent_binarization_loss = latent_loss_circle + latent_loss_square
    
    # Total loss
    total_loss = loss_model1_count + loss_model2_count + LATENT_BINARIZATION_LOSS_WEIGHT * latent_binarization_loss

    # Metrics for logging
    metrics = {
        'loss_model1_count': loss_model1_count,
        'loss_model2_count': loss_model2_count,
        'total_loss': total_loss,
        'num_input_circles': num_input_circles,
        'num_input_squares': num_input_squares,
        'num_decoded_squares': num_decoded_squares,
        'num_decoded_circles': num_decoded_circles,
        'latent_binarization_loss': latent_binarization_loss
    }
    
    return total_loss, metrics

# --- JIT-compiled Training Step ---
# Arguments:
# 0: params_circle (dynamic: PyTree of JAX Arrays)
# 1: params_square (dynamic: PyTree of JAX Arrays)
# 2: opt_state_circle (dynamic: PyTree of JAX Arrays)
# 3: opt_state_square (dynamic: PyTree of JAX Arrays)
# 4: input_circles_img (dynamic: JAX Array - changes every step)
# 5: num_input_circles (dynamic: JAX Array - changes every step)
# 6: input_squares_img (dynamic: JAX Array - changes every step)
# 7: num_input_squares (dynamic: JAX Array - changes every step)
# 8: circle_encoder (static: Flax Module instance)
# 9: circle_decoder (static: Flax Module instance)
# 10: square_encoder (static: Flax Module instance)
# 11: square_decoder (static: Flax Module instance)
# 12: optimizer_circle_def (static: Optax Optimizer instance)
# 13: optimizer_square_def (static: Optax Optimizer instance)
@partial(jax.jit, static_argnums=(8, 9, 10, 11, 12, 13))
def train_step(params_circle, params_square, opt_state_circle, opt_state_square, 
               input_circles_img, num_input_circles, 
               input_squares_img, num_input_squares, 
               circle_encoder, circle_decoder, square_encoder, square_decoder, 
               optimizer_circle_def, optimizer_square_def):
    
    # Compute loss and gradients
    grad_fn = jax.value_and_grad(custom_loss, argnums=(0, 1), has_aux=True)
    (loss, metrics), (grads_circle, grads_square) = grad_fn(
        params_circle, params_square, 
        input_circles_img, num_input_circles,
        input_squares_img, num_input_squares, # FIXED: Changed 'num_squares' to 'num_input_squares'
        circle_encoder, circle_decoder, square_encoder, square_decoder
    )
    
    # Apply updates for circle model
    updates_circle, opt_state_circle = optimizer_circle_def.update(grads_circle, opt_state_circle, params_circle)
    params_circle = optax.apply_updates(params_circle, updates_circle)

    # Apply updates for square model
    updates_square, opt_state_square = optimizer_square_def.update(grads_square, opt_state_square, params_square)
    params_square = optax.apply_updates(params_square, updates_square)

    return params_circle, params_square, opt_state_circle, opt_state_square, metrics

def initialize_models(rng_key):
    """Initializes Encoder and Decoder models for both circles and squares."""
    circle_encoder = Encoder(latent_dim=LATENT_SIZE)
    circle_decoder = Decoder(output_dim=IMG_SIZE)
    square_encoder = Encoder(latent_dim=LATENT_SIZE)
    square_decoder = Decoder(output_dim=IMG_SIZE)

    # Create dummy inputs for model initialization
    dummy_input = jnp.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=DTYPE)
    dummy_latent = jnp.zeros((1, LATENT_SIZE, LATENT_SIZE, 1), dtype=DTYPE)

    # Initialize parameters for circle models
    key_enc_c, key_dec_c = jax.random.split(rng_key)
    params_circle_encoder = circle_encoder.init(key_enc_c, dummy_input)['params']
    params_circle_decoder = circle_decoder.init(key_dec_c, dummy_latent)['params']
    params_circle = {'encoder': params_circle_encoder, 'decoder': params_circle_decoder}

    # Initialize parameters for square models
    key_enc_s, key_dec_s = jax.random.split(key_enc_c) # Split again from a new key
    params_square_encoder = square_encoder.init(key_enc_s, dummy_input)['params']
    params_square_decoder = square_decoder.init(key_dec_s, dummy_latent)['params']
    params_square = {'encoder': params_square_encoder, 'decoder': params_square_decoder}
    
    return params_circle, params_square, circle_encoder, circle_decoder, square_encoder, square_decoder


def save_checkpoint(params_circle, opt_state_circle, params_square, opt_state_square, step, filepath):
    """Saves model parameters and optimizer states to a file."""
    data = {
        'params_circle': serialization.to_state_dict(params_circle),
        'opt_state_circle': serialization.to_state_dict(opt_state_circle),
        'params_square': serialization.to_state_dict(params_square),
        'opt_state_square': serialization.to_state_dict(opt_state_square),
        'step': step
    }
    encoded = msgpack.packb(data, default=serialization.to_bytes, use_bin_type=True)
    with open(filepath, "wb") as f:
        f.write(encoded)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, params_circle_template, opt_state_circle_template, 
                    params_square_template, opt_state_square_template):
    """Loads model parameters and optimizer states from a file."""
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}. Starting from scratch.")
        return None, None, None, None, 0

    with open(filepath, "rb") as f:
        encoded = f.read()
    decoded = msgpack.unpackb(encoded, object_hook=serialization.from_bytes, raw=False)

    params_circle = serialization.from_state_dict(params_circle_template, decoded['params_circle'])
    opt_state_circle = serialization.from_state_dict(opt_state_circle_template, decoded['opt_state_circle'])
    params_square = serialization.from_state_dict(params_square_template, decoded['params_square'])
    opt_state_square = serialization.from_state_dict(opt_state_square_template, decoded['opt_state_square'])
    step = decoded['step']
    print(f"Checkpoint loaded from {filepath}, resuming from step {step}")
    return params_circle, opt_state_circle, params_square, opt_state_square, step


def main():
    rng_key = jax.random.key(0) # Initial PRNG key

    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Ensure checkpoint directory exists

    # Initialize models and their parameters
    rng_key, init_key = jax.random.split(rng_key)
    params_circle, params_square, \
    circle_encoder, circle_decoder, square_encoder, square_decoder = initialize_models(init_key)

    # Initialize optimizers and their states
    optimizer = optax.adam(learning_rate=LEARNING_RATE) # This is the optimizer definition object
    opt_state_circle = optimizer.init(params_circle)
    opt_state_square = optimizer.init(params_square)
    
    start_step = 0

    # Handle resuming from checkpoint
    if RESUME_FROM_CHECKPOINT:
        loaded_params_circle, loaded_opt_state_circle, \
        loaded_params_square, loaded_opt_state_square, loaded_step = \
            load_checkpoint(LAST_CHECKPOINT_PATH, params_circle, opt_state_circle, 
                            params_square, opt_state_square)
        
        if loaded_params_circle is not None:
            params_circle = loaded_params_circle
            opt_state_circle = loaded_opt_state_circle
            params_square = loaded_params_square
            opt_state_square = loaded_opt_state_square
            start_step = loaded_step + 1

    # Determine actual number of steps and checkpoint interval based on TEST_MODE
    num_training_steps_actual = 10 if TEST_MODE else NUM_TRAINING_STEPS
    checkpoint_interval_actual = 5 if TEST_MODE else CHECKPOINT_INTERVAL

    print(f"Starting training for {num_training_steps_actual} steps...")
    for step in range(start_step, num_training_steps_actual):
        rng_key, step_key_data_gen = jax.random.split(rng_key) # Split key for data generation

        # Generate data *outside* the JIT-compiled train_step for dynamic inputs
        input_circles_img, num_input_circles = generate_non_overlapping_circles(step_key_data_gen)
        # It's generally good practice to use a separate key or split the key for each data generation if they are independent
        rng_key, step_key_data_gen_s = jax.random.split(rng_key) 
        input_squares_img, num_input_squares = generate_non_overlapping_squares(step_key_data_gen_s) 
        
        # Add batch and channel dimensions for model input
        input_circles_img_batch = jnp.expand_dims(input_circles_img, (0, -1))
        input_squares_img_batch = jnp.expand_dims(input_squares_img, (0, -1))

        # Perform one training step
        params_circle, params_square, opt_state_circle, opt_state_square, metrics = \
            train_step(params_circle, params_square, opt_state_circle, opt_state_square, 
                       input_circles_img_batch, num_input_circles, 
                       input_squares_img_batch, num_input_squares, 
                       circle_encoder, circle_decoder, square_encoder, square_decoder, 
                       optimizer, optimizer) # Pass optimizer definition objects

        # Log training progress
        if (step + 1) % LOG_INTERVAL == 0 or (TEST_MODE and (step + 1) % checkpoint_interval_actual == 0):
            print(f"Step {step + 1}/{num_training_steps_actual}:")
            print(f"  Total Loss: {metrics['total_loss']:.4f}")
            print(f"  Input Circles: {metrics['num_input_circles']}, Decoded Squares: {metrics['num_decoded_squares']}")
            print(f"  Input Squares: {metrics['num_input_squares']}, Decoded Circles: {metrics['num_decoded_circles']}")
            print(f"  Latent Binarization Loss: {metrics['latent_binarization_loss']:.4f}")


        # Save checkpoint periodically
        if (step + 1) % checkpoint_interval_actual == 0:
            save_checkpoint(params_circle, opt_state_circle, params_square, opt_state_square, step + 1, LAST_CHECKPOINT_PATH)


    print("Training finished.")
    # Save final checkpoint
    save_checkpoint(params_circle, opt_state_circle, params_square, opt_state_square, num_training_steps_actual, LAST_CHECKPOINT_PATH) 
    
    # Additional logic for non-test mode if needed
    if not TEST_MODE:
        pass

if __name__ == '__main__':
    main()
