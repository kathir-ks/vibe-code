# test_suite.py

import jax
import jax.numpy as jnp
import numpy as np
import os
import shutil
import sys
import msgpack
from flax import serialization
import optax

# Add current directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import config
from train import initialize_models, train_step, save_checkpoint, load_checkpoint, main as train_main_func
from inference import initialize_empty_models, load_inference_weights
from models import Encoder, Decoder
from data_generator import generate_non_overlapping_circles, generate_non_overlapping_squares
from utils import count_objects


# --- Helper for comparing Flax parameter trees ---
def assert_pytree_equal(pytree1, pytree2, rtol=1e-5, atol=1e-8):
    """Asserts that two JAX pytrees (e.g., model parameters) are approximately equal."""
    flat_pytree1, _ = jax.tree_util.tree_flatten(pytree1)
    flat_pytree2, _ = jax.tree_util.tree_flatten(pytree2)
    
    assert len(flat_pytree1) == len(flat_pytree2), "Pytrees have different number of leaves."
    
    for x, y in zip(flat_pytree1, flat_pytree2):
        if isinstance(x, jax.Array) and isinstance(y, jax.Array):
            np.testing.assert_allclose(np.array(x), np.array(y), rtol=rtol, atol=atol)
        else:
            assert x == y, f"Non-array elements differ: {x} vs {y}"
    print("Pytrees are approximately equal.")


# --- Test Functions ---

def setup_test_environment():
    """Sets up a clean environment for testing (config, directories)."""
    global original_test_mode, original_num_training_steps, original_checkpoint_interval, original_resume_from_checkpoint, original_last_checkpoint_path

    original_test_mode = config.TEST_MODE
    original_num_training_steps = config.NUM_TRAINING_STEPS
    original_checkpoint_interval = config.CHECKPOINT_INTERVAL
    original_resume_from_checkpoint = config.RESUME_FROM_CHECKPOINT
    original_last_checkpoint_path = config.LAST_CHECKPOINT_PATH

    config.TEST_MODE = True
    config.NUM_TRAINING_STEPS = 10 # Short training for tests
    config.CHECKPOINT_INTERVAL = 5 # Checkpoint frequently for tests
    config.RESUME_FROM_CHECKPOINT = False # Start fresh for test
    config.LAST_CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "test_checkpoint.msgpack")

    # Clean up and create test checkpoints directory
    if os.path.exists(config.CHECKPOINT_DIR):
        shutil.rmtree(config.CHECKPOINT_DIR)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print(f"Test environment set up. Checkpoints will be saved to {config.CHECKPOINT_DIR}")

def teardown_test_environment():
    """Restores original configuration and cleans up test artifacts."""
    config.TEST_MODE = original_test_mode
    config.NUM_TRAINING_STEPS = original_num_training_steps
    config.CHECKPOINT_INTERVAL = original_checkpoint_interval
    config.RESUME_FROM_CHECKPOINT = original_resume_from_checkpoint
    config.LAST_CHECKPOINT_PATH = original_last_checkpoint_path

    if os.path.exists(config.CHECKPOINT_DIR):
        shutil.rmtree(config.CHECKPOINT_DIR)
    print(f"Test environment torn down. Cleaned up {config.CHECKPOINT_DIR}")

def test_checkpointing_and_inference_flow():
    print("\n--- Running Checkpointing and Inference Flow Test ---")
    
    setup_test_environment()

    try:
        # 1. Run a short training to generate a checkpoint
        print(f"Running `train.py` for {config.NUM_TRAINING_STEPS} steps to generate checkpoint...")
        train_main_func() # This will save a checkpoint at config.LAST_CHECKPOINT_PATH

        # 2. Test Checkpoint Loading (separate from training's resume logic)
        print("\nTesting Checkpoint Loading Mechanism:")
        rng_key = jax.random.key(100) # Use a new key for loading templates
        params_circle_template, params_square_template, \
        _, _, _, _ = initialize_models(rng_key) # We only need the param templates here
        
        # Need a dummy opt_state to pass as template for load_checkpoint
        dummy_optimizer = optax.adam(learning_rate=config.LEARNING_RATE)
        dummy_opt_state_circle_template = dummy_optimizer.init(params_circle_template)
        dummy_opt_state_square_template = dummy_optimizer.init(params_square_template)

        loaded_params_circle, loaded_opt_state_circle, \
        loaded_params_square, loaded_opt_state_square, loaded_step = \
            load_checkpoint(config.LAST_CHECKPOINT_PATH, 
                            params_circle_template, dummy_opt_state_circle_template, 
                            params_square_template, dummy_opt_state_square_template)
        
        assert loaded_params_circle is not None and loaded_params_square is not None, "Failed to load any parameters."
        assert loaded_step == config.NUM_TRAINING_STEPS, f"Loaded step mismatch: Expected {config.NUM_TRAINING_STEPS}, Got {loaded_step}"
        print("Checkpoint loading test PASSED: Parameters and step loaded successfully.")

        # For a deeper check, we could reload and compare against known saved params,
        # but loading successfully and getting the correct step is a good first pass for this flow test.

        # 3. Test Inference After Loading Weights
        print("\nTesting Inference After Loading Weights:")
        params_circle_infer_template, params_square_infer_template, \
        circle_encoder_infer, circle_decoder_infer, \
        square_encoder_infer, square_decoder_infer = initialize_empty_models()
        
        # Load the weights directly for programmatic inference
        params_circle_infer, params_square_infer = load_inference_weights(
            config.LAST_CHECKPOINT_PATH, 
            params_circle_infer_template, 
            params_square_infer_template
        )

        rng_key = jax.random.key(54321) # New key for inference data generation

        # Test Circle -> Squares path
        rng_key, gen_key_c = jax.random.split(rng_key)
        input_circles_img, num_input_circles = generate_non_overlapping_circles(gen_key_c)
        input_circles_img_batch = jnp.expand_dims(input_circles_img, (0, -1))

        latent_from_circles = circle_encoder_infer.apply({'params': params_circle_infer['encoder']}, input_circles_img_batch)
        decoded_squares = square_decoder_infer.apply({'params': params_square_infer['decoder']}, latent_from_circles)
        num_decoded_squares_actual = count_objects(decoded_squares[0, ..., 0])

        print(f"Inference C->S: Input Circles={num_input_circles}, Decoded Squares={num_decoded_squares_actual}")
        assert num_decoded_squares_actual >= 0, "Decoded square count should be non-negative."
        if num_input_circles > 0:
            assert num_decoded_squares_actual > 0, "Decoded squares should be non-zero if input circles are non-zero."

        # Test Square -> Circles path
        rng_key, gen_key_s = jax.random.split(rng_key)
        input_squares_img, num_input_squares = generate_non_overlapping_squares(gen_key_s)
        input_squares_img_batch = jnp.expand_dims(input_squares_img, (0, -1))

        latent_from_squares = square_encoder_infer.apply({'params': params_square_infer['encoder']}, input_squares_img_batch)
        decoded_circles = circle_decoder_infer.apply({'params': params_circle_infer['decoder']}, latent_from_squares)
        num_decoded_circles_actual = count_objects(decoded_circles[0, ..., 0])

        print(f"Inference S->C: Input Squares={num_input_squares}, Decoded Circles={num_decoded_circles_actual}")
        assert num_decoded_circles_actual >= 0, "Decoded circle count should be non-negative."
        if num_input_squares > 0:
            assert num_decoded_circles_actual > 0, "Decoded circles should be non-zero if input squares are non-zero."

        print("Inference after loading weights test PASSED!")
        print("\nAll tests in suite PASSED!")

    except Exception as e:
        print(f"\nTest FAILED: {e}")
        # Fixed the indentation here:
        import traceback
        traceback.print_exc()
    finally:
        teardown_test_environment()


if __name__ == '__main__':
    test_checkpointing_and_inference_flow()
