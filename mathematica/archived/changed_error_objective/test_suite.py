# test_suite.py

import jax
import jax.numpy as jnp
import numpy as np
import os
import shutil
import sys
import optax

# Add current directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import config
# MODIFIED: Removed imports for functions that no longer exist in train.py
from train import main as train_main_func
from inference import load_models_and_params
from data_generator import generate_paired_images

# --- Test Functions ---

def setup_test_environment():
    """Sets up a clean environment for testing (config, directories)."""
    # Store original config values to restore them later
    global original_config
    original_config = {
        'TEST_MODE': getattr(config, 'TEST_MODE', False),
        'NUM_TRAINING_STEPS': config.NUM_TRAINING_STEPS,
        'CHECKPOINT_INTERVAL': config.CHECKPOINT_INTERVAL,
        'LAST_CHECKPOINT_PATH': config.LAST_CHECKPOINT_PATH,
    }

    # Set config values for a quick test run
    config.TEST_MODE = True
    config.NUM_TRAINING_STEPS = 2 # Train for only 2 steps
    config.CHECKPOINT_INTERVAL = 2 # Checkpoint immediately
    config.LAST_CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "test_checkpoint.msgpack")

    # Clean up and create test checkpoints directory
    if os.path.exists(config.CHECKPOINT_DIR):
        shutil.rmtree(config.CHECKPOINT_DIR)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print(f"Test environment set up. Checkpoints will be saved to {config.CHECKPOINT_DIR}")

def teardown_test_environment():
    """Restores original configuration and cleans up test artifacts."""
    # Restore original config values
    for key, value in original_config.items():
        setattr(config, key, value)

    if os.path.exists(config.CHECKPOINT_DIR):
        shutil.rmtree(config.CHECKPOINT_DIR)
    print(f"Test environment torn down. Cleaned up {config.CHECKPOINT_DIR}")

def test_full_pipeline():
    """
    Tests the full pipeline: short training, checkpoint saving, loading, and inference.
    """
    print("\n--- Running Full Pipeline Test ---")
    
    setup_test_environment()

    try:
        # 1. Run a short training session to generate a checkpoint.
        print(f"\nRunning `train.py` for {config.NUM_TRAINING_STEPS} steps to generate checkpoint...")
        train_main_func()
        
        # Verify that the checkpoint was created
        assert os.path.exists(config.LAST_CHECKPOINT_PATH), "Training finished but no checkpoint was created."
        print("Checkpoint created successfully.")

        # 2. Test inference with the generated checkpoint.
        print("\nTesting inference using the saved weights...")
        models, params = load_models_and_params()
        (circle_encoder, circle_decoder, square_encoder, square_decoder) = models
        (params_circle, params_square) = params

        # Generate a new data pair for inference
        rng_key = jax.random.key(123)
        gt_circles_img, gt_squares_img = generate_paired_images(rng_key)
        circle_batch = jnp.expand_dims(gt_circles_img, (0, -1))

        # Perform a forward pass (circles -> squares)
        latent_c = circle_encoder.apply({'params': params_circle['encoder']}, circle_batch)
        decoded_s_logits = square_decoder.apply({'params': params_square['decoder']}, latent_c)
        decoded_s_img = jax.nn.sigmoid(decoded_s_logits)

        # Assertion: The model should produce a non-blank image.
        # A simple check is that the sum of all pixel values is greater than zero.
        output_sum = jnp.sum(decoded_s_img)
        assert output_sum > 0, f"Inference produced a blank image (sum of pixels = {output_sum})."
        print("Inference test passed: Model produced a non-blank output.")
        
        print("\nAll pipeline tests PASSED!")

    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        teardown_test_environment()


if __name__ == '__main__':
    test_full_pipeline()
