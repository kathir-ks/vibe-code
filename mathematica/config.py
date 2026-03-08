# config.py — Central configuration for the emergent number line experiment

import jax.numpy as jnp

# Image dimensions
IMG_SIZE = 1080
LATENT_SIZE = 32

# Shape generation
NUM_SHAPES_MIN = 0
NUM_SHAPES_MAX = 30
CIRCLE_RADIUS_MIN = 20
CIRCLE_RADIUS_MAX = 80
TRIANGLE_SIDE_MIN = 30
TRIANGLE_SIDE_MAX = 100
MAX_PLACEMENT_ATTEMPTS = 200

# Training
LEARNING_RATE = 1e-4
NUM_TRAINING_STEPS = 100_000
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 5000

# Loss weights
LAMBDA_SELF = 1.0          # Self-reconstruction (keeps decoders sharp)
LAMBDA_CROSS = 1.0         # Cross-area-matching (forces count encoding)
LAMBDA_BINARIZATION = 0.1  # Latent binarization (clean latent space)

# Checkpointing
CHECKPOINT_DIR = "./checkpoints"

# Data types
DTYPE = jnp.float32
