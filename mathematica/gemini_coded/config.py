# config.py

import jax.numpy as jnp
import os

# Image dimensions
IMG_SIZE = 1080
LATENT_SIZE = 32

# Shape Generation parameters
NUM_SHAPES_MIN = 10
NUM_SHAPES_MAX = 30
CIRCLE_RADIUS_MIN = 15
CIRCLE_RADIUS_MAX = 60
SQUARE_SIDE_LENGTH_MIN = 20
SQUARE_SIDE_LENGTH_MAX = 80
MAX_ATTEMPTS_PER_SHAPE = 500 # Max attempts to place a non-overlapping shape

# Model training parameters
LEARNING_RATE = 1e-4
NUM_TRAINING_STEPS = 2 # Default training steps
LOG_INTERVAL = 1
LATENT_BINARIZATION_LOSS_WEIGHT = 0.1 # Weight for the auxiliary latent binarization loss

# Checkpointing
CHECKPOINT_DIR = "./checkpoints" # Directory to save model checkpoints
CHECKPOINT_INTERVAL = 500 # Save checkpoint every N steps
RESUME_FROM_CHECKPOINT = False # Set to True to resume training
LAST_CHECKPOINT_FILENAME= "latest_checkpoint.msgpack"
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, LAST_CHECKPOINT_FILENAME)

# Test mode flag: Set to True by test scripts to adjust parameters for quick testing
TEST_MODE = False

# Data types
DTYPE = jnp.float32
