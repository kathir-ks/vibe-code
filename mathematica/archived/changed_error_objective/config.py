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
MAX_ATTEMPTS_PER_SHAPE = 500

# Model training parameters
LEARNING_RATE = 1e-4
NUM_TRAINING_STEPS = 1000000
LOG_INTERVAL = 50
# This weight encourages the latent space to be clean (black & white)
LATENT_BINARIZATION_LOSS_WEIGHT = 1.0

# Checkpointing
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 10000
RESUME_FROM_CHECKPOINT = False
LAST_CHECKPOINT_FILENAME = "latest_checkpoint.msgpack"
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, LAST_CHECKPOINT_FILENAME)

# Data types
DTYPE = jnp.float32
