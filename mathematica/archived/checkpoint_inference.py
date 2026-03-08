import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax # Import optax as it's used in the training script
from functools import partial
from typing import Sequence, Tuple, Dict, Any, Optional
from flax.training import train_state
import orbax.checkpoint as ocp
from etils import epath
import os
from PIL import Image

# --- 1. Model and State Definitions (Copied from training script for consistency) ---
# It's crucial that these definitions match the ones used during training exactly.

class Encoder(nn.Module):
    latent_dim_spatial: int = 32
    latent_channels: int = 4 # IMPORTANT: Match this with the trained model's latent_channels
    features: Sequence[int] = (64, 128, 256, 512, 512)

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        # Input x: (batch, 1080, 1080, 1)
        x = nn.Conv(features=self.features[0], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='enc_conv_1')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_1')(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='enc_conv_2')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_2')(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=self.features[2], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='enc_conv_3')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_3')(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=self.features[3], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='enc_conv_4')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_4')(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=self.features[4], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='enc_conv_5')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_5')(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=self.latent_channels, kernel_size=(3, 3), strides=(1, 1), padding='VALID', name='enc_conv_final')(x)
        return x

class Decoder(nn.Module):
    original_image_size: int = 1080
    encoder_features: Sequence[int] = (64, 128, 256, 512, 512)

    @nn.compact
    def __call__(self, z: jnp.ndarray, training: bool) -> jnp.ndarray:
        # Input z: (batch, 32, 32, latent_channels)
        x = nn.ConvTranspose(features=self.encoder_features[4], kernel_size=(3, 3), strides=(1, 1), padding='VALID', name='dec_conv_t_1')(z)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_1')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.encoder_features[3], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='dec_conv_t_2')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_2')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.encoder_features[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='dec_conv_t_3')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_3')(x)
        x = nn.relu(x)
        if x.shape[1] == 136:
            x = x[:, :135, :135, :]
        x = nn.ConvTranspose(features=self.encoder_features[1], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='dec_conv_t_4')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_4')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.encoder_features[0], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='dec_conv_t_5')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_5')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='dec_conv_t_final')(x)
        if x.shape[1] != self.original_image_size or x.shape[2] != self.original_image_size:
            x = jax.image.resize(x,
                                  (x.shape[0], self.original_image_size, self.original_image_size, x.shape[3]),
                                  method='bilinear')
        return x

# This class comes directly from your training script
class ComponentTrainState(train_state.TrainState):
    batch_stats: Any

# --- 2. Data Generation (For creating a test image) ---
# Copied directly from training script for consistency
def generate_circle_image(key: jax.random.PRNGKey,
                          image_size: int = 1080,
                          min_circles: int = 5,
                          max_circles: int = 15,
                          min_radius: int = 30,
                          max_radius: int = 90,
                          max_placement_attempts: int = 50) -> jnp.ndarray:
    """Generates a single binary image with non-overlapping circles."""
    key_num_circles, key_circles_props = jax.random.split(key, 2)
    num_circles = jax.random.randint(key_num_circles, shape=(), minval=min_circles, maxval=max_circles + 1)
    canvas_np = np.zeros((image_size, image_size), dtype=np.float32)
    placed_circles_params = []
    circle_keys = jax.random.split(key_circles_props, num_circles * max_placement_attempts + 1)
    key_idx = 0
    for i in range(num_circles):
        for attempt in range(max_placement_attempts):
            if key_idx >= len(circle_keys): break
            current_key = circle_keys[key_idx]
            key_idx += 1
            key_radius, key_x, key_y = jax.random.split(current_key, 3)
            radius = jax.random.randint(key_radius, shape=(), minval=min_radius, maxval=max_radius + 1)
            center_x = jax.random.randint(key_x, shape=(), minval=radius, maxval=image_size - radius)
            center_y = jax.random.randint(key_y, shape=(), minval=radius, maxval=image_size - radius)
            overlap = False
            if placed_circles_params:
                for pc_x, pc_y, pc_r in placed_circles_params:
                    dist_sq = (float(center_x) - pc_x)**2 + (float(center_y) - pc_y)**2
                    min_dist_sq = (float(radius) + pc_r)**2
                    if dist_sq < min_dist_sq:
                        overlap = True
                        break
            if not overlap:
                yy, xx = np.mgrid[:image_size, :image_size]
                circle = (xx - float(center_x))**2 + (yy - float(center_y))**2 <= float(radius)**2
                canvas_np[circle] = 1.0
                placed_circles_params.append((float(center_x), float(center_y), float(radius)))
                break
    return jnp.array(canvas_np).reshape((image_size, image_size, 1))

# --- 3. Inference Functions ---

def binarize_latent(logits: jnp.ndarray) -> jnp.ndarray:
    """Applies sigmoid and rounds to {0,1} for the latent space."""
    probs = jax.nn.sigmoid(logits)
    binary_values = jnp.round(probs)
    return binary_values

@partial(jax.jit, static_argnames=['encoder', 'decoder'])
def run_inference(
    encoder: Encoder,
    decoder: Decoder,
    params_enc: Dict,
    batch_stats_enc: Dict,
    params_dec: Dict,
    batch_stats_dec: Dict,
    image: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Runs the full inference pipeline: image -> encoder -> latent -> decoder -> reconstructed_image.
    """
    variables_enc = {'params': params_enc, 'batch_stats': batch_stats_enc}
    latent_logits = encoder.apply(variables_enc, image, training=False)
    latent_binary = binarize_latent(latent_logits)
    variables_dec = {'params': params_dec, 'batch_stats': batch_stats_dec}
    reconstructed_logits = decoder.apply(variables_dec, latent_binary, training=False)
    reconstructed_image = jax.nn.sigmoid(reconstructed_logits)
    return reconstructed_image, latent_binary

def create_dummy_train_state_for_restore(key: jax.random.PRNGKey, model: nn.Module, dummy_input: jnp.ndarray, learning_rate: float, latent_channels: int):
    """
    Creates a dummy TrainState with the correct structure for Orbax restoration.
    This mimics the `create_train_state` in the training script.
    """
    variables = model.init(key, dummy_input, training=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    # The optimizer used in training script is `optax.adamw`
    tx = optax.adamw(learning_rate=learning_rate)
    
    # Manually create the opt_state structure by initializing the optimizer
    dummy_opt_state = tx.init(params)
    
    # The step needs to be a JAX array for consistency with how Orbax saves it.
    return ComponentTrainState(
        step=jnp.array(0),
        apply_fn=model.apply,
        params=params,
        tx=tx, # Although not used for inference, the `tx` field is part of the `TrainState` structure
        opt_state=dummy_opt_state, # This is the crucial part for matching structure
        batch_stats=batch_stats
    )


def load_latest_checkpoint(
    ckpt_dir: str,
    encoder: Encoder,
    decoder: Decoder,
    latent_channels: int,
    image_size: int
) -> Optional[Tuple[Any, Any]]:
    """
    Loads the most recent Orbax checkpoint and returns the parameters and batch stats
    for the 'A' model (encoder and decoder).
    """
    absolute_ckpt_dir = epath.Path(os.path.abspath(ckpt_dir))
    if not absolute_ckpt_dir.exists():
        print(f"Error: Checkpoint directory not found at {absolute_ckpt_dir}")
        return None

    # List all possible item names that might be in the checkpoint
    # This comes from the `items_to_save` dictionary in the training script's `main_training_loop`
    options = ocp.CheckpointManagerOptions(
        step_prefix='ckpt_step',
        item_names=('state_enc_A', 'state_dec_A', 'state_enc_B', 'state_dec_B', 'saved_epoch_idx', 'saved_step_in_epoch_idx')
    )
    mngr = ocp.CheckpointManager(absolute_ckpt_dir, options=options)

    latest_step = mngr.latest_step()

    if latest_step is None:
        print(f"Error: No checkpoints found in {absolute_ckpt_dir}")
        return None

    print(f"Found latest checkpoint at step: {latest_step}. Restoring...")

    key = jax.random.PRNGKey(0)
    key_encA_init, key_decA_init, key_encB_init, key_decB_init = jax.random.split(key, 4) # Need 4 keys for dummy states

    dummy_image_input = jnp.ones((1, image_size, image_size, 1), dtype=jnp.float32)
    dummy_latent_input = jnp.ones((1, encoder.latent_dim_spatial, encoder.latent_dim_spatial, latent_channels), dtype=jnp.float32)

    # Create dummy states with the *exact* same structure as saved during training
    # The learning rate for dummy init doesn't matter, just the structure of the optimizer state
    dummy_learning_rate = 1e-4 # Match training script's default for consistency, though value doesn't affect structure

    state_enc_A_structure = create_dummy_train_state_for_restore(key_encA_init, encoder, dummy_image_input, dummy_learning_rate, latent_channels)
    state_dec_A_structure = create_dummy_train_state_for_restore(key_decA_init, decoder, dummy_latent_input, dummy_learning_rate, latent_channels)
    state_enc_B_structure = create_dummy_train_state_for_restore(key_encB_init, encoder, dummy_image_input, dummy_learning_rate, latent_channels)
    state_dec_B_structure = create_dummy_train_state_for_restore(key_decB_init, decoder, dummy_latent_input, dummy_learning_rate, latent_channels)

    # The target_restore_dict must match the items_to_save from the training script
    target_restore_dict = {
        'state_enc_A': state_enc_A_structure,
        'state_dec_A': state_dec_A_structure,
        'state_enc_B': state_enc_B_structure, # Include this as it's saved
        'state_dec_B': state_dec_B_structure, # Include this as it's saved
        'saved_epoch_idx': jnp.array(0),         # Placeholder for metadata
        'saved_step_in_epoch_idx': jnp.array(0), # Placeholder for metadata
    }

    try:
        restored_items = mngr.restore(
            latest_step,
            args=ocp.args.StandardRestore(target_restore_dict)
        )
        restored_state_enc_A = restored_items['state_enc_A']
        restored_state_dec_A = restored_items['state_dec_A']
        # You might also want to log or use the restored B states or metadata,
        # but for A-only inference, we just need A's states.
        print("Successfully restored models 'A' and 'B' from checkpoint.")
        return restored_state_enc_A, restored_state_dec_A
    except Exception as e:
        print(f"Error during checkpoint restoration: {e}")
        return None

def save_image(array: np.ndarray, path: str):
    """Saves a numpy array as a PNG image."""
    img_array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)
    print(f"Saved image to {path}")

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    IMAGE_SIZE = 1080
    LATENT_CHANNELS = 4 # Use 4 as per your training script's main_training_loop call
    CHECKPOINT_DIR = './checkpoints_comm_circle'
    OUTPUT_DIR = './inference_output'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Initialize Models ---
    # Ensure latent_channels matches what was used during training
    encoder_model = Encoder(latent_channels=LATENT_CHANNELS)
    decoder_model = Decoder(original_image_size=IMAGE_SIZE)

    # --- Load Checkpoint ---
    loaded_data = load_latest_checkpoint(
        CHECKPOINT_DIR,
        encoder_model,
        decoder_model,
        LATENT_CHANNELS,
        IMAGE_SIZE
    )

    if loaded_data:
        state_enc, state_dec = loaded_data
        params_enc_A = state_enc.params
        batch_stats_enc_A = state_enc.batch_stats
        params_dec_A = state_dec.params
        batch_stats_dec_A = state_dec.batch_stats

        # --- Prepare Input Data ---
        print("\nGenerating a new image for inference...")
        inference_key = jax.random.PRNGKey(42)
        original_image = generate_circle_image(inference_key, image_size=IMAGE_SIZE)
        input_batch = jnp.expand_dims(original_image, axis=0)

        # --- Run Inference ---
        print("Running inference...")
        reconstructed_image, latent_code = run_inference(
            encoder=encoder_model,
            decoder=decoder_model,
            params_enc=params_enc_A,
            batch_stats_enc=batch_stats_enc_A,
            params_dec=params_dec_A,
            batch_stats_dec=batch_stats_dec_A,
            image=input_batch
        )

        # --- Process and Save Results ---
        original_image_np = np.array(original_image.squeeze())
        reconstructed_image_np = np.array(reconstructed_image.squeeze())
        latent_code_np = np.array(latent_code.squeeze())

        print(f"\nInference complete. Latent code shape: {latent_code_np.shape}")
        
        save_image(original_image_np, os.path.join(OUTPUT_DIR, 'original_image.png'))
        save_image(reconstructed_image_np, os.path.join(OUTPUT_DIR, 'reconstructed_image.png'))

    else:
        print("\nInference could not be run due to checkpoint loading failure.")