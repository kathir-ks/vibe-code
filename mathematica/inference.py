import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from functools import partial
from typing import Sequence, Tuple, Dict, Any, Optional
from flax.training import train_state

# For checkpointing
import orbax.checkpoint as ocp
from etils import epath
import os

# For visualization
import matplotlib.pyplot as plt

# --- 1. Data Generation (Copied from training script for self-containment) ---
def generate_circle_image(key: jax.random.PRNGKey,
                          image_size: int = 1080,
                          min_circles: int = 0,
                          max_circles: int = 30,
                          min_radius: int = 20,
                          max_radius: int = 100,
                          max_placement_attempts: int = 50) -> jnp.ndarray:
    """
    Generates a single binary image with non-overlapping circles.
    Output image values are 0 (black background) or 1 (white circle).
    """
    key_num_circles, key_circles_props = jax.random.split(key, 2)
    
    num_circles = jax.random.randint(key_num_circles, shape=(), minval=min_circles, maxval=max_circles + 1)
    
    canvas_np = np.zeros((image_size, image_size), dtype=np.float32)
    placed_circles_params = [] # Store (x, y, r) of placed circles

    circle_keys = jax.random.split(key_circles_props, num_circles * max_placement_attempts + 1)
    key_idx = 0

    for i in range(num_circles):
        placed_successfully = False
        for attempt in range(max_placement_attempts):
            if key_idx >= len(circle_keys):
                break
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
                placed_successfully = True
                break

    return jnp.array(canvas_np).reshape((image_size, image_size, 1))

# --- 2. Model Architecture (Encoder and Decoder - Copied for self-containment) ---
class Encoder(nn.Module):
    latent_dim_spatial: int = 32
    latent_channels: int = 16 # Number of channels in the 32x32 latent image
    features: Sequence[int] = (64, 128, 256, 512, 512) # Features for conv layers

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
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
        x = nn.tanh(x)
        return x

class Decoder(nn.Module):
    original_image_size: int = 1080
    encoder_features: Sequence[int] = (64, 128, 256, 512, 512) # Should match encoder's features in reverse

    @nn.compact
    def __call__(self, z: jnp.ndarray, training: bool) -> jnp.ndarray:
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

# Define a TrainState for each component (Encoder/Decoder of Model A/B)
class ComponentTrainState(train_state.TrainState):
    batch_stats: Any # For BatchNorm

def create_train_state(key: jax.random.PRNGKey, model: nn.Module, dummy_input: jnp.ndarray, learning_rate: float):
    """Creates an initial ComponentTrainState."""
    variables = model.init(key, dummy_input, training=False) # Initialize with training=False for BN stats
    params = variables['params']
    batch_stats = variables.get('batch_stats', {}) # Handle models without BN
    
    tx = optax.adamw(learning_rate=learning_rate)
    return ComponentTrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)

# --- 3. Loading Checkpoint for Inference ---
def load_model_for_inference(
    base_ckpt_dir: str = './checkpoints', # Base directory containing model subdirectories
    model_subdir: str = 'circle_autoencoder', # Subdirectory for the specific model
    image_size: int = 1080,
    latent_channels: int = 16,
    learning_rate: float = 1e-4, # Learning rate is needed to initialize TrainState, but not for inference
    target_step: Optional[int] = None # New: Optional specific step to load
) -> Tuple[Encoder, Decoder, Dict[str, Any]]:
    """
    Loads the latest model checkpoint for inference.
    Returns the initialized Encoder, Decoder, and their loaded parameters/batch_stats.
    """
    key = jax.random.PRNGKey(42) # Use a fixed key for initialization consistency
    key_encA, key_decA, key_encB, key_decB = jax.random.split(key, 4)

    # Initialize models (structure only)
    encoder = Encoder(latent_channels=latent_channels)
    decoder = Decoder(original_image_size=image_size)
    
    dummy_image_input = jnp.ones((1, image_size, image_size, 1), dtype=jnp.float32)
    dummy_latent_input = jnp.ones((1, encoder.latent_dim_spatial, encoder.latent_dim_spatial, latent_channels), dtype=jnp.float32)

    # Create dummy states to define the structure for Orbax restore
    state_enc_A_dummy = create_train_state(key_encA, encoder, dummy_image_input, learning_rate)
    state_dec_A_dummy = create_train_state(key_decA, decoder, dummy_latent_input, learning_rate)
    state_enc_B_dummy = create_train_state(key_encB, encoder, dummy_image_input, learning_rate)
    state_dec_B_dummy = create_train_state(key_decB, decoder, dummy_latent_input, learning_rate)

    # These dummy states define the structure Orbax expects to restore into
    # Moved this block AFTER dummy states are defined
    restoration_structure = {
        'state_enc_A': state_enc_A_dummy,
        'state_dec_A': state_dec_A_dummy,
        'state_enc_B': state_enc_B_dummy,
        'state_dec_B': state_dec_B_dummy,
        'epoch': 0,
        'step': 0
    }

    # Construct the absolute path to the specific model's checkpoint directory
    full_model_ckpt_path = epath.Path(os.path.abspath(os.path.join(base_ckpt_dir, model_subdir)))
    mngr = ocp.CheckpointManager(full_model_ckpt_path)

    # Determine which step to load
    step_to_load = target_step if target_step is not None else mngr.latest_step()

    if step_to_load is None:
        raise FileNotFoundError(
            f"No checkpoint found in {full_model_ckpt_path}. "
            "Please ensure training has been run and completed successfully, "
            "or specify a valid `target_step` if `latest_step()` is not working."
        )

    print(f"Loading checkpoint from step {step_to_load}...")
    # Wrap the restoration structure in ocp.args.Composite, unpacking the dictionary
    restored_items = mngr.restore(step_to_load, args=ocp.args.Composite(**restoration_structure))
    mngr.close() # Close the manager after loading

    # Extract the parameters and batch stats for Model A's Encoder and Model B's Decoder
    # (or vice-versa, depending on which communication path you want to test)
    # Let's use Model A's Encoder and Model B's Decoder for demonstration
    enc_A_params = restored_items['state_enc_A'].params
    enc_A_batch_stats = restored_items['state_enc_A'].batch_stats

    dec_B_params = restored_items['state_dec_B'].params
    dec_B_batch_stats = restored_items['state_dec_B'].batch_stats

    print("Model loaded successfully.")
    return encoder, decoder, {
        'enc_A_params': enc_A_params,
        'enc_A_batch_stats': enc_A_batch_stats,
        'dec_B_params': dec_B_params,
        'dec_B_batch_stats': dec_B_batch_stats,
    }

# --- 4. Inference and Visualization ---
def run_inference_and_visualize(
    image_size: int = 1080,
    latent_channels: int = 16,
    base_ckpt_dir: str = './checkpoints', # Base directory for checkpoints
    model_subdir: str = 'circle_autoencoder', # Subdirectory for the specific model
    target_step: Optional[int] = 13390 # New: Specify a known good step to load
):
    # Load the models and their states
    encoder_model, decoder_model, loaded_states = load_model_for_inference(
        base_ckpt_dir=base_ckpt_dir, # Pass base directory
        model_subdir=model_subdir,   # Pass model subdirectory
        image_size=image_size,
        latent_channels=latent_channels,
        target_step=target_step      # Pass the target step
    )

    enc_A_params = loaded_states['enc_A_params']
    enc_A_batch_stats = loaded_states['enc_A_batch_stats']
    dec_B_params = loaded_states['dec_B_params']
    dec_B_batch_stats = loaded_states['dec_B_batch_stats']

    # Generate a new random image for inference
    inference_key = jax.random.PRNGKey(12345) # Use a different key for inference data
    input_image = generate_circle_image(inference_key, image_size=image_size)
    
    # Add batch dimension for inference (model expects (B, H, W, C))
    input_image_batch = jnp.expand_dims(input_image, axis=0)

    # Perform inference (forward pass)
    # Note: For inference, `training=False` is used for BatchNorm layers
    # and `mutable=[]` as we don't update batch_stats during inference.
    
    # Encode the input image using Model A's encoder
    variables_enc_A = {'params': enc_A_params, 'batch_stats': enc_A_batch_stats}
    latent_representation = encoder_model.apply(
        variables_enc_A, input_image_batch, training=False, mutable=[]
    )
    # The latent_representation will be a JAX array of shape (1, 32, 32, latent_channels)

    # Decode the latent representation using Model B's decoder
    variables_dec_B = {'params': dec_B_params, 'batch_stats': dec_B_batch_stats}
    reconstructed_image_logits = decoder_model.apply(
        variables_dec_B, latent_representation, training=False, mutable=[]
    )
    # Apply sigmoid to get probabilities for the binary image
    reconstructed_image = jax.nn.sigmoid(reconstructed_image_logits)

    # Convert JAX arrays to NumPy for visualization
    input_image_np = np.array(input_image)
    latent_representation_np = np.array(latent_representation).squeeze(axis=0) # Remove batch dim
    reconstructed_image_np = np.array(reconstructed_image).squeeze(axis=0) # Remove batch dim

    # --- Visualization ---
    plt.figure(figsize=(18, 6))

    # Plot Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(input_image_np.squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.title('Original Input Image')
    plt.axis('off')

    # Plot Latent Image
    plt.subplot(1, 3, 2)
    # For visualization, we can take the mean across latent channels or pick one channel
    # Taking the mean can give a general idea of activation
    latent_visual = np.mean(latent_representation_np, axis=-1) # Mean across channels
    # Or, to visualize a specific channel:
    # latent_visual = latent_representation_np[:, :, 0] # First channel
    
    plt.imshow(latent_visual, cmap='viridis') # Use a colormap suitable for continuous values
    plt.title(f'Latent Representation ({latent_visual.shape[0]}x{latent_visual.shape[1]})')
    plt.colorbar(label='Activation Value')
    plt.axis('off')

    # Plot Reconstructed Image
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_image_np.squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("\nVisualization complete. Displaying input, latent, and reconstructed images.")

if __name__ == '__main__':
    # Configuration for inference
    # Ensure these match the values used during training if you want to load correctly
    INFERENCE_IMAGE_SIZE = 1080
    INFERENCE_LATENT_CHANNELS = 16
    
    # IMPORTANT: Set BASE_CHECKPOINT_DIRECTORY to the parent directory
    # that contains your specific model's checkpoint subdirectory (e.g., 'circle_autoencoder').
    BASE_CHECKPOINT_DIRECTORY = './checkpoints' 
    MODEL_SUBDIRECTORY = 'circle_autoencoder'

    # Try loading a specific step. If this works, the issue is with mngr.latest_step().
    # If it still fails, the checkpoint files themselves might be corrupted or incomplete.
    TARGET_CHECKPOINT_STEP = 13390 # Or another step like 13370, 13380

    run_inference_and_visualize(
        image_size=INFERENCE_IMAGE_SIZE,
        latent_channels=INFERENCE_LATENT_CHANNELS,
        base_ckpt_dir=BASE_CHECKPOINT_DIRECTORY,
        model_subdir=MODEL_SUBDIRECTORY,
        target_step=TARGET_CHECKPOINT_STEP
    )
