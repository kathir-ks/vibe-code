import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from functools import partial
from typing import Sequence, Tuple, Dict, Any
from flax import serialization
import os
import argparse # For command-line arguments
from etils import epath # For path consistency

# --- Copied from training script: START ---
# For TPU training, jax.device_count() will give the number of available devices.
# On a v4-8, this should be 8.
try:
    NUM_DEVICES_INFERENCE = jax.device_count() # Can be different from training
except RuntimeError: # Happens if JAX is not yet initialized or no TPU/GPU
    print("JAX runtime not fully initialized or no accelerator found. Defaulting to 1 device for inference.")
    NUM_DEVICES_INFERENCE = 1
print(f"Number of JAX devices detected for inference: {NUM_DEVICES_INFERENCE}")


# --- 1. Data Generation (copied) ---
def generate_circle_image(key: jax.random.PRNGKey,
                          image_size: int = 1080,
                          min_circles: int = 0,
                          max_circles: int = 30,
                          min_radius: int = 20,
                          max_radius: int = 100,
                          max_placement_attempts: int = 50) -> jnp.ndarray:
    key_num_circles, key_circles_props = jax.random.split(key, 2)
    num_circles = jax.random.randint(key_num_circles, shape=(), minval=min_circles, maxval=max_circles + 1)
    canvas_np = np.zeros((image_size, image_size), dtype=np.float32)
    placed_circles_params = []
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

# --- 2. Model Architecture (Encoder and Decoder) (copied) ---
class Encoder(nn.Module):
    latent_dim_spatial: int = 32
    latent_channels: int = 8
    features: Sequence[int] = (64, 128, 256, 512, 512)

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
        return x

class Decoder(nn.Module):
    original_image_size: int = 1080
    encoder_features: Sequence[int] = (64, 128, 256, 512, 512)
    # Added latent_channels to match Encoder for consistency if needed by init
    latent_channels: int = 8 # Needs to match the latent_channels of the encoder used to produce its input

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
        if x.shape[1] == 136: # Specific fix from training
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

# Binarization function (copied)
def binarize_latent_with_ste(logits: jnp.ndarray) -> jnp.ndarray:
    probs = jax.nn.sigmoid(logits)
    binary_values = jnp.round(probs)
    return probs + jax.lax.stop_gradient(binary_values - probs)
# --- Copied from training script: END ---


# --- 4. Inference Specific Code ---

def load_inference_params(param_path: epath.Path, model_to_init_for_shape: nn.Module, dummy_input_for_shape: jnp.ndarray, init_key: jax.random.PRNGKey):
    """Loads params from a .msgpack file. Also returns initial batch_stats."""
    if not param_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_path}")

    with open(param_path, "rb") as f:
        params_bytes = f.read()

    # Initialize model to get the param structure and initial batch_stats
    # The loaded params will replace model_init_vars['params']
    # The model_init_vars['batch_stats'] will be used (these are the initial, not trained running averages)
    model_init_vars = model_to_init_for_shape.init(init_key, dummy_input_for_shape, training=False)
    
    # Deserialize params into a PyTree matching the structure of model_init_vars['params']
    loaded_params = serialization.from_bytes(model_init_vars['params'], params_bytes)
    
    return loaded_params, model_init_vars.get('batch_stats', {})


def visualize_array_terminal(arr: np.ndarray, title: str, threshold: float = 0.5, max_dim: int = 40):
    """Visualizes a 2D numpy array in the terminal."""
    if arr.ndim == 3 and arr.shape[-1] == 1: # (H, W, 1)
        arr = arr.squeeze(axis=-1)
    elif arr.ndim != 2:
        print(f"Warning: Cannot visualize array with shape {arr.shape} for title '{title}'. Expected 2D or (H,W,1).")
        return

    # Downsample for terminal display
    h, w = arr.shape
    scale_h = max(1, int(np.ceil(h / max_dim)))
    scale_w = max(1, int(np.ceil(w / (max_dim * 2)))) # Characters are taller than wide

    # Simple block averaging for downsampling
    # For binary, mode might be better, but average is fine
    small_arr = np.zeros((h // scale_h, w // scale_w))
    for i in range(small_arr.shape[0]):
        for j in range(small_arr.shape[1]):
            small_arr[i,j] = np.mean(arr[i*scale_h:(i+1)*scale_h, j*scale_w:(j+1)*scale_w])

    print(f"\n--- {title} (approx. {small_arr.shape[0]}x{small_arr.shape[1]}) ---")
    for row in small_arr:
        for val in row:
            print('█' if val > threshold else ' ', end='')
        print()
    print("-" * (small_arr.shape[1] + 6))


def run_inference(
    image_size: int,
    latent_channels: int,
    checkpoint_step: int,
    base_ckpt_dir_str: str,
    seed: int = 42
):
    """Runs inference: load models, generate image, encode, decode, visualize."""
    key = jax.random.PRNGKey(seed)
    key_gen, key_enc_init, key_dec_init = jax.random.split(key, 3)

    # --- Configuration (Must match training for the loaded checkpoint) ---
    # These should ideally be stored with the checkpoint or inferred,
    # but for now, we pass them or use defaults from the model definition.
    ENCODER_LATENT_CHANNELS = latent_channels
    ENCODER_LATENT_SPATIAL_DIM = 32 # Based on architecture, 1080 / (2^5) then conv 3x3 valid = 34-2 = 32.
                                    # 1080 -> 540 -> 270 -> 135 -> 68 -> 34 -> (VALID 3x3) -> 32

    # --- Initialize Models ---
    encoder = Encoder(latent_channels=ENCODER_LATENT_CHANNELS, latent_dim_spatial=ENCODER_LATENT_SPATIAL_DIM)
    decoder = Decoder(original_image_size=image_size, latent_channels=ENCODER_LATENT_CHANNELS) # Pass latent_channels to Decoder too

    # --- Prepare Dummy Inputs for Initialization ---
    dummy_image_input = jnp.ones((1, image_size, image_size, 1), dtype=jnp.float32)
    dummy_latent_input = jnp.ones(
        (1, ENCODER_LATENT_SPATIAL_DIM, ENCODER_LATENT_SPATIAL_DIM, ENCODER_LATENT_CHANNELS),
        dtype=jnp.float32
    )

    # --- Load Parameters ---
    base_ckpt_dir = epath.Path(os.path.abspath(base_ckpt_dir_str))
    inference_weights_dir = base_ckpt_dir.parent / f"{base_ckpt_dir.name}_inference_weights"
    
    enc_A_param_path = inference_weights_dir / f"enc_A_params_step_{checkpoint_step}.msgpack"
    dec_A_param_path = inference_weights_dir / f"dec_A_params_step_{checkpoint_step}.msgpack"

    print(f"Loading Encoder A parameters from: {enc_A_param_path}")
    enc_A_params, enc_A_batch_stats = load_inference_params(enc_A_param_path, encoder, dummy_image_input, key_enc_init)
    
    print(f"Loading Decoder A parameters from: {dec_A_param_path}")
    dec_A_params, dec_A_batch_stats = load_inference_params(dec_A_param_path, decoder, dummy_latent_input, key_dec_init)

    variables_enc_A = {'params': enc_A_params, 'batch_stats': enc_A_batch_stats}
    variables_dec_A = {'params': dec_A_params, 'batch_stats': dec_A_batch_stats}

    # --- Generate a Sample Image ---
    print(f"\nGenerating a sample image ({image_size}x{image_size})...")
    # For inference, generate one image, add batch dim
    original_image_single = generate_circle_image(key_gen, image_size=image_size, min_circles=5, max_circles=15)
    original_image_batch = jnp.expand_dims(original_image_single, axis=0) # (1, H, W, C)

    # --- Perform Inference ---
    print("Encoding image with Encoder A...")
    # Note: training=False is crucial for BatchNorm to use running averages (even if initial ones)
    logits_Z_from_A = encoder.apply(variables_enc_A, original_image_batch, training=False)

    print("Binarizing latent representation...")
    latent_Z_for_dec_A = binarize_latent_with_ste(logits_Z_from_A) # (1, 32, 32, latent_channels)

    print("Decoding latent representation with Decoder A...")
    reconstructed_logits_by_A = decoder.apply(variables_dec_A, latent_Z_for_dec_A, training=False)
    reconstructed_image_probs_by_A = jax.nn.sigmoid(reconstructed_logits_by_A) # (1, H, W, 1)

    # --- Convert JAX arrays to NumPy for visualization ---
    # Squeeze the batch dimension for visualization
    original_np = np.array(original_image_batch[0])
    latent_np = np.array(latent_Z_for_dec_A[0]) # (32, 32, latent_channels)
    reconstructed_np = np.array(reconstructed_image_probs_by_A[0])

    # --- Visualize ---
    print("\n" + "="*50)
    print("           INFERENCE RESULTS (TERMINAL VIZ)")
    print("="*50)

    visualize_array_terminal(original_np, "Original Image", threshold=0.5)
    
    # Visualize the first channel of the latent space
    visualize_array_terminal(latent_np[:, :, 0], "Latent Representation (Channel 0)", threshold=0.5)
    if latent_np.shape[-1] > 1:
         visualize_array_terminal(latent_np[:, :, 1], "Latent Representation (Channel 1)", threshold=0.5)


    visualize_array_terminal(reconstructed_np, "Reconstructed Image (from A)", threshold=0.5)

    print(f"\nOriginal image shape: {original_np.shape}")
    print(f"Latent representation shape: {latent_np.shape}")
    print(f"Reconstructed image shape: {reconstructed_np.shape}")

    # You can also save these as images if matplotlib is available:
    # import matplotlib.pyplot as plt
    # plt.imsave("original_image.png", original_np.squeeze(), cmap='gray')
    # plt.imsave("latent_channel0.png", latent_np[:,:,0], cmap='gray')
    # plt.imsave("reconstructed_image.png", reconstructed_np.squeeze(), cmap='gray')
    # print("\nSaved images: original_image.png, latent_channel0.png, reconstructed_image.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference for circle generation model.")
    parser.add_argument(
        "--image_size", type=int, default=1080,
        help="Size of the images (must match training)."
    )
    parser.add_argument(
        "--latent_channels", type=int, default=4, # Or 8, depending on what you trained with
        help="Number of latent channels (must match training)."
    )
    parser.add_argument(
        "--checkpoint_step", type=int, required=True,
        help="Global step number of the inference checkpoint to load."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="./checkpoints_comm_circle",
        help="Base directory where training checkpoints (and thus inference weights subfolder) are stored."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for image generation and model initialization keys."
    )

    args = parser.parse_args()

    # Example check for Decoder's latent_channels parameter:
    # If your Encoder was trained with latent_channels=X, the Decoder also needs to know this
    # to correctly define its first ConvTranspose layer's input.
    # The current Decoder class takes `latent_channels` as an argument.
    # This should be consistent with Encoder's output.

    run_inference(
        image_size=args.image_size,
        latent_channels=args.latent_channels,
        checkpoint_step=args.checkpoint_step,
        base_ckpt_dir_str=args.ckpt_dir,
        seed=args.seed
    )