import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from functools import partial
from typing import Sequence, Tuple, Dict, Any
from flax.training import train_state

# For checkpointing
import orbax.checkpoint as ocp
from etils import epath
import os

# For TPU training, jax.device_count() will give the number of available devices.
# On a v4-8, this should be 8.
try:
    NUM_DEVICES = jax.device_count()
except RuntimeError: # Happens if JAX is not yet initialized or no TPU/GPU
    print("JAX runtime not fully initialized or no accelerator found. Defaulting to 1 device.")
    NUM_DEVICES = 1
print(f"Number of JAX devices detected: {NUM_DEVICES}")

# --- 1. Data Generation ---
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
    
    # Using NumPy for imperative drawing logic, then convert to JAX array
    # JAX is purely functional, making this kind of iterative placement complex
    # For batch generation, this function would be vmapped or called in a loop
    # and results converted to JAX arrays.
    # For actual large-scale training, consider a more JAX-native or pre-generated dataset.
    
    # This part will run on CPU due to NumPy.
    # For efficient batching on device, you'd ideally want a JAX-native generation
    # or pre-generate and load data.
    canvas_np = np.zeros((image_size, image_size), dtype=np.float32)
    placed_circles_params = [] # Store (x, y, r) of placed circles

    # Split keys for each circle attempt
    # We split more keys than strictly necessary to avoid running out during attempts
    circle_keys = jax.random.split(key_circles_props, num_circles * max_placement_attempts + 1)
    key_idx = 0

    for i in range(num_circles):
        placed_successfully = False
        for attempt in range(max_placement_attempts):
            if key_idx >= len(circle_keys): # Safety check
                break
            current_key = circle_keys[key_idx]
            key_idx += 1
            
            key_radius, key_x, key_y = jax.random.split(current_key, 3)
            
            radius = jax.random.randint(key_radius, shape=(), minval=min_radius, maxval=max_radius + 1)
            # Ensure circle center allows full circle within bounds
            center_x = jax.random.randint(key_x, shape=(), minval=radius, maxval=image_size - radius)
            center_y = jax.random.randint(key_y, shape=(), minval=radius, maxval=image_size - radius)
            
            # Check for overlap with previously placed circles
            overlap = False
            if placed_circles_params: # Only check if there are existing circles
                # Convert to JAX arrays for distance calculation if preferred, but NumPy is fine here
                
                for pc_x, pc_y, pc_r in placed_circles_params:
                    dist_sq = (float(center_x) - pc_x)**2 + (float(center_y) - pc_y)**2
                    min_dist_sq = (float(radius) + pc_r)**2
                    if dist_sq < min_dist_sq:
                        overlap = True
                        break
            
            if not overlap:
                # Draw circle (using NumPy for indexing)
                yy, xx = np.mgrid[:image_size, :image_size]
                circle = (xx - float(center_x))**2 + (yy - float(center_y))**2 <= float(radius)**2
                canvas_np[circle] = 1.0
                placed_circles_params.append((float(center_x), float(center_y), float(radius)))
                placed_successfully = True
                break
        # If not placed successfully after attempts, just continue (fewer circles)

    return jnp.array(canvas_np).reshape((image_size, image_size, 1))


def generate_batch(key: jax.random.PRNGKey, batch_size: int, image_size: int = 1080) -> jnp.ndarray:
    """Generates a batch of circle images."""
    keys = jax.random.split(key, batch_size)
    # Using a Python list comprehension with NumPy-based generation for simplicity.
    # For performance, explore jax.lax.map or pre-generation.
    images_list = [generate_circle_image(k, image_size=image_size) for k in keys]
    batch = jnp.stack(images_list)
    return batch

# --- 2. Model Architecture (Encoder and Decoder) ---
class Encoder(nn.Module):
    latent_dim_spatial: int = 32
    latent_channels: int = 16 # Number of channels in the 32x32 latent image
    features: Sequence[int] = (64, 128, 256, 512, 512) # Features for conv layers

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        # Input x: (batch, 1080, 1080, 1)
        
        # Layer 1: 1080 -> 540
        x = nn.Conv(features=self.features[0], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='enc_conv_1')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_1')(x)
        x = nn.leaky_relu(x)

        # Layer 2: 540 -> 270
        x = nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='enc_conv_2')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_2')(x)
        x = nn.leaky_relu(x)

        # Layer 3: 270 -> 135
        x = nn.Conv(features=self.features[2], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='enc_conv_3')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_3')(x)
        x = nn.leaky_relu(x)
        # Current shape: (batch, 135, 135, features[2])

        # Layer 4: 135 -> 68 (ceil(135/2))
        x = nn.Conv(features=self.features[3], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='enc_conv_4')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_4')(x)
        x = nn.leaky_relu(x)
        # Current shape: (batch, 68, 68, features[3])

        # Layer 5: 68 -> 34
        x = nn.Conv(features=self.features[4], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='enc_conv_5')(x)
        x = nn.BatchNorm(use_running_average=not training, name='enc_bn_5')(x)
        x = nn.leaky_relu(x)
        # Current shape: (batch, 34, 34, features[4])

        # Final Conv to get to 32x32 spatial dimensions and desired latent channels
        # Input 34x34, Kernel 3x3, Stride 1, Padding VALID: (34-3)/1 + 1 = 32
        x = nn.Conv(features=self.latent_channels, kernel_size=(3, 3), strides=(1, 1), padding='VALID', name='enc_conv_final')(x)
        # Current shape: (batch, 32, 32, latent_channels)
        x = nn.tanh(x) # Output latent representation in [-1, 1] range
        return x

class Decoder(nn.Module):
    original_image_size: int = 1080
    encoder_features: Sequence[int] = (64, 128, 256, 512, 512) # Should match encoder's features in reverse

    @nn.compact
    def __call__(self, z: jnp.ndarray, training: bool) -> jnp.ndarray:
        # Input z: (batch, 32, 32, latent_channels) from Encoder
        
        # Layer 1: 32x32 -> 34x34 (matching enc_conv_final input)
        # Kernel 3x3, Stride 1, Padding VALID for ConvTranspose: (32-1)*1 + 3 = 34
        x = nn.ConvTranspose(features=self.encoder_features[4], kernel_size=(3, 3), strides=(1, 1), padding='VALID', name='dec_conv_t_1')(z)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_1')(x)
        x = nn.relu(x)
        # Current shape: (batch, 34, 34, encoder_features[4])

        # Layer 2: 34x34 -> 68x68 (matching enc_conv_5 input)
        x = nn.ConvTranspose(features=self.encoder_features[3], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='dec_conv_t_2')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_2')(x)
        x = nn.relu(x)
        # Current shape: (batch, 68, 68, encoder_features[3])

        # Layer 3: 68x68 -> 136x136 (Encoder had 135x135 here)
        x = nn.ConvTranspose(features=self.encoder_features[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='dec_conv_t_3')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_3')(x)
        x = nn.relu(x)
        # Current shape: (batch, 136, 136, encoder_features[2])
        # Crop to 135x135 to match encoder's L3 output shape if necessary
        # Flax ConvTranspose with 'SAME' padding and stride 2 doubles the input dimension.
        # Encoder: 135 -> Conv(s=2, p='SAME') -> 68. Decoder: 68 -> ConvT(s=2, p='SAME') -> 136.
        # We need to get to 135.
        if x.shape[1] == 136: # If it's 136, crop to 135
            x = x[:, :135, :135, :]


        # Layer 4: 135x135 -> 270x270 (matching enc_conv_3 input)
        x = nn.ConvTranspose(features=self.encoder_features[1], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='dec_conv_t_4')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_4')(x)
        x = nn.relu(x)
        # Current shape: (batch, 270, 270, encoder_features[1])

        # Layer 5: 270x270 -> 540x540 (matching enc_conv_2 input)
        x = nn.ConvTranspose(features=self.encoder_features[0], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='dec_conv_t_5')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_5')(x)
        x = nn.relu(x)
        # Current shape: (batch, 540, 540, encoder_features[0])

        # Final Layer: 540x540 -> 1080x1080
        # Output 1 channel for the binary image (logits)
        x = nn.ConvTranspose(features=1, kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='dec_conv_t_final')(x)
        # No activation here, outputting logits for sigmoid_binary_cross_entropy
        # Current shape: (batch, 1080, 1080, 1)
        
        # Ensure final output is exactly original_image_size x original_image_size
        if x.shape[1] != self.original_image_size or x.shape[2] != self.original_image_size:
            x = jax.image.resize(x, 
                                 (x.shape[0], self.original_image_size, self.original_image_size, x.shape[3]), 
                                 method='bilinear') # or 'nearest'
        return x

# --- 3. Training State and Step ---

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

# Combined loss function for the two-way communication
def communication_loss_fn(
    params_enc_A, batch_stats_enc_A,
    params_dec_A, batch_stats_dec_A,
    params_enc_B, batch_stats_enc_B,
    params_dec_B, batch_stats_dec_B,
    encoder_model: Encoder, decoder_model: Decoder,
    images: jnp.ndarray, dropout_key: jax.random.PRNGKey
):
    # --- Model A speaks, Model B listens ---
    # Encoder A processes image
    variables_enc_A = {'params': params_enc_A, 'batch_stats': batch_stats_enc_A}
    latent_Z_from_A, new_batch_stats_enc_A = encoder_model.apply(
        variables_enc_A, images, training=True, mutable=['batch_stats']
    )
    
    # Decoder B reconstructs from A's latent representation
    variables_dec_B = {'params': params_dec_B, 'batch_stats': batch_stats_dec_B}
    reconstructed_by_B, new_batch_stats_dec_B = decoder_model.apply(
        variables_dec_B, latent_Z_from_A, training=True, mutable=['batch_stats']
    )
    loss1 = optax.sigmoid_binary_cross_entropy(reconstructed_by_B, images).mean()

    # --- Model B speaks, Model A listens ---
    # Encoder B processes image
    variables_enc_B = {'params': params_enc_B, 'batch_stats': batch_stats_enc_B}
    latent_Z_from_B, new_batch_stats_enc_B = encoder_model.apply(
        variables_enc_B, images, training=True, mutable=['batch_stats']
    )

    # Decoder A reconstructs from B's latent representation
    variables_dec_A = {'params': params_dec_A, 'batch_stats': batch_stats_dec_A}
    reconstructed_by_A, new_batch_stats_dec_A = decoder_model.apply(
        variables_dec_A, latent_Z_from_B, training=True, mutable=['batch_stats']
    )
    loss2 = optax.sigmoid_binary_cross_entropy(reconstructed_by_A, images).mean()
    
    total_loss = loss1 + loss2
    
    aux_data = {
        "loss1": loss1, "loss2": loss2,
        "new_batch_stats_enc_A": new_batch_stats_enc_A['batch_stats'],
        "new_batch_stats_dec_A": new_batch_stats_dec_A['batch_stats'],
        "new_batch_stats_enc_B": new_batch_stats_enc_B['batch_stats'],
        "new_batch_stats_dec_B": new_batch_stats_dec_B['batch_stats'],
    }
    return total_loss, aux_data

# Training step function
@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(4,5))
def train_step(
    state_enc_A: ComponentTrainState, state_dec_A: ComponentTrainState,
    state_enc_B: ComponentTrainState, state_dec_B: ComponentTrainState,
    encoder_model: Encoder, decoder_model: Decoder, # Static
    batch_images: jnp.ndarray, dropout_key: jax.random.PRNGKey
):
    # Gradient function
    grad_fn = jax.value_and_grad(
        communication_loss_fn,
        argnums=(0, 2, 4, 6), # Grad for params of enc_A, dec_A, enc_B, dec_B
        has_aux=True
    )
    
    # Compute gradients and loss
    (total_loss, aux_data), grads = grad_fn(
        state_enc_A.params, state_enc_A.batch_stats,
        state_dec_A.params, state_dec_A.batch_stats,
        state_enc_B.params, state_enc_B.batch_stats,
        state_dec_B.params, state_dec_B.batch_stats,
        encoder_model, decoder_model,
        batch_images, dropout_key
    )

    # Average gradients and losses across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    total_loss = jax.lax.pmean(total_loss, axis_name='batch')
    loss1 = jax.lax.pmean(aux_data['loss1'], axis_name='batch')
    loss2 = jax.lax.pmean(aux_data['loss2'], axis_name='batch')

    # Update states
    new_state_enc_A = state_enc_A.apply_gradients(
        grads=grads[0], batch_stats=aux_data['new_batch_stats_enc_A']
    )
    new_state_dec_A = state_dec_A.apply_gradients(
        grads=grads[1], batch_stats=aux_data['new_batch_stats_dec_A']
    )
    new_state_enc_B = state_enc_B.apply_gradients(
        grads=grads[2], batch_stats=aux_data['new_batch_stats_enc_B']
    )
    new_state_dec_B = state_dec_B.apply_gradients(
        grads=grads[3], batch_stats=aux_data['new_batch_stats_dec_B']
    )
    
    metrics = {'total_loss': total_loss, 'loss_A_speaks': loss1, 'loss_B_speaks': loss2}
    return new_state_enc_A, new_state_dec_A, new_state_enc_B, new_state_dec_B, metrics

# --- 4. Main Training Loop ---
def main_training_loop(
    num_epochs: int = 100,
    batch_size_per_device: int = 4, # Adjust based on TPU memory
    learning_rate: float = 1e-4,
    image_size: int = 1080,
    latent_channels: int = 16,
    ckpt_dir: str = './checkpoints/circle_autoencoder',
    start_epoch: int = 0,
    start_step: int = 0
):
    global_batch_size = batch_size_per_device * NUM_DEVICES
    
    key = jax.random.PRNGKey(0)
    key_init, key_data, key_dropout_master = jax.random.split(key, 3)

    # Initialize models
    encoder = Encoder(latent_channels=latent_channels)
    decoder = Decoder(original_image_size=image_size)
    
    dummy_image_input = jnp.ones((1, image_size, image_size, 1), dtype=jnp.float32)
    dummy_latent_input = jnp.ones((1, encoder.latent_dim_spatial, encoder.latent_dim_spatial, latent_channels), dtype=jnp.float32)

    key_encA, key_decA, key_encB, key_decB = jax.random.split(key_init, 4)

    # Initialize states (these will be overwritten if loading from checkpoint)
    state_enc_A = create_train_state(key_encA, encoder, dummy_image_input, learning_rate)
    state_dec_A = create_train_state(key_decA, decoder, dummy_latent_input, learning_rate)
    state_enc_B = create_train_state(key_encB, encoder, dummy_image_input, learning_rate) # Same architecture
    state_dec_B = create_train_state(key_decB, decoder, dummy_latent_input, learning_rate) # Same architecture

    # Setup CheckpointManager
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=10,  # Save every 10 steps
        max_to_keep=3,           # Keep only the 3 most recent checkpoints
        step_prefix='checkpoint',
    )
    mngr = ocp.CheckpointManager(epath.Path(ckpt_dir), options=options)

    # Try to restore the latest checkpoint
    latest_step = mngr.latest_step()
    if latest_step is not None:
        print(f"Restoring checkpoint from step {latest_step}...")
        restored_items = mngr.restore(latest_step, args=ocp.args.StandardRestore(
            {
                'state_enc_A': state_enc_A, # Provide target structure
                'state_dec_A': state_dec_A,
                'state_enc_B': state_enc_B,
                'state_dec_B': state_dec_B,
                'epoch': 0, # Placeholder, will be overwritten
                'step': 0   # Placeholder
            }
        ))
        state_enc_A = restored_items['state_enc_A']
        state_dec_A = restored_items['state_dec_A']
        state_enc_B = restored_items['state_enc_B']
        state_dec_B = restored_items['state_dec_B']
        start_epoch = restored_items['epoch'] + 1 # Resume from next epoch
        start_step = restored_items['step'] + 1 # Resume from next step
        print(f"Successfully restored. Resuming from Epoch {start_epoch}, Step {start_step}.")
    else:
        print("No checkpoint found. Starting training from scratch.")
        start_epoch = 0
        start_step = 0

    # Replicate states across devices for pmap (after potential restoration)
    state_enc_A = jax.device_put_replicated(state_enc_A, jax.devices())
    state_dec_A = jax.device_put_replicated(state_dec_A, jax.devices())
    state_enc_B = jax.device_put_replicated(state_enc_B, jax.devices())
    state_dec_B = jax.device_put_replicated(state_dec_B, jax.devices())

    print(f"Starting training for {num_epochs} epochs.")
    print(f"Global batch size: {global_batch_size} ({batch_size_per_device} per device on {NUM_DEVICES} devices)")

    # Adjust key_data based on start_step to ensure reproducibility if resuming
    # This is a simplification; for full reproducibility, you'd need to store the PRNGKey state
    # and restore it, or re-derive it based on the restored step.
    key_data = jax.random.fold_in(key_data, start_step) 
    key_dropout_master = jax.random.fold_in(key_dropout_master, start_step)

    num_steps_per_epoch = 50 # Example: 50 steps per epoch

    for epoch in range(start_epoch, num_epochs):
        key_data, key_epoch_data = jax.random.split(key_data)
        
        epoch_total_loss = 0.0
        epoch_loss1_avg = 0.0
        epoch_loss2_avg = 0.0

        current_step_in_epoch = 0
        if epoch == start_epoch:
            current_step_in_epoch = start_step % num_steps_per_epoch # Handle resuming mid-epoch

        for step in range(current_step_in_epoch, num_steps_per_epoch):
            key_step_data, key_dropout_step = jax.random.split(key_epoch_data, 2)
            
            # Generate a batch of images
            # Note: generate_batch uses NumPy internally, which is not ideal for pure JAX pipeline.
            # This will be slow.
            batch_images_host = generate_batch(key_step_data, global_batch_size, image_size=image_size)
            
            # Shard data to devices
            # Reshape to (NUM_DEVICES, batch_size_per_device, H, W, C)
            sharded_batch_images = batch_images_host.reshape(
                (NUM_DEVICES, batch_size_per_device) + batch_images_host.shape[1:]
            )
            
            # Distribute dropout keys
            step_dropout_keys = jax.random.split(key_dropout_step, NUM_DEVICES)

            state_enc_A, state_dec_A, state_enc_B, state_dec_B, metrics = train_step(
                state_enc_A, state_dec_A, state_enc_B, state_dec_B,
                encoder, decoder, # Pass by reference (static)
                sharded_batch_images, step_dropout_keys
            )
            
            # Metrics are already averaged by pmap
            epoch_total_loss += metrics['total_loss'][0] # metrics are replicated, take first element
            epoch_loss1_avg += metrics['loss_A_speaks'][0]
            epoch_loss2_avg += metrics['loss_B_speaks'][0]

            global_step = epoch * num_steps_per_epoch + step
            if global_step % 10 == 0: # Log every 10 steps
                print(f"  Epoch {epoch+1}/{num_epochs}, Step {step+1}/{num_steps_per_epoch}, "
                      f"Global Step: {global_step}, "
                      f"Step Total Loss: {metrics['total_loss'][0]:.4f}, "
                      f"Loss A->B: {metrics['loss_A_speaks'][0]:.4f}, "
                      f"Loss B->A: {metrics['loss_B_speaks'][0]:.4f}")

            # Save checkpoint
            if mngr.should_save(global_step):
                # Unreplicate states before saving
                unreplicated_state_enc_A = jax.device_get(state_enc_A)
                unreplicated_state_dec_A = jax.device_get(state_dec_A)
                unreplicated_state_enc_B = jax.device_get(state_enc_B)
                unreplicated_state_dec_B = jax.device_get(state_dec_B)

                items_to_save = {
                    'state_enc_A': unreplicated_state_enc_A,
                    'state_dec_A': unreplicated_state_dec_A,
                    'state_enc_B': unreplicated_state_enc_B,
                    'state_dec_B': unreplicated_state_dec_B,
                    'epoch': epoch,
                    'step': step
                }
                mngr.save(global_step, args=ocp.args.StandardSave(items_to_save))
                print(f"Saved checkpoint at global step {global_step}.")


        avg_epoch_loss = epoch_total_loss / num_steps_per_epoch
        avg_loss1 = epoch_loss1_avg / num_steps_per_epoch
        avg_loss2 = epoch_loss2_avg / num_steps_per_epoch
        print(f"Epoch {epoch+1} completed. Avg Total Loss: {avg_epoch_loss:.4f}, "
              f"Avg Loss A->B: {avg_loss1:.4f}, Avg Loss B->A: {avg_loss2:.4f}")

    # Ensure all saves are finished
    mngr.wait_until_finished()
    mngr.close()
    print("Training finished. Checkpoint manager closed.")

if __name__ == '__main__':
    # Configuration
    # WARNING: 1080x1080 images with batch_size_per_device=4 will require significant memory.
    # For initial testing, you might want to reduce image_size (e.g., to 128 or 256)
    # or batch_size_per_device. The current data generation is also slow.
    
    # Example: Small scale test
    # main_training_loop(num_epochs=5, batch_size_per_device=1, image_size=128, latent_channels=8)

    # Full scale as per request (might be slow and memory intensive to start)
    # Consider profiling and optimizing data generation and model size for 1080p.
    main_training_loop(
        num_epochs=10, # Small number of epochs for demonstration
        batch_size_per_device=1, # Small batch size due to large image and model
        learning_rate=1e-4,
        image_size=1080, # As requested
        latent_channels=16 # Channels in the 32x32 latent image
    )
