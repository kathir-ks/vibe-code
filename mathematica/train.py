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
import os # Import os for path manipulation

# For saving inference weights
from flax import serialization

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


def generate_batch(key: jax.random.PRNGKey, batch_size: int, image_size: int = 1080) -> jnp.ndarray:
    """Generates a batch of circle images."""
    keys = jax.random.split(key, batch_size)
    images_list = [generate_circle_image(k, image_size=image_size) for k in keys]
    batch = jnp.stack(images_list)
    return batch

# --- 2. Model Architecture (Encoder and Decoder) ---
class Encoder(nn.Module):
    latent_dim_spatial: int = 32
    latent_channels: int = 8 # Default changed to 8
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
        # Output logits, binarization will happen outside with STE
        return x

class Decoder(nn.Module):
    original_image_size: int = 1080
    encoder_features: Sequence[int] = (64, 128, 256, 512, 512) 

    @nn.compact
    def __call__(self, z: jnp.ndarray, training: bool) -> jnp.ndarray:
        # Input z: (batch, 32, 32, latent_channels), values are {0,1}
        
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

# --- 3. Training State and Step ---

class ComponentTrainState(train_state.TrainState):
    batch_stats: Any 

def create_train_state(key: jax.random.PRNGKey, model: nn.Module, dummy_input: jnp.ndarray, learning_rate: float):
    variables = model.init(key, dummy_input, training=False) 
    params = variables['params']
    batch_stats = variables.get('batch_stats', {}) 
    
    tx = optax.adamw(learning_rate=learning_rate)
    return ComponentTrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)

def binarize_latent_with_ste(logits: jnp.ndarray) -> jnp.ndarray:
    """Applies sigmoid, rounds to {0,1}, and uses STE for gradients."""
    probs = jax.nn.sigmoid(logits)
    binary_values = jnp.round(probs) # Forward pass uses {0,1}
    # STE: gradient passes through `probs` as if `binary_values` was `probs`.
    return probs + jax.lax.stop_gradient(binary_values - probs)

def communication_loss_fn(
    params_enc_A, batch_stats_enc_A,
    params_dec_A, batch_stats_dec_A,
    params_enc_B, batch_stats_enc_B,
    params_dec_B, batch_stats_dec_B,
    encoder_model: Encoder, decoder_model: Decoder,
    images: jnp.ndarray, dropout_key: jax.random.PRNGKey # dropout_key currently unused
):
    # --- Model A speaks, Model B listens ---
    variables_enc_A = {'params': params_enc_A, 'batch_stats': batch_stats_enc_A}
    logits_Z_from_A, new_batch_stats_enc_A = encoder_model.apply(
        variables_enc_A, images, training=True, mutable=['batch_stats']
    )
    latent_for_dec_B = binarize_latent_with_ste(logits_Z_from_A)
    
    variables_dec_B = {'params': params_dec_B, 'batch_stats': batch_stats_dec_B}
    reconstructed_by_B, new_batch_stats_dec_B = decoder_model.apply(
        variables_dec_B, latent_for_dec_B, training=True, mutable=['batch_stats']
    )
    loss1 = optax.sigmoid_binary_cross_entropy(reconstructed_by_B, images).mean()

    # --- Model B speaks, Model A listens ---
    variables_enc_B = {'params': params_enc_B, 'batch_stats': batch_stats_enc_B}
    logits_Z_from_B, new_batch_stats_enc_B = encoder_model.apply(
        variables_enc_B, images, training=True, mutable=['batch_stats']
    )
    latent_for_dec_A = binarize_latent_with_ste(logits_Z_from_B)

    variables_dec_A = {'params': params_dec_A, 'batch_stats': batch_stats_dec_A}
    reconstructed_by_A, new_batch_stats_dec_A = decoder_model.apply(
        variables_dec_A, latent_for_dec_A, training=True, mutable=['batch_stats']
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

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(4,5))
def train_step(
    state_enc_A: ComponentTrainState, state_dec_A: ComponentTrainState,
    state_enc_B: ComponentTrainState, state_dec_B: ComponentTrainState,
    encoder_model: Encoder, decoder_model: Decoder, 
    batch_images: jnp.ndarray, dropout_key: jax.random.PRNGKey
):
    grad_fn = jax.value_and_grad(
        communication_loss_fn,
        argnums=(0, 2, 4, 6), 
        has_aux=True
    )
    
    (total_loss, aux_data), grads = grad_fn(
        state_enc_A.params, state_enc_A.batch_stats,
        state_dec_A.params, state_dec_A.batch_stats,
        state_enc_B.params, state_enc_B.batch_stats,
        state_dec_B.params, state_dec_B.batch_stats,
        encoder_model, decoder_model,
        batch_images, dropout_key
    )

    grads = jax.lax.pmean(grads, axis_name='batch')
    total_loss = jax.lax.pmean(total_loss, axis_name='batch')
    loss1 = jax.lax.pmean(aux_data['loss1'], axis_name='batch')
    loss2 = jax.lax.pmean(aux_data['loss2'], axis_name='batch')

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
    batch_size_per_device: int = 4,
    learning_rate: float = 1e-4,
    image_size: int = 1080,
    latent_channels: int = 8, # Default changed to 8
    ckpt_dir: str = './checkpoints_comm_circle', 
    inference_weights_save_steps: int = 1000, # Steps interval for saving inference weights
    orbax_save_interval_steps: int = 50, # Steps interval for saving full checkpoints
    num_steps_per_epoch: int = 50 
):
    global_batch_size = batch_size_per_device * NUM_DEVICES
    
    key = jax.random.PRNGKey(0)
    key_init, key_data_master, key_dropout_master = jax.random.split(key, 3)

    encoder = Encoder(latent_channels=latent_channels)
    decoder = Decoder(original_image_size=image_size)
    
    dummy_image_input = jnp.ones((1, image_size, image_size, 1), dtype=jnp.float32)
    dummy_latent_input = jnp.ones((1, encoder.latent_dim_spatial, encoder.latent_dim_spatial, latent_channels), dtype=jnp.float32)

    key_encA, key_decA, key_encB, key_decB = jax.random.split(key_init, 4)

    state_enc_A = create_train_state(key_encA, encoder, dummy_image_input, learning_rate)
    state_dec_A = create_train_state(key_decA, decoder, dummy_latent_input, learning_rate)
    state_enc_B = create_train_state(key_encB, encoder, dummy_image_input, learning_rate)
    state_dec_B = create_train_state(key_decB, decoder, dummy_latent_input, learning_rate)

    absolute_ckpt_dir = epath.Path(os.path.abspath(ckpt_dir))
    inference_weights_dir = absolute_ckpt_dir.parent / f"{absolute_ckpt_dir.name}_inference_weights"
    os.makedirs(inference_weights_dir, exist_ok=True)
    print(f"Training checkpoints will be saved to: {absolute_ckpt_dir}")
    print(f"Inference weights will be saved to: {inference_weights_dir}")

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=orbax_save_interval_steps,
        max_to_keep=3,
        step_prefix='ckpt_step', # Changed prefix slightly for clarity
    )
    mngr = ocp.CheckpointManager(absolute_ckpt_dir, options=options)

    initial_global_step_from_ckpt = -1 # Sentinel for no checkpoint
    latest_orbax_step = mngr.latest_step()

    if latest_orbax_step is not None:
        print(f"Found existing checkpoint at global_step {latest_orbax_step}. Restoring...")
        target_restore_dict = {
            'state_enc_A': state_enc_A, 'state_dec_A': state_dec_A,
            'state_enc_B': state_enc_B, 'state_dec_B': state_dec_B,
            'saved_epoch_idx': 0, 'saved_step_in_epoch_idx': 0 # Placeholders for metadata
        }
        try:
            restored_items = mngr.restore(latest_orbax_step, args=ocp.args.StandardRestore(target_restore_dict))
            state_enc_A = restored_items['state_enc_A']
            state_dec_A = restored_items['state_dec_A']
            state_enc_B = restored_items['state_enc_B']
            state_dec_B = restored_items['state_dec_B']
            # Metadata can be logged if needed:
            # restored_epoch = restored_items['saved_epoch_idx']
            # restored_step_in_epoch = restored_items['saved_step_in_epoch_idx']
            initial_global_step_from_ckpt = latest_orbax_step
            print(f"Successfully restored from global_step {initial_global_step_from_ckpt}.")
        except Exception as e:
            print(f"Warning: Failed to restore checkpoint {latest_orbax_step}: {e}. Starting from scratch.")
            initial_global_step_from_ckpt = -1
            # States remain as freshly initialized
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Determine loop starting points
    next_global_step_to_run = 0
    if initial_global_step_from_ckpt != -1: # A checkpoint was successfully loaded
        next_global_step_to_run = initial_global_step_from_ckpt + 1
    
    start_epoch_for_loop = next_global_step_to_run // num_steps_per_epoch
    start_step_in_epoch_for_loop = next_global_step_to_run % num_steps_per_epoch

    # Adjust PRNG keys based on the global step we are starting from
    key_data = jax.random.fold_in(key_data_master, next_global_step_to_run)
    current_key_dropout_master = jax.random.fold_in(key_dropout_master, next_global_step_to_run)
    
    # Replicate states across devices (must happen after potential restoration and before pmap)
    state_enc_A = jax.device_put_replicated(state_enc_A, jax.devices())
    state_dec_A = jax.device_put_replicated(state_dec_A, jax.devices())
    state_enc_B = jax.device_put_replicated(state_enc_B, jax.devices())
    state_dec_B = jax.device_put_replicated(state_dec_B, jax.devices())

    print(f"Starting training loop from Epoch {start_epoch_for_loop}, Step-in-epoch {start_step_in_epoch_for_loop} (Global Step {next_global_step_to_run}).")
    print(f"Total epochs: {num_epochs}. Steps per epoch: {num_steps_per_epoch}.")
    print(f"Global batch size: {global_batch_size} ({batch_size_per_device} per device on {NUM_DEVICES} devices)")

    for epoch in range(start_epoch_for_loop, num_epochs):
        key_data, key_epoch_data = jax.random.split(key_data)
        current_key_dropout_master, key_epoch_dropout = jax.random.split(current_key_dropout_master)
        
        epoch_total_loss = 0.0
        epoch_loss1_avg = 0.0
        epoch_loss2_avg = 0.0
        
        steps_in_current_epoch = range(num_steps_per_epoch)
        if epoch == start_epoch_for_loop: # Only for the first epoch of this run
            steps_in_current_epoch = range(start_step_in_epoch_for_loop, num_steps_per_epoch)

        for step_in_epoch in steps_in_current_epoch:
            global_step = epoch * num_steps_per_epoch + step_in_epoch
            
            key_step_data, key_epoch_data = jax.random.split(key_epoch_data) # Use key_epoch_data for next step's split
            key_dropout_step, key_epoch_dropout = jax.random.split(key_epoch_dropout) # Use key_epoch_dropout for next step's split
            
            batch_images_host = generate_batch(key_step_data, global_batch_size, image_size=image_size)
            sharded_batch_images = batch_images_host.reshape(
                (NUM_DEVICES, batch_size_per_device) + batch_images_host.shape[1:]
            )
            step_dropout_keys = jax.random.split(key_dropout_step, NUM_DEVICES)

            state_enc_A, state_dec_A, state_enc_B, state_dec_B, metrics = train_step(
                state_enc_A, state_dec_A, state_enc_B, state_dec_B,
                encoder, decoder, 
                sharded_batch_images, step_dropout_keys
            )
            
            epoch_total_loss += metrics['total_loss'][0] 
            epoch_loss1_avg += metrics['loss_A_speaks'][0]
            epoch_loss2_avg += metrics['loss_B_speaks'][0]

            if global_step % 10 == 0: 
                print(f"  Epoch {epoch+1}/{num_epochs}, Step {step_in_epoch+1}/{num_steps_per_epoch}, "
                      f"Global Step: {global_step}, "
                      f"Step Total Loss: {metrics['total_loss'][0]:.4f}, "
                      f"Loss A->B: {metrics['loss_A_speaks'][0]:.4f}, "
                      f"Loss B->A: {metrics['loss_B_speaks'][0]:.4f}")

            # Save full training checkpoint (Orbax)
            if mngr.should_save(global_step):
                # Unreplicate states for saving (Orbax handles sharding if needed, but explicit unreplicate is safer for StandardSave)
                items_to_save = {
                    'state_enc_A': jax.device_get(state_enc_A), # Gets from one device
                    'state_dec_A': jax.device_get(state_dec_A),
                    'state_enc_B': jax.device_get(state_enc_B),
                    'state_dec_B': jax.device_get(state_dec_B),
                    'saved_epoch_idx': epoch,         # Metadata: current epoch index
                    'saved_step_in_epoch_idx': step_in_epoch # Metadata: current step in epoch index
                }
                mngr.save(global_step, args=ocp.args.StandardSave(items_to_save))
                mngr.wait_until_finished() # Ensure save completes before proceeding too far
                print(f"Saved Orbax checkpoint at global_step {global_step}.")

            # Save inference-only model weights
            if global_step > 0 and global_step % inference_weights_save_steps == 0:
                print(f"Saving inference weights at global_step {global_step}...")
                states_to_save_params = {
                    "enc_A": state_enc_A, "dec_A": state_dec_A,
                    "enc_B": state_enc_B, "dec_B": state_dec_B,
                }
                for name, state_obj in states_to_save_params.items():
                    unreplicated_params = jax.device_get(state_obj.params) # Get params from one device
                    params_bytes = serialization.to_bytes(unreplicated_params)
                    file_path = inference_weights_dir / f"{name}_params_step_{global_step}.msgpack"
                    with open(file_path, "wb") as f:
                        f.write(params_bytes)
                print(f"Inference weights saved to {inference_weights_dir}")


        num_steps_this_epoch = len(steps_in_current_epoch)
        if num_steps_this_epoch > 0:
            avg_epoch_loss = epoch_total_loss / num_steps_this_epoch
            avg_loss1 = epoch_loss1_avg / num_steps_this_epoch
            avg_loss2 = epoch_loss2_avg / num_steps_this_epoch
            print(f"Epoch {epoch+1} completed. Avg Total Loss: {avg_epoch_loss:.4f}, "
                  f"Avg Loss A->B: {avg_loss1:.4f}, Avg Loss B->A: {avg_loss2:.4f}")
        else:
            print(f"Epoch {epoch+1} completed (no steps run in this epoch, likely due to resumption).")


    mngr.wait_until_finished()
    mngr.close()
    print("Training finished. Checkpoint manager closed.")

if __name__ == '__main__':
    # For testing on a resource-constrained environment (like a CPU or single GPU):
    # main_training_loop(
    #     num_epochs=5, 
    #     batch_size_per_device=1, 
    #     image_size=128, # Smaller image
    #     latent_channels=4, # Fewer latent channels
    #     num_steps_per_epoch=10, # Fewer steps per epoch for faster testing
    #     orbax_save_interval_steps=5,
    #     inference_weights_save_steps=8 
    # )

    # Full scale as per request (might be slow and memory intensive)
    main_training_loop(
        num_epochs=1000, 
        batch_size_per_device=1, 
        learning_rate=1e-4,
        image_size=1080, 
        latent_channels=4, # Can be 4 or 8 as requested
        num_steps_per_epoch=50, # Example steps per epoch
        orbax_save_interval_steps=2000, # Save full checkpoint every 50 steps
        inference_weights_save_steps=1000 # Save inference weights every 1000 steps
    )