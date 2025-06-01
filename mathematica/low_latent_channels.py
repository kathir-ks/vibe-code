import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from functools import partial
from typing import Sequence, Tuple, Dict, Any
from flax.training import train_state

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
    canvas_np = np.zeros((image_size, image_size), dtype=np.float32)
    placed_circles_params = [] # Store (x, y, r) of placed circles

    # Split keys for each circle attempt
    # Ensure enough keys for worst-case (max_circles * max_placement_attempts)
    # This can be large, but jax.random.split handles it.
    # If num_circles is 0, this will be an empty array, which is fine.
    if num_circles > 0:
        circle_keys = jax.random.split(key_circles_props, num_circles * max_placement_attempts)
    else:
        circle_keys = jnp.array([]) # Empty array if no circles

    key_idx = 0

    for i in range(num_circles):
        placed_successfully = False
        for attempt in range(max_placement_attempts):
            # Check if key_idx is out of bounds for circle_keys (can happen if num_circles was small)
            if key_idx >= len(circle_keys):
                 # This case should ideally not be reached if keys are split for num_circles * max_placement_attempts
                 # but as a safeguard if num_circles was small initially.
                break

            current_key = circle_keys[key_idx]
            key_idx += 1
            
            key_radius, key_x, key_y = jax.random.split(current_key, 3)
            
            radius = jax.random.randint(key_radius, shape=(), minval=min_radius, maxval=max_radius + 1)
            # Ensure circle center allows full circle within bounds
            # Add a small epsilon to maxval if image_size - radius can be equal to radius (minval)
            center_x = jax.random.randint(key_x, shape=(), minval=radius, maxval=max(radius + 1, image_size - radius))
            center_y = jax.random.randint(key_y, shape=(), minval=radius, maxval=max(radius + 1, image_size - radius))
            
            # Check for overlap with previously placed circles
            overlap = False
            if placed_circles_params: 
                for pc_x, pc_y, pc_r in placed_circles_params:
                    # Ensure all operands are float for calculation
                    dist_sq = (float(center_x) - float(pc_x))**2 + (float(center_y) - float(pc_y))**2
                    min_dist_sq = (float(radius) + float(pc_r))**2
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
        if not placed_successfully and key_idx >= len(circle_keys) and i < num_circles -1 :
            # If we ran out of keys before trying all circles, break outer loop
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
    latent_channels: int = 1 # MODIFIED: Default to 1 latent channel
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
    # The number of channels for the first ConvTranspose layer in the decoder
    # should match the features of the last layer before the latent_channels conv in the encoder.
    first_deconv_features: int = 512 # Corresponds to self.encoder_features[4]
    encoder_features: Sequence[int] = (64, 128, 256, 512, 512) # Should match encoder's features in reverse


    @nn.compact
    def __call__(self, z: jnp.ndarray, training: bool) -> jnp.ndarray:
        # Input z: (batch, 32, 32, latent_channels) from Encoder
        
        # Layer 1: 32x32 -> 34x34 
        # The features here should match the channel depth feeding into the Encoder's final 3x3 VALID conv.
        # This was self.features[4] (e.g. 512) in the Encoder before the latent_channels conv.
        x = nn.ConvTranspose(features=self.first_deconv_features, kernel_size=(3, 3), strides=(1, 1), padding='VALID', name='dec_conv_t_1')(z)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_1')(x)
        x = nn.relu(x)
        # Current shape: (batch, 34, 34, self.first_deconv_features)

        # Layer 2: 34x34 -> 68x68
        x = nn.ConvTranspose(features=self.encoder_features[3], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='dec_conv_t_2')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_2')(x)
        x = nn.relu(x)
        # Current shape: (batch, 68, 68, encoder_features[3])

        # Layer 3: 68x68 -> 136x136 (Encoder had 135x135 here)
        x = nn.ConvTranspose(features=self.encoder_features[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='dec_conv_t_3')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_3')(x)
        x = nn.relu(x)
        # Current shape: (batch, 136, 136, encoder_features[2])
        if x.shape[1] == 136: 
            x = x[:, :135, :135, :]


        # Layer 4: 135x135 -> 270x270
        x = nn.ConvTranspose(features=self.encoder_features[1], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='dec_conv_t_4')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_4')(x)
        x = nn.relu(x)
        # Current shape: (batch, 270, 270, encoder_features[1])

        # Layer 5: 270x270 -> 540x540
        x = nn.ConvTranspose(features=self.encoder_features[0], kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='dec_conv_t_5')(x)
        x = nn.BatchNorm(use_running_average=not training, name='dec_bn_5')(x)
        x = nn.relu(x)
        # Current shape: (batch, 540, 540, encoder_features[0])

        # Final Layer: 540x540 -> 1080x1080
        x = nn.ConvTranspose(features=1, kernel_size=(4, 4), strides=(2, 2), padding='SAME', name='dec_conv_t_final')(x)
        # Current shape: (batch, 1080, 1080, 1)
        
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

def communication_loss_fn(
    params_enc_A, batch_stats_enc_A,
    params_dec_A, batch_stats_dec_A,
    params_enc_B, batch_stats_enc_B,
    params_dec_B, batch_stats_dec_B,
    encoder_model: Encoder, decoder_model: Decoder,
    images: jnp.ndarray, dropout_key: jax.random.PRNGKey # Dropout key not used in current model, but good practice
):
    # --- Model A speaks, Model B listens ---
    variables_enc_A = {'params': params_enc_A, 'batch_stats': batch_stats_enc_A}
    latent_Z_from_A, new_batch_stats_enc_A = encoder_model.apply(
        variables_enc_A, images, training=True, mutable=['batch_stats']
    )
    
    variables_dec_B = {'params': params_dec_B, 'batch_stats': batch_stats_dec_B}
    reconstructed_by_B, new_batch_stats_dec_B = decoder_model.apply(
        variables_dec_B, latent_Z_from_A, training=True, mutable=['batch_stats']
    )
    loss1 = optax.sigmoid_binary_cross_entropy(reconstructed_by_B, images).mean()

    # --- Model B speaks, Model A listens ---
    variables_enc_B = {'params': params_enc_B, 'batch_stats': batch_stats_enc_B}
    latent_Z_from_B, new_batch_stats_enc_B = encoder_model.apply(
        variables_enc_B, images, training=True, mutable=['batch_stats']
    )

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
    latent_channels: int = 1 # MODIFIED: Default to 1 latent channel
):
    global_batch_size = batch_size_per_device * NUM_DEVICES
    
    key = jax.random.PRNGKey(0)
    key_init, key_data, key_dropout_master = jax.random.split(key, 3)

    # Initialize models
    # Encoder will be initialized with the 'latent_channels' value passed to this function
    encoder = Encoder(latent_channels=latent_channels) 
    # Decoder needs to know the features of the layer in encoder that feeds into the final latent conv
    # This is encoder.features[-1] if latent_channels conv is the very last one.
    # In our case, it's features[4] or 512.
    decoder = Decoder(original_image_size=image_size, first_deconv_features=encoder.features[-1])
    
    dummy_image_input = jnp.ones((1, image_size, image_size, 1), dtype=jnp.float32)
    # dummy_latent_input shape must match the output of the encoder
    dummy_latent_input = jnp.ones((1, encoder.latent_dim_spatial, encoder.latent_dim_spatial, latent_channels), dtype=jnp.float32)

    key_encA, key_decA, key_encB, key_decB = jax.random.split(key_init, 4)

    state_enc_A = create_train_state(key_encA, encoder, dummy_image_input, learning_rate)
    state_dec_A = create_train_state(key_decA, decoder, dummy_latent_input, learning_rate)
    state_enc_B = create_train_state(key_encB, encoder, dummy_image_input, learning_rate) 
    state_dec_B = create_train_state(key_decB, decoder, dummy_latent_input, learning_rate)

    state_enc_A = jax.device_put_replicated(state_enc_A, jax.devices())
    state_dec_A = jax.device_put_replicated(state_dec_A, jax.devices())
    state_enc_B = jax.device_put_replicated(state_enc_B, jax.devices())
    state_dec_B = jax.device_put_replicated(state_dec_B, jax.devices())

    print(f"Starting training for {num_epochs} epochs.")
    print(f"Global batch size: {global_batch_size} ({batch_size_per_device} per device on {NUM_DEVICES})")
    print(f"Latent channels: {latent_channels}")

    for epoch in range(num_epochs):
        key_data, key_epoch_data = jax.random.split(key_data)
        
        num_steps_per_epoch = 50 
        
        epoch_total_loss = 0.0
        epoch_loss1_avg = 0.0
        epoch_loss2_avg = 0.0

        for step in range(num_steps_per_epoch):
            key_step_data, key_dropout_step = jax.random.split(key_epoch_data, 2)
            
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
            
            current_total_loss = metrics['total_loss'][0]
            current_loss1 = metrics['loss_A_speaks'][0]
            current_loss2 = metrics['loss_B_speaks'][0]

            epoch_total_loss += current_total_loss
            epoch_loss1_avg += current_loss1
            epoch_loss2_avg += current_loss2


            if step % 10 == 0: 
                print(f"  Epoch {epoch+1}/{num_epochs}, Step {step+1}/{num_steps_per_epoch}, "
                      f"Step Total Loss: {current_total_loss:.4f}, "
                      f"Loss A->B: {current_loss1:.4f}, "
                      f"Loss B->A: {current_loss2:.4f}")

        avg_epoch_loss = epoch_total_loss / num_steps_per_epoch
        avg_loss1 = epoch_loss1_avg / num_steps_per_epoch
        avg_loss2 = epoch_loss2_avg / num_steps_per_epoch
        print(f"Epoch {epoch+1} completed. Avg Total Loss: {avg_epoch_loss:.4f}, "
              f"Avg Loss A->B: {avg_loss1:.4f}, Avg Loss B->A: {avg_loss2:.4f}")

    print("Training finished.")
    # To retrieve states for saving (example for state_enc_A):
    # unreplicated_state_enc_A_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state_enc_A.params))
    # unreplicated_state_enc_A_batch_stats = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state_enc_A.batch_stats))
    # Then save these using your preferred method (e.g., flax.serialization)

if __name__ == '__main__':
    # MODIFIED: Call with 1 latent channel for a very tight bottleneck.
    # You can change image_size to something smaller (e.g., 128 or 256) for faster testing.
    # batch_size_per_device=1 is recommended for 1080p images due to memory.
    main_training_loop(
         num_epochs=10, 
         batch_size_per_device=1, 
         learning_rate=1e-4,
         image_size=1080, 
         latent_channels=1 # Set to 1 or 2 as desired
    )
