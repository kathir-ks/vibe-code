import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import flax.linen as nn
from flax.training import train_state, checkpoints
from flax.core import freeze, unfreeze
import optax
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os
import time
from functools import partial
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Configuration
@dataclass
class Config:
    # Data parameters
    image_size: int = 1080
    encoded_size: int = 32
    max_circles: int = 30
    min_radius: int = 10
    max_radius: int = 50
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    
    # Model parameters
    encoder_features: Tuple[int, ...] = (64, 128, 256, 512)
    decoder_features: Tuple[int, ...] = (512, 256, 128, 64)
    
    # Checkpoint parameters
    checkpoint_dir: str = "checkpoints"
    inference_checkpoint_dir: str = "inference_checkpoints"
    save_every_n_steps: int = 500
    inference_save_every_n_steps: int = 2000
    keep_last_n_checkpoints: int = 5
    
    # TPU parameters
    mesh_shape: Tuple[int, int] = (1, 8)  # Adjust based on your TPU configuration

config = Config()

# Data Generation
class CircleDataGenerator:
    def __init__(self, config: Config, seed: int = 42):
        self.config = config
        self.key = random.PRNGKey(seed)
    
    def generate_circle_params(self, key: jax.Array, num_circles: int) -> Dict[str, jax.Array]:
        """Generate random circle parameters ensuring no overlap."""
        keys = random.split(key, 4)
        
        # Generate random centers and radii
        centers_x = random.uniform(keys[0], (num_circles,), 
                                 minval=self.config.max_radius, 
                                 maxval=self.config.image_size - self.config.max_radius)
        centers_y = random.uniform(keys[1], (num_circles,), 
                                 minval=self.config.max_radius, 
                                 maxval=self.config.image_size - self.config.max_radius)
        radii = random.uniform(keys[2], (num_circles,), 
                             minval=self.config.min_radius, 
                             maxval=self.config.max_radius)
        
        return {
            'centers': jnp.stack([centers_x, centers_y], axis=1),
            'radii': radii,
            'count': num_circles
        }
    
    def render_circles(self, circle_params: Dict[str, jax.Array]) -> jax.Array:
        """Render circles into a binary image."""
        y, x = jnp.mgrid[0:self.config.image_size, 0:self.config.image_size]
        image = jnp.zeros((self.config.image_size, self.config.image_size))
        
        centers = circle_params['centers']
        radii = circle_params['radii']
        
        for i in range(circle_params['count']):
            cx, cy = centers[i]
            r = radii[i]
            mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
            image = jnp.maximum(image, mask.astype(jnp.float32))
        
        return image
    
    def generate_batch(self, batch_size: int) -> Tuple[jax.Array, jax.Array]:
        """Generate a batch of circle images with their numerical representations."""
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, batch_size)
        
        images = []
        numerical_targets = []
        
        for key in keys:
            key1, key2 = random.split(key)
            num_circles = random.randint(key1, (), 0, self.config.max_circles + 1)
            
            if num_circles > 0:
                circle_params = self.generate_circle_params(key2, num_circles)
                image = self.render_circles(circle_params)
                
                # Numerical representation: [count, center_x1, center_y1, radius1, ...]
                numerical = jnp.zeros(1 + self.config.max_circles * 3)
                numerical = numerical.at[0].set(num_circles)
                
                for i in range(num_circles):
                    idx = 1 + i * 3
                    numerical = numerical.at[idx:idx+3].set(
                        jnp.array([circle_params['centers'][i, 0], 
                                  circle_params['centers'][i, 1], 
                                  circle_params['radii'][i]])
                    )
            else:
                image = jnp.zeros((self.config.image_size, self.config.image_size))
                numerical = jnp.zeros(1 + self.config.max_circles * 3)
            
            images.append(image)
            numerical_targets.append(numerical)
        
        return jnp.stack(images), jnp.stack(numerical_targets)

# Model Architecture
class Encoder(nn.Module):
    features: Tuple[int, ...]
    encoded_size: int
    
    @nn.compact
    def __call__(self, x):
        # Add channel dimension
        x = jnp.expand_dims(x, axis=-1)
        
        # Convolutional layers with downsampling
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.BatchNorm()(x)
        
        # Global average pooling and reshape to encoded size
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.encoded_size * self.encoded_size)(x)
        x = nn.relu(x)
        x = x.reshape(-1, self.encoded_size, self.encoded_size, 1)
        
        return x

class Decoder(nn.Module):
    features: Tuple[int, ...]
    output_size: int
    output_channels: int = 1
    
    @nn.compact
    def __call__(self, x):
        # Flatten encoded representation
        x = x.reshape(x.shape[0], -1)
        
        # Dense layers
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
            x = nn.BatchNorm()(x)
        
        # Output layer for numerical representation
        numerical_out = nn.Dense(1 + 30 * 3)(x)  # count + max_circles * 3
        
        return numerical_out

class CommunicationModel(nn.Module):
    encoder: Encoder
    decoder: Decoder
    
    def setup(self):
        self.encoder = Encoder(
            features=config.encoder_features,
            encoded_size=config.encoded_size
        )
        self.decoder = Decoder(
            features=config.decoder_features,
            output_size=config.image_size,
            output_channels=1
        )
    
    def encode(self, image):
        return self.encoder(image)
    
    def decode(self, encoded):
        return self.decoder(encoded)
    
    def __call__(self, image, mode='speaking'):
        if mode == 'speaking':
            return self.encode(image)
        else:  # listening mode
            return self.decode(image)

# Training State Management
def create_train_state(model, key, input_shape, learning_rate):
    """Create initial training state."""
    dummy_input = jnp.ones(input_shape)
    variables = model.init(key, dummy_input)
    
    # Create learning rate schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.num_epochs * 1000,  # Approximate steps per epoch
        end_value=learning_rate * 0.01
    )
    
    optimizer = optax.adamw(learning_rate=schedule)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

# Loss Functions
def communication_loss(params_speaker, params_listener, model, images, numerical_targets):
    """Compute communication loss between two models."""
    
    # Speaker encodes the image
    encoded = model.apply({'params': params_speaker}, images, mode='speaking')
    
    # Listener decodes the encoded representation
    decoded_numerical = model.apply({'params': params_listener}, encoded, mode='listening')
    
    # Loss: MSE between predicted and actual numerical representation
    mse_loss = jnp.mean((decoded_numerical - numerical_targets) ** 2)
    
    # Additional regularization: encourage meaningful encoding
    encoding_variance = jnp.var(encoded)
    regularization = -jnp.log(encoding_variance + 1e-8)  # Encourage variance
    
    total_loss = mse_loss + 0.01 * regularization
    
    return total_loss, {
        'mse_loss': mse_loss,
        'encoding_variance': encoding_variance,
        'total_loss': total_loss
    }

@jit
def train_step(state_speaker, state_listener, images, numerical_targets):
    """Single training step for both models."""
    
    def loss_fn_speaker(params_speaker):
        return communication_loss(
            params_speaker, state_listener.params, 
            CommunicationModel(), images, numerical_targets
        )
    
    def loss_fn_listener(params_listener):
        return communication_loss(
            state_speaker.params, params_listener,
            CommunicationModel(), images, numerical_targets
        )
    
    # Compute gradients for both models
    (loss_speaker, metrics_speaker), grads_speaker = jax.value_and_grad(
        loss_fn_speaker, has_aux=True
    )(state_speaker.params)
    
    (loss_listener, metrics_listener), grads_listener = jax.value_and_grad(
        loss_fn_listener, has_aux=True
    )(state_listener.params)
    
    # Update both models
    new_state_speaker = state_speaker.apply_gradients(grads=grads_speaker)
    new_state_listener = state_listener.apply_gradients(grads=grads_listener)
    
    metrics = {
        'speaker_loss': loss_speaker,
        'listener_loss': loss_listener,
        **{f'speaker_{k}': v for k, v in metrics_speaker.items()},
        **{f'listener_{k}': v for k, v in metrics_listener.items()}
    }
    
    return new_state_speaker, new_state_listener, metrics

# Checkpoint Management
class CheckpointManager:
    def __init__(self, config: Config):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.inference_dir = Path(config.inference_checkpoint_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.inference_dir.mkdir(exist_ok=True)
    
    def save_training_checkpoint(self, step: int, state_speaker, state_listener, metrics_history):
        """Save training checkpoint with rotation."""
        checkpoint_data = {
            'step': step,
            'state_speaker': state_speaker,
            'state_listener': state_listener,
            'metrics_history': metrics_history
        }
        
        # Save with step number
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}"
        checkpoints.save_checkpoint(
            checkpoint_path, checkpoint_data, step, 
            keep=self.config.keep_last_n_checkpoints
        )
        
        print(f"Saved training checkpoint at step {step}")
    
    def save_inference_checkpoint(self, step: int, state_speaker, state_listener):
        """Save inference-only checkpoint."""
        inference_data = {
            'step': step,
            'speaker_params': state_speaker.params,
            'listener_params': state_listener.params,
            'config': self.config
        }
        
        checkpoint_path = self.inference_dir / f"inference_{step}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(inference_data, f)
        
        print(f"Saved inference checkpoint at step {step}")
    
    def load_latest_checkpoint(self):
        """Load the latest training checkpoint."""
        try:
            return checkpoints.restore_checkpoint(self.checkpoint_dir, None)
        except:
            return None

# Training Loop
def train_models():
    """Main training loop."""
    
    # Initialize TPU
    print("Initializing TPU...")
    jax.distributed.initialize()
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX local devices: {jax.local_devices()}")
    
    # Initialize data generator
    data_generator = CircleDataGenerator(config)
    
    # Initialize models
    key = random.PRNGKey(42)
    key_speaker, key_listener = random.split(key)
    
    model = CommunicationModel()
    input_shape = (config.batch_size, config.image_size, config.image_size)
    
    # Create training states
    state_speaker = create_train_state(model, key_speaker, input_shape, config.learning_rate)
    state_listener = create_train_state(model, key_listener, input_shape, config.learning_rate)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(config)
    
    # Try to restore from checkpoint
    restored_checkpoint = checkpoint_manager.load_latest_checkpoint()
    if restored_checkpoint is not None:
        print("Restored from checkpoint")
        state_speaker = restored_checkpoint['state_speaker']
        state_listener = restored_checkpoint['state_listener']
        start_step = restored_checkpoint['step']
        metrics_history = restored_checkpoint['metrics_history']
    else:
        start_step = 0
        metrics_history = []
    
    # Training loop
    step = start_step
    total_steps = config.num_epochs * 1000  # Approximate
    
    print(f"Starting training from step {start_step}")
    
    while step < total_steps:
        # Generate batch
        images, numerical_targets = data_generator.generate_batch(config.batch_size)
        
        # Training step
        start_time = time.time()
        state_speaker, state_listener, metrics = train_step(
            state_speaker, state_listener, images, numerical_targets
        )
        step_time = time.time() - start_time
        
        # Log metrics
        if step % 100 == 0:
            print(f"Step {step}: Speaker Loss: {metrics['speaker_loss']:.4f}, "
                  f"Listener Loss: {metrics['listener_loss']:.4f}, "
                  f"Time: {step_time:.3f}s")
        
        metrics_history.append(metrics)
        
        # Save training checkpoint
        if step % config.save_every_n_steps == 0 and step > 0:
            checkpoint_manager.save_training_checkpoint(
                step, state_speaker, state_listener, metrics_history
            )
        
        # Save inference checkpoint
        if step % config.inference_save_every_n_steps == 0 and step > 0:
            checkpoint_manager.save_inference_checkpoint(
                step, state_speaker, state_listener
            )
        
        step += 1
    
    # Final checkpoint save
    checkpoint_manager.save_training_checkpoint(
        step, state_speaker, state_listener, metrics_history
    )
    checkpoint_manager.save_inference_checkpoint(
        step, state_speaker, state_listener
    )
    
    print("Training completed!")
    return state_speaker, state_listener, metrics_history

# Inference and Evaluation
def evaluate_communication(state_speaker, state_listener, data_generator, num_samples=10):
    """Evaluate the communication between models."""
    model = CommunicationModel()
    
    # Generate test data
    test_images, test_numerical = data_generator.generate_batch(num_samples)
    
    # Test communication pipeline
    encoded = model.apply({'params': state_speaker.params}, test_images, mode='speaking')
    decoded_numerical = model.apply({'params': state_listener.params}, encoded, mode='listening')
    
    # Compute accuracy metrics
    mse = jnp.mean((decoded_numerical - test_numerical) ** 2)
    mae = jnp.mean(jnp.abs(decoded_numerical - test_numerical))
    
    # Count accuracy (first element is the count)
    predicted_counts = jnp.round(decoded_numerical[:, 0])
    actual_counts = test_numerical[:, 0]
    count_accuracy = jnp.mean(predicted_counts == actual_counts)
    
    print(f"Evaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Count Accuracy: {count_accuracy:.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'count_accuracy': count_accuracy,
        'test_images': test_images,
        'test_numerical': test_numerical,
        'decoded_numerical': decoded_numerical
    }

# Main execution
if __name__ == "__main__":
    # Set up for TPU usage
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    # Start training
    speaker_state, listener_state, history = train_models()
    
    # Evaluate final models
    data_gen = CircleDataGenerator(config)
    evaluation_results = evaluate_communication(speaker_state, listener_state, data_gen)
    
    print("Mathematical re-origination experiment completed!")
    print("Models have learned to communicate about spatial-numerical relationships.")