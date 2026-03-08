"""
Training pipeline for cross-modal object communication system.
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import numpy as np
from typing import Tuple, Dict, Any
import time
import pickle
import os
from functools import partial

from models import CrossModalModel, create_model_pair
from data import DataLoader, DataConfig, ObjectCounter, Visualizer


class TrainingConfig:
    """Configuration for training parameters."""
    
    def __init__(self):
        # Training hyperparameters
        self.learning_rate = 1e-4
        self.batch_size = 2  # Small due to large images and TPU memory
        self.num_epochs = 200
        self.batches_per_epoch = 50
        
        # Loss weights
        self.count_loss_weight = 1.0
        self.reconstruction_loss_weight = 0.1
        self.latent_sparsity_weight = 0.01
        
        # Evaluation parameters
        self.eval_frequency = 10
        self.save_frequency = 20
        
        # Paths
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


class CrossModalTrainer:
    """Main trainer class for the cross-modal communication system."""
    
    def __init__(self, train_config: TrainingConfig, data_config: DataConfig):
        self.train_config = train_config
        self.data_config = data_config
        self.data_loader = DataLoader(data_config)
        
        # Initialize models
        self.model1, self.model2 = create_model_pair()
        
        # Training states will be initialized in setup_training
        self.state1 = None
        self.state2 = None
        
        # Training history
        self.history = {
            'epoch': [],
            'loss1': [],
            'loss2': [],
            'count_accuracy1': [],
            'count_accuracy2': [],
            'train_time': []
        }
    
    def create_train_state(self, model: CrossModalModel, key: jax.random.PRNGKey, 
                          input_shape: Tuple[int, ...]) -> train_state.TrainState:
        """Create training state for a model."""
        params = model.init(key, jnp.ones(input_shape))['params']
        tx = optax.adam(self.train_config.learning_rate)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    def setup_training(self, key: jax.random.PRNGKey):
        """Initialize training states and JIT-compile training functions."""
        input_shape = (self.train_config.batch_size, self.data_config.img_size, 
                      self.data_config.img_size, 1)
        
        key1, key2 = jax.random.split(key)
        self.state1 = self.create_train_state(self.model1, key1, input_shape)
        self.state2 = self.create_train_state(self.model2, key2, input_shape)
        
        # JIT compile training step
        self.train_step_jit = jax.jit(self._train_step)
        
        print("Training setup complete!")
        print(f"Model 1 parameters: {sum(x.size for x in jax.tree_leaves(self.state1.params)):,}")
        print(f"Model 2 parameters: {sum(x.size for x in jax.tree_leaves(self.state2.params)):,}")
    
    def _compute_losses(self, state1: train_state.TrainState, state2: train_state.TrainState,
                       batch_circles: jnp.ndarray, batch_squares: jnp.ndarray,
                       true_circle_counts: jnp.ndarray, true_square_counts: jnp.ndarray) -> Tuple[float, float]:
        """Compute losses for both models."""
        
        def loss_fn_1(params1, params2):
            # Model 1: circles -> latent -> squares (decoded by model 2)
            _, latent1 = state1.apply_fn({'params': params1}, batch_circles)
            reconstructed_squares = state2.apply_fn({'params': params2}, latent1, method='decode')
            
            # Count objects in reconstructed squares
            predicted_square_counts = ObjectCounter.batch_count_objects(reconstructed_squares)
            
            # Count loss: difference between true circle count and predicted square count
            count_loss = jnp.mean(jnp.abs(true_circle_counts - predicted_square_counts))
            
            # Reconstruction loss to encourage meaningful representations
            reconstruction_loss = jnp.mean((reconstructed_squares - batch_squares)**2)
            
            # Latent sparsity loss to encourage efficient encoding
            latent_sparsity = jnp.mean(jnp.abs(latent1))
            
            total_loss = (self.train_config.count_loss_weight * count_loss + 
                         self.train_config.reconstruction_loss_weight * reconstruction_loss +
                         self.train_config.latent_sparsity_weight * latent_sparsity)
            
            return total_loss, (count_loss, reconstruction_loss, latent_sparsity)
        
        def loss_fn_2(params1, params2):
            # Model 2: squares -> latent -> circles (decoded by model 1)
            _, latent2 = state2.apply_fn({'params': params2}, batch_squares)
            reconstructed_circles = state1.apply_fn({'params': params1}, latent2, method='decode')
            
            # Count objects in reconstructed circles
            predicted_circle_counts = ObjectCounter.batch_count_objects(reconstructed_circles)
            
            # Count loss: difference between true square count and predicted circle count
            count_loss = jnp.mean(jnp.abs(true_square_counts - predicted_circle_counts))
            
            # Reconstruction loss
            reconstruction_loss = jnp.mean((reconstructed_circles - batch_circles)**2)
            
            # Latent sparsity loss
            latent_sparsity = jnp.mean(jnp.abs(latent2))
            
            total_loss = (self.train_config.count_loss_weight * count_loss + 
                         self.train_config.reconstruction_loss_weight * reconstruction_loss +
                         self.train_config.latent_sparsity_weight * latent_sparsity)
            
            return total_loss, (count_loss, reconstruction_loss, latent_sparsity)
        
        # Compute losses and gradients
        (loss1, aux1), grads1 = jax.value_and_grad(loss_fn_1, has_aux=True)(state1.params, state2.params)
        (loss2, aux2), grads2 = jax.value_and_grad(loss_fn_2, has_aux=True)(state1.params, state2.params)
        
        return loss1, loss2, grads1, grads2, aux1, aux2
    
    def _train_step(self, state1: train_state.TrainState, state2: train_state.TrainState,
                   batch_circles: jnp.ndarray, batch_squares: jnp.ndarray,
                   true_circle_counts: jnp.ndarray, true_square_counts: jnp.ndarray):
        """Single training step."""
        
        loss1, loss2, grads1, grads2, aux1, aux2 = self._compute_losses(
            state1, state2, batch_circles, batch_squares, true_circle_counts, true_square_counts
        )
        
        # Update parameters
        state1 = state1.apply_gradients(grads=grads1)
        state2 = state2.apply_gradients(grads=grads2)
        
        return state1, state2, loss1, loss2, aux1, aux2
    
    def evaluate(self, key: jax.random.PRNGKey, num_eval_batches: int = 10) -> Dict[str, float]:
        """Evaluate model performance."""
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_count_acc1 = 0.0
        total_count_acc2 = 0.0
        
        for _ in range(num_eval_batches):
            # Generate evaluation batch
            circles, squares, c_counts, s_counts, key = self.data_loader.generate_batch(
                self.train_config.batch_size, key
            )
            
            # Forward pass for evaluation
            _, latent1 = self.state1.apply_fn({'params': self.state1.params}, circles)
            _, latent2 = self.state2.apply_fn({'params': self.state2.params}, squares)
            
            reconstructed_squares = self.state2.apply_fn({'params': self.state2.params}, latent1, method='decode')
            reconstructed_circles = self.state1.apply_fn({'params': self.state1.params}, latent2, method='decode')
            
            # Count accuracy
            pred_s_counts = ObjectCounter.batch_count_objects(reconstructed_squares)
            pred_c_counts = ObjectCounter.batch_count_objects(reconstructed_circles)
            
            count_acc1 = jnp.mean(jnp.abs(c_counts - pred_s_counts) <= 1)  # Within 1 object
            count_acc2 = jnp.mean(jnp.abs(s_counts - pred_c_counts) <= 1)
            
            # Compute losses
            loss1, loss2, _, _, _, _ = self._compute_losses(
                self.state1, self.state2, circles, squares, c_counts, s_counts
            )
            
            total_loss1 += loss1
            total_loss2 += loss2
            total_count_acc1 += count_acc1
            total_count_acc2 += count_acc2
        
        return {
            'eval_loss1': total_loss1 / num_eval_batches,
            'eval_loss2': total_loss2 / num_eval_batches,
            'eval_count_acc1': total_count_acc1 / num_eval_batches,
            'eval_count_acc2': total_count_acc2 / num_eval_batches
        }
    
    def save_checkpoint(self, epoch: int, filepath: str = None):
        """Save model checkpoints."""
        if filepath is None:
            filepath = os.path.join(self.train_config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")
        
        checkpoint = {
            'epoch': epoch,
            'state1': self.state1,
            'state2': self.state2,
            'train_config': self.train_config,
            'data_config': self.data_config,
            'history': self.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoints."""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.state1 = checkpoint['state1']
        self.state2 = checkpoint['state2']
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch']
    
    def train(self, key: jax.random.PRNGKey):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.train_config.num_epochs):
            epoch_start_time = time.time()
            epoch_loss1 = 0.0
            epoch_loss2 = 0.0
            
            # Training batches
            for batch_idx in range(self.train_config.batches_per_epoch):
                # Generate batch
                circles, squares, c_counts, s_counts, key = self.data_loader.generate_batch(
                    self.train_config.batch_size, key
                )
                
                # Training step
                self.state1, self.state2, loss1, loss2, aux1, aux2 = self.train_step_jit(
                    self.state1, self.state2, circles, squares, c_counts, s_counts
                )
                
                epoch_loss1 += loss1
                epoch_loss2 += loss2
            
            # Average losses
            avg_loss1 = epoch_loss1 / self.train_config.batches_per_epoch
            avg_loss2 = epoch_loss2 / self.train_config.batches_per_epoch
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['loss1'].append(float(avg_loss1))
            self.history['loss2'].append(float(avg_loss2))
            self.history['train_time'].append(epoch_time)
            
            # Evaluation
            if epoch % self.train_config.eval_frequency == 0:
                eval_metrics = self.evaluate(key)
                self.history['count_accuracy1'].append(float(eval_metrics['eval_count_acc1']))
                self.history['count_accuracy2'].append(float(eval_metrics['eval_count_acc2']))
                
                print(f"Epoch {epoch:3d} | Loss1: {avg_loss1:.4f} | Loss2: {avg_loss2:.4f} | "
                      f"Acc1: {eval_metrics['eval_count_acc1']:.3f} | Acc2: {eval_metrics['eval_count_acc2']:.3f} | "
                      f"Time: {epoch_time:.1f}s")
            else:
                print(f"Epoch {epoch:3d} | Loss1: {avg_loss1:.4f} | Loss2: {avg_loss2:.4f} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            if epoch % self.train_config.save_frequency == 0 and epoch > 0:
                self.save_checkpoint(epoch)
        
        print("Training completed!")
        self.save_checkpoint(self.train_config.num_epochs - 1, "final_checkpoint.pkl")
    
    def visualize_training_progress(self):
        """Plot training progress."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['epoch'], self.history['loss1'], label='Model 1')
        axes[0, 0].plot(self.history['epoch'], self.history['loss2'], label='Model 2')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Count accuracy
        eval_epochs = self.history['epoch'][::self.train_config.eval_frequency]
        axes[0, 1].plot(eval_epochs, self.history['count_accuracy1'], label='Model 1')
        axes[0, 1].plot(eval_epochs, self.history['count_accuracy2'], label='Model 2')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Count Accuracy')
        axes[0, 1].set_title('Count Accuracy (±1 object)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Training time per epoch
        axes[1, 0].plot(self.history['epoch'], self.history['train_time'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].grid(True)
        
        # Loss distribution
        axes[1, 1].hist(self.history['loss1'], alpha=0.7, label='Model 1', bins=20)
        axes[1, 1].hist(self.history['loss2'], alpha=0.7, label='Model 2', bins=20)
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example training script
    train_config = TrainingConfig()
    data_config = DataConfig()
    
    # Adjust for quick testing
    train_config.num_epochs = 50
    train_config.batches_per_epoch = 10
    train_config.eval_frequency = 5
    train_config.save_frequency = 10
    
    # Initialize trainer
    trainer = CrossModalTrainer(train_config, data_config)
    
    # Setup training
    key = jax.random.PRNGKey(42)
    trainer.setup_training(key)
    
    # Start training
    trainer.train(key)
    
    # Visualize results
    trainer.visualize_training_progress()