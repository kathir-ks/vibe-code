"""
Neural network models for cross-modal object communication system.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


class Encoder(nn.Module):
    """
    Encoder that compresses 1080x1080 input to 32x32 latent space.
    
    Uses convolutional layers with stride 2 for progressive downsampling:
    1080 -> 540 -> 270 -> 135 -> 67 -> 33 -> 32
    """
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of encoder.
        
        Args:
            x: Input tensor of shape (batch, 1080, 1080, 1)
            
        Returns:
            Latent representation of shape (batch, 32, 32, 1)
        """
        # Progressive downsampling with increasing feature depth
        x = nn.Conv(features=32, kernel_size=(7, 7), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        # Final compression to single channel latent space
        x = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        
        # Ensure exact 32x32 output size
        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='bilinear')
        
        # Apply sigmoid to keep latent space values between 0 and 1 (B/W)
        x = nn.sigmoid(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder that reconstructs 1080x1080 output from 32x32 latent space.
    
    Uses transpose convolutional layers for progressive upsampling:
    32 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 1080
    """
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of decoder.
        
        Args:
            x: Latent tensor of shape (batch, 32, 32, 1)
            
        Returns:
            Reconstructed image of shape (batch, 1080, 1080, 1)
        """
        # Progressive upsampling with decreasing feature depth
        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        x = nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        x = nn.ConvTranspose(features=128, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        x = nn.ConvTranspose(features=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        x = nn.ConvTranspose(features=32, kernel_size=(7, 7), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        # Final layer to get single channel output
        x = nn.ConvTranspose(features=1, kernel_size=(7, 7), strides=(1, 1), padding='SAME')(x)
        
        # Resize to exact target size
        x = jax.image.resize(x, (x.shape[0], 1080, 1080, 1), method='bilinear')
        
        # Apply sigmoid for B/W output
        x = nn.sigmoid(x)
        
        return x


class CrossModalModel(nn.Module):
    """
    Complete cross-modal model combining encoder and decoder.
    
    This model can encode input images to latent space and decode 
    latent representations back to images.
    """
    
    def setup(self):
        """Initialize encoder and decoder components."""
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through encoder and decoder.
        
        Args:
            x: Input tensor of shape (batch, 1080, 1080, 1)
            
        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent
    
    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Encode input to latent space only.
        
        Args:
            x: Input tensor of shape (batch, 1080, 1080, 1)
            
        Returns:
            Latent representation of shape (batch, 32, 32, 1)
        """
        return self.encoder(x)
    
    def decode(self, latent: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent representation to output image.
        
        Args:
            latent: Latent tensor of shape (batch, 32, 32, 1)
            
        Returns:
            Reconstructed image of shape (batch, 1080, 1080, 1)
        """
        return self.decoder(latent)


class ObjectCounter(nn.Module):
    """
    Simple CNN for counting objects in images.
    Used for evaluation and debugging purposes.
    """
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Count objects in input image.
        
        Args:
            x: Input image of shape (batch, height, width, 1)
            
        Returns:
            Predicted object count of shape (batch, 1)
        """
        # Convolutional feature extraction
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Dense layers for count prediction
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        
        # Output single count value
        x = nn.Dense(features=1)(x)
        
        return x


def create_model_pair() -> Tuple[CrossModalModel, CrossModalModel]:
    """
    Create a pair of cross-modal models for the communication system.
    
    Returns:
        Tuple of (model1, model2) where:
        - model1: Processes circles -> latent -> squares (via model2)
        - model2: Processes squares -> latent -> circles (via model1)
    """
    model1 = CrossModalModel()  # Circle to square model
    model2 = CrossModalModel()  # Square to circle model
    
    return model1, model2


def get_model_info():
    """
    Print information about the model architecture.
    """
    print("Cross-Modal Communication System Architecture:")
    print("=" * 50)
    print("Model 1 (Circle-to-Square):")
    print("  - Encoder: 1080×1080×1 -> 32×32×1")
    print("  - Decoder: 32×32×1 -> 1080×1080×1")
    print("\nModel 2 (Square-to-Circle):")
    print("  - Encoder: 1080×1080×1 -> 32×32×1") 
    print("  - Decoder: 32×32×1 -> 1080×1080×1")
    print("\nCross-Modal Communication:")
    print("  - Model 1 Encoder -> Model 2 Decoder")
    print("  - Model 2 Encoder -> Model 1 Decoder")
    print("  - Objective: Communicate object counts through latent space")


if __name__ == "__main__":
    # Test model creation
    model1, model2 = create_model_pair()
    get_model_info()
    
    # Test with dummy input
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 1080, 1080, 1))
    
    # Initialize models
    params1 = model1.init(key, dummy_input)
    params2 = model2.init(key, dummy_input)
    
    print(f"\nModel 1 parameters: {sum(x.size for x in jax.tree_leaves(params1))}")
    print(f"Model 2 parameters: {sum(x.size for x in jax.tree_leaves(params2))}")