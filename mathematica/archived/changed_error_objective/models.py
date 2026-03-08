# models.py

import flax.linen as nn
import jax.numpy as jnp
from config import IMG_SIZE, LATENT_SIZE

class Encoder(nn.Module):
    latent_dim: int = LATENT_SIZE

    @nn.compact
    def __call__(self, x):
        # Input: 1080x1080x1
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 540x540
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 270x270
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 135x135
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 68x68
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 34x34
        x = nn.relu(x)
        x = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x) # Output 32x32x1
        return nn.sigmoid(x) # Constrain latent space to 0-1

class Decoder(nn.Module):
    output_dim: int = IMG_SIZE

    @nn.compact
    def __call__(self, x):
        # Input: 32x32x1
        x = nn.ConvTranspose(features=512, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x) # 34x34
        x = nn.relu(x)
        x = nn.ConvTranspose(features=512, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 68x68
        x = nn.relu(x)
        
        # This layer's upsampling from 68x68 would create a 136x136 image
        x = nn.ConvTranspose(features=256, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        
        # FIX: Manually crop the output to the correct dimension (135x135)
        x = x[:, :135, :135, :]
        
        x = nn.relu(x)
        
        # Subsequent layers now produce the correct dimensions
        x = nn.ConvTranspose(features=128, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 270x270
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 540x540
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 1080x1080
        
        # Return raw logits for numerical stability with the loss function
        return x
