# models.py

import flax.linen as nn
import jax.numpy as jnp
from config import IMG_SIZE, LATENT_SIZE

class Encoder(nn.Module):
    latent_dim: int = LATENT_SIZE

    @nn.compact
    def __call__(self, x):
        # Input: 1080x1080x1 (assuming grayscale input)
        # We need a series of convolutions to reduce 1080 to 32
        # Target downsampling factor = 1080 / 32 = 33.75
        # Use strides of 2 for each conv layer
        # 1080 -> 540 -> 270 -> 135 -> ~68 -> ~34 -> 32

        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 540x540
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 270x270
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 135x135
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 68x68 (135/2 = 67.5 -> 68 with SAME)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 34x34 (68/2 = 34)
        x = nn.relu(x)
        
        # From 34x34 to 32x32: use a 3x3 conv with stride 1 and VALID padding
        # (Input_dim - Kernel_size) + 1 = Output_dim
        # (34 - 3) + 1 = 32
        x = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x) # Output 32x32x1
        
        return nn.sigmoid(x) # Constrain to 0-1 for B/W latent space

class Decoder(nn.Module):
    output_dim: int = IMG_SIZE

    @nn.compact
    def __call__(self, x):
        # Input: 32x32x1 (latent space)
        # We need to reverse the encoder steps to reach 1080x1080
        # Start by padding to get to 34x34 (reverse of VALID padding)
        # The padding argument in ConvTranspose is for the *output* padding,
        # not input padding. It can be tricky.
        # Let's consider the input as 32x32.
        # To get 34x34 from 32x32 using ConvTranspose, we need to consider output shape.
        # Output shape = (Input_dim - 1) * Stride + Kernel_size - 2 * Padding + Output_padding
        # For VALID padding in Conv, it means no padding, so for ConvTranspose, the equivalent is
        # a kernel that expands.

        # From 32x32 to 34x34:
        # A 3x3 ConvTranspose with stride 1 will increase by (kernel_size - 1) if no output_padding
        # 32 -> 32 + (3-1) = 34
        x = nn.ConvTranspose(features=512, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x) # 34x34
        x = nn.relu(x)

        # Reverse the Encoder steps
        x = nn.ConvTranspose(features=512, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 68x68
        x = nn.relu(x)
        x = nn.ConvTranspose(features=256, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 135x135
        x = nn.relu(x)
        x = nn.ConvTranspose(features=128, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 270x270
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 540x540
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x) # 1080x1080
        
        return nn.sigmoid(x) # Output B/W image