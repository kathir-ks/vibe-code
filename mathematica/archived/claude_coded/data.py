"""
Data generation and processing utilities for cross-modal object communication.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import ndimage


@dataclass
class DataConfig:
    """Configuration for data generation."""
    img_size: int = 1080
    min_objects: int = 10
    max_objects: int = 30
    circle_min_radius: int = 15
    circle_max_radius: int = 50
    square_min_size: int = 20
    square_max_size: int = 80
    overlap_buffer: int = 5
    max_placement_attempts: int = 1000


class ShapeGenerator:
    """Generates images with non-overlapping geometric shapes."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def generate_circles(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, int, jax.random.PRNGKey]:
        """
        Generate image with non-overlapping circles.
        
        Args:
            key: JAX random key
            
        Returns:
            Tuple of (image, num_circles, updated_key)
        """
        img = jnp.zeros((self.config.img_size, self.config.img_size))
        
        # Random number of circles
        key, subkey = jax.random.split(key)
        n_circles = jax.random.randint(
            subkey, (), self.config.min_objects, self.config.max_objects + 1
        )
        
        circles = []
        attempts = 0
        
        while len(circles) < n_circles and attempts < self.config.max_placement_attempts:
            key, subkey = jax.random.split(key)
            
            # Generate random radius
            radius = jax.random.randint(
                subkey, (), self.config.circle_min_radius, self.config.circle_max_radius + 1
            )
            
            key, subkey = jax.random.split(key)
            # Ensure circle fits within image bounds
            center_x = jax.random.randint(
                subkey, (), radius, self.config.img_size - radius
            )
            
            key, subkey = jax.random.split(key)
            center_y = jax.random.randint(
                subkey, (), radius, self.config.img_size - radius
            )
            
            # Check for overlap with existing circles
            overlap = False
            for existing_circle in circles:
                distance = jnp.sqrt(
                    (center_x - existing_circle[0])**2 + 
                    (center_y - existing_circle[1])**2
                )
                min_distance = radius + existing_circle[2] + self.config.overlap_buffer
                if distance < min_distance:
                    overlap = True
                    break
            
            if not overlap:
                circles.append((center_x, center_y, radius))
            
            attempts += 1
        
        # Draw circles on image
        img = self._draw_circles(img, circles)
        
        return img, len(circles), key
    
    def generate_squares(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, int, jax.random.PRNGKey]:
        """
        Generate image with non-overlapping squares.
        
        Args:
            key: JAX random key
            
        Returns:
            Tuple of (image, num_squares, updated_key)
        """
        img = jnp.zeros((self.config.img_size, self.config.img_size))
        
        # Random number of squares
        key, subkey = jax.random.split(key)
        n_squares = jax.random.randint(
            subkey, (), self.config.min_objects, self.config.max_objects + 1
        )
        
        squares = []
        attempts = 0
        
        while len(squares) < n_squares and attempts < self.config.max_placement_attempts:
            key, subkey = jax.random.split(key)
            
            # Generate random size
            size = jax.random.randint(
                subkey, (), self.config.square_min_size, self.config.square_max_size + 1
            )
            
            key, subkey = jax.random.split(key)
            # Ensure square fits within image bounds
            top_x = jax.random.randint(
                subkey, (), 0, self.config.img_size - size
            )
            
            key, subkey = jax.random.split(key)
            top_y = jax.random.randint(
                subkey, (), 0, self.config.img_size - size
            )
            
            # Check for overlap with existing squares
            overlap = False
            for existing_square in squares:
                if self._rectangles_overlap(
                    (top_x, top_y, size, size),
                    (existing_square[0], existing_square[1], existing_square[2], existing_square[2])
                ):
                    overlap = True
                    break
            
            if not overlap:
                squares.append((top_x, top_y, size))
            
            attempts += 1
        
        # Draw squares on image
        img = self._draw_squares(img, squares)
        
        return img, len(squares), key
    
    def _draw_circles(self, img: jnp.ndarray, circles: List[Tuple[int, int, int]]) -> jnp.ndarray:
        """Draw circles on image."""
        y, x = jnp.ogrid[:self.config.img_size, :self.config.img_size]
        
        for center_x, center_y, radius in circles:
            mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
            img = img.at[mask].set(1.0)
        
        return img
    
    def _draw_squares(self, img: jnp.ndarray, squares: List[Tuple[int, int, int]]) -> jnp.ndarray:
        """Draw squares on image."""
        for top_x, top_y, size in squares:
            img = img.at[top_y:top_y+size, top_x:top_x+size].set(1.0)
        return img
    
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], 
                           rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap."""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


class DataLoader:
    """Handles batch generation and data loading."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.generator = ShapeGenerator(config)
    
    def generate_batch(self, batch_size: int, key: jax.random.PRNGKey) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey
    ]:
        """
        Generate a batch of training data.
        
        Args:
            batch_size: Number of samples in batch
            key: JAX random key
            
        Returns:
            Tuple of (circle_images, square_images, circle_counts, square_counts, updated_key)
        """
        batch_circles = []
        batch_squares = []
        circle_counts = []
        square_counts = []
        
        for _ in range(batch_size):
            # Generate circle image
            key, subkey = jax.random.split(key)
            circle_img, n_circles, subkey = self.generator.generate_circles(subkey)
            batch_circles.append(circle_img)
            circle_counts.append(n_circles)
            
            # Generate square image
            key, subkey = jax.random.split(key)
            square_img, n_squares, subkey = self.generator.generate_squares(subkey)
            batch_squares.append(square_img)
            square_counts.append(n_squares)
        
        # Convert to JAX arrays and add channel dimension
        batch_circles = jnp.array(batch_circles)[..., None]
        batch_squares = jnp.array(batch_squares)[..., None]
        circle_counts = jnp.array(circle_counts)
        square_counts = jnp.array(square_counts)
        
        return batch_circles, batch_squares, circle_counts, square_counts, key
    
    def generate_dataset(self, num_samples: int, key: jax.random.PRNGKey) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """
        Generate a complete dataset.
        
        Args:
            num_samples: Total number of samples to generate
            key: JAX random key
            
        Returns:
            Tuple of (circle_images, square_images, circle_counts, square_counts)
        """
        all_circles = []
        all_squares = []
        all_circle_counts = []
        all_square_counts = []
        
        batch_size = 32  # Process in batches to avoid memory issues
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            circles, squares, c_counts, s_counts, key = self.generate_batch(
                current_batch_size, key
            )
            
            all_circles.append(circles)
            all_squares.append(squares)
            all_circle_counts.append(c_counts)
            all_square_counts.append(s_counts)
        
        # Concatenate all batches
        all_circles = jnp.concatenate(all_circles, axis=0)
        all_squares = jnp.concatenate(all_squares, axis=0)
        all_circle_counts = jnp.concatenate(all_circle_counts, axis=0)
        all_square_counts = jnp.concatenate(all_square_counts, axis=0)
        
        return all_circles, all_squares, all_circle_counts, all_square_counts


class ObjectCounter:
    """Utility class for counting objects in images."""
    
    @staticmethod
    def count_connected_components(img: jnp.ndarray, threshold: float = 0.5) -> int:
        """
        Count connected components (objects) in binary image.
        
        Args:
            img: Input image (height, width)
            threshold: Threshold for binarization
            
        Returns:
            Number of connected components
        """
        # Convert to numpy for scipy processing
        binary_img = (np.array(img) > threshold).astype(np.int32)
        
        # Use scipy's connected components algorithm
        labeled_img, num_objects = ndimage.label(binary_img)
        
        return num_objects
    
    @staticmethod
    def batch_count_objects(batch_imgs: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
        """
        Count objects in a batch of images.
        
        Args:
            batch_imgs: Batch of images (batch, height, width, channels)
            threshold: Threshold for binarization
            
        Returns:
            Array of object counts for each image
        """
        counts = []
        for i in range(batch_imgs.shape[0]):
            count = ObjectCounter.count_connected_components(
                batch_imgs[i, :, :, 0], threshold
            )
            counts.append(count)
        
        return jnp.array(counts)


class Visualizer:
    """Visualization utilities for the cross-modal system."""
    
    @staticmethod
    def plot_batch_samples(circles: jnp.ndarray, squares: jnp.ndarray, 
                          circle_counts: jnp.ndarray, square_counts: jnp.ndarray,
                          num_samples: int = 4):
        """
        Plot sample images from a batch.
        
        Args:
            circles: Batch of circle images
            squares: Batch of square images  
            circle_counts: Circle counts for each image
            square_counts: Square counts for each image
            num_samples: Number of samples to plot
        """
        fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
        
        for i in range(num_samples):
            # Plot circles
            axes[0, i].imshow(circles[i, :, :, 0], cmap='gray')
            axes[0, i].set_title(f'Circles: {circle_counts[i]}')
            axes[0, i].axis('off')
            
            # Plot squares
            axes[1, i].imshow(squares[i, :, :, 0], cmap='gray')
            axes[1, i].set_title(f'Squares: {square_counts[i]}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_cross_modal_results(original_circles: jnp.ndarray, original_squares: jnp.ndarray,
                                latent1: jnp.ndarray, latent2: jnp.ndarray,
                                reconstructed_squares: jnp.ndarray, reconstructed_circles: jnp.ndarray,
                                circle_counts: jnp.ndarray, square_counts: jnp.ndarray,
                                sample_idx: int = 0):
        """
        Plot results of cross-modal communication.
        
        Args:
            original_circles: Original circle images
            original_squares: Original square images
            latent1: Latent representations from model 1
            latent2: Latent representations from model 2
            reconstructed_squares: Squares reconstructed by model 2
            reconstructed_circles: Circles reconstructed by model 1
            circle_counts: True circle counts
            square_counts: True square counts
            sample_idx: Index of sample to visualize
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Model 1 pipeline: Circles -> Latent -> Squares
        axes[0, 0].imshow(original_circles[sample_idx, :, :, 0], cmap='gray')
        axes[0, 0].set_title(f'Original Circles ({circle_counts[sample_idx]})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(latent1[sample_idx, :, :, 0], cmap='gray')
        axes[0, 1].set_title('Latent Space (Model 1)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(reconstructed_squares[sample_idx, :, :, 0], cmap='gray')
        axes[0, 2].set_title('Reconstructed Squares')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(original_squares[sample_idx, :, :, 0], cmap='gray')
        axes[0, 3].set_title(f'Target Squares ({square_counts[sample_idx]})')
        axes[0, 3].axis('off')
        
        # Model 2 pipeline: Squares -> Latent -> Circles
        axes[1, 0].imshow(original_squares[sample_idx, :, :, 0], cmap='gray')
        axes[1, 0].set_title(f'Original Squares ({square_counts[sample_idx]})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(latent2[sample_idx, :, :, 0], cmap='gray')
        axes[1, 1].set_title('Latent Space (Model 2)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(reconstructed_circles[sample_idx, :, :, 0], cmap='gray')
        axes[1, 2].set_title('Reconstructed Circles')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(original_circles[sample_idx, :, :, 0], cmap='gray')
        axes[1, 3].set_title(f'Target Circles ({circle_counts[sample_idx]})')
        axes[1, 3].axis('off')
        
        plt.suptitle('Cross-Modal Communication Results', fontsize=16)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Test data generation
    config = DataConfig()
    data_loader = DataLoader(config)
    
    # Generate sample batch
    key = jax.random.PRNGKey(42)
    circles, squares, c_counts, s_counts, _ = data_loader.generate_batch(4, key)
    
    print(f"Generated batch with shapes:")
    print(f"Circles: {circles.shape}, Counts: {c_counts}")
    print(f"Squares: {squares.shape}, Counts: {s_counts}")
    
    # Visualize samples
    Visualizer.plot_batch_samples(circles, squares, c_counts, s_counts)
    
    # Test object counting
    predicted_circle_counts = ObjectCounter.batch_count_objects(circles)
    predicted_square_counts = ObjectCounter.batch_count_objects(squares)
    
    print(f"True circle counts: {c_counts}")
    print(f"Predicted circle counts: {predicted_circle_counts}")
    print(f"True square counts: {s_counts}")
    print(f"Predicted square counts: {predicted_square_counts}")