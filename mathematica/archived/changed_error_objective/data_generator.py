# data_generator.py

import jax
import jax.numpy as jnp
import numpy as np
from config import IMG_SIZE, NUM_SHAPES_MIN, NUM_SHAPES_MAX, CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX, MAX_ATTEMPTS_PER_SHAPE, DTYPE

def draw_circle(image, center_x, center_y, radius, color=1):
    """Draws a filled circle on the image."""
    y, x = jnp.ogrid[-center_y:image.shape[0]-center_y, -center_x:image.shape[1]-center_x]
    mask = x*x + y*y <= radius*radius
    return jnp.where(mask, color, image)

def draw_square(image, top_left_x, top_left_y, side_length, color=1):
    """Draws a filled square on the image."""
    return image.at[top_left_y:top_left_y+side_length, top_left_x:top_left_x+side_length].set(color)

def generate_paired_images(rng_key):
    """
    Generates a pair of images with corresponding circles and squares.
    This function uses a Python loop and is not JIT-compiled.
    """
    # 1. Generate a set of non-overlapping CIRCLE parameters first.
    rng_key, num_shapes_key, placement_key = jax.random.split(rng_key, 3)
    num_shapes = jax.random.randint(num_shapes_key, (), NUM_SHAPES_MIN, NUM_SHAPES_MAX + 1).item()

    shape_params = [] # Python list of tuples (cx, cy, radius)
    subkeys = jax.random.split(placement_key, num_shapes)

    for i in range(num_shapes):
        subkey = subkeys[i]
        found_place = False
        for attempt in range(MAX_ATTEMPTS_PER_SHAPE):
            radius_key, x_key, y_key, subkey = jax.random.split(subkey, 4)
            radius = jax.random.randint(radius_key, (), CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX + 1).item()
            
            # Ensure the shape is fully within the image bounds
            center_x = jax.random.randint(x_key, (), radius, IMG_SIZE - radius).item()
            center_y = jax.random.randint(y_key, (), radius, IMG_SIZE - radius).item()
            
            # Check for overlap with existing shapes
            overlap = False
            for (ex, ey, er) in shape_params:
                dist_sq = (center_x - ex)**2 + (center_y - ey)**2
                min_dist_sq = (radius + er)**2
                if dist_sq < min_dist_sq:
                    overlap = True
                    break
            
            if not overlap:
                shape_params.append((center_x, center_y, radius))
                found_place = True
                break
    
    # 2. Create two blank images and draw the corresponding shapes.
    circle_image = jnp.zeros((IMG_SIZE, IMG_SIZE), dtype=DTYPE)
    square_image = jnp.zeros((IMG_SIZE, IMG_SIZE), dtype=DTYPE)

    for cx, cy, r in shape_params:
        circle_image = draw_circle(circle_image, cx, cy, r)
        
        side_length = 2 * r
        top_left_x = cx - r
        top_left_y = cy - r
        
        square_image = draw_square(square_image, top_left_x, top_left_y, side_length)

    return circle_image, square_image
