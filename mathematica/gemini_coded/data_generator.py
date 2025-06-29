# data_generator.py

import jax
import jax.numpy as jnp
from functools import partial
from config import IMG_SIZE, NUM_SHAPES_MIN, NUM_SHAPES_MAX, CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX, \
    SQUARE_SIDE_LENGTH_MIN, SQUARE_SIDE_LENGTH_MAX, MAX_ATTEMPTS_PER_SHAPE, DTYPE

def draw_circle(image, center_x, center_y, radius, color=1):
    """Draws a circle on the image."""
    y, x = jnp.ogrid[-center_y:image.shape[0]-center_y, -center_x:image.shape[1]-center_x]
    mask = x*x + y*y <= radius*radius
    return jnp.where(mask, color, image)

def draw_square(image, top_left_x, top_left_y, side_length, color=1):
    """Draws a square on the image."""
    x_coords, y_coords = jnp.meshgrid(
        jnp.arange(top_left_x, top_left_x + side_length),
        jnp.arange(top_left_y, top_left_y + side_length)
    )
    # Ensure coordinates are within image bounds
    x_coords = jnp.clip(x_coords, 0, image.shape[1] - 1)
    y_coords = jnp.clip(y_coords, 0, image.shape[0] - 1)

    mask = jnp.zeros_like(image, dtype=bool)
    mask = mask.at[y_coords.flatten(), x_coords.flatten()].set(True)
    return jnp.where(mask, color, image)

def check_overlap_circle(new_circle, existing_circles):
    """Checks if a new circle overlaps with any existing circles."""
    new_cx, new_cy, new_r = new_circle
    for cx, cy, r in existing_circles:
        dist_sq = (new_cx - cx)**2 + (new_cy - cy)**2
        min_dist_sq = (new_r + r)**2
        if dist_sq < min_dist_sq:
            return True
    return False

def check_overlap_square(new_square, existing_squares):
    """Checks if a new square overlaps with any existing squares."""
    new_x, new_y, new_s = new_square
    new_x2, new_y2 = new_x + new_s, new_y + new_s
    for x1, y1, s1 in existing_squares:
        x1_2, y1_2 = x1 + s1, y1 + s1
        # Check for overlap on x-axis
        overlap_x = max(new_x, x1) < min(new_x2, x1_2)
        # Check for overlap on y-axis
        overlap_y = max(new_y, y1) < min(new_y2, y1_2)
        if overlap_x and overlap_y:
            return True
    return False

def generate_non_overlapping_shapes(rng_key, shape_type, img_size):
    """
    Generates an image with non-overlapping shapes.
    shape_type: 'circle' or 'square'
    """
    img = jnp.zeros((img_size, img_size), dtype=DTYPE)
    shapes = []
    
    rng_key, num_shapes_key = jax.random.split(rng_key)
    num_shapes = jax.random.randint(num_shapes_key, (), NUM_SHAPES_MIN, NUM_SHAPES_MAX + 1)
    
    keys_for_shapes = jax.random.split(rng_key, num_shapes * 3) # For radius/side, x, y

    for i in range(num_shapes):
        shape_param_key = keys_for_shapes[3*i]
        pos_x_key = keys_for_shapes[3*i+1]
        pos_y_key = keys_for_shapes[3*i+2]

        found_place = False
        for attempt in range(MAX_ATTEMPTS_PER_SHAPE):
            if shape_type == 'circle':
                radius = jax.random.randint(shape_param_key, (), CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX + 1)
                center_x = jax.random.randint(pos_x_key, (), radius, img_size - radius + 1)
                center_y = jax.random.randint(pos_y_key, (), radius, img_size - radius + 1)
                new_shape = (center_x, center_y, radius)
                if not check_overlap_circle(new_shape, shapes):
                    shapes.append(new_shape)
                    found_place = True
                    break
            elif shape_type == 'square':
                side_length = jax.random.randint(shape_param_key, (), SQUARE_SIDE_LENGTH_MIN, SQUARE_SIDE_LENGTH_MAX + 1)
                top_left_x = jax.random.randint(pos_x_key, (), 0, img_size - side_length + 1)
                top_left_y = jax.random.randint(pos_y_key, (), 0, img_size - side_length + 1)
                new_shape = (top_left_x, top_left_y, side_length)
                if not check_overlap_square(new_shape, shapes):
                    shapes.append(new_shape)
                    found_place = True
                    break
        if not found_place:
            # print(f"Warning: Could not place shape {i+1} of {num_shapes}. Reducing number of actual shapes.")
            break # Stop trying to place more shapes if it's too hard

    final_num_shapes = len(shapes)
    drawn_img = jnp.zeros((img_size, img_size), dtype=DTYPE)

    if shape_type == 'circle':
        for cx, cy, r in shapes:
            drawn_img = draw_circle(drawn_img, cx, cy, r)
    elif shape_type == 'square':
        for x, y, s in shapes:
            drawn_img = draw_square(drawn_img, x, y, s)
            
    return drawn_img, final_num_shapes

# JIT compile these generation functions for performance
generate_non_overlapping_circles = jax.jit(partial(generate_non_overlapping_shapes, shape_type='circle', img_size=IMG_SIZE))
generate_non_overlapping_squares = jax.jit(partial(generate_non_overlapping_shapes, shape_type='square', img_size=IMG_SIZE))