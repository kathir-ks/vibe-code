# data.py — Data generation: circles and triangles with shared counts

import jax
import jax.numpy as jnp
import numpy as np
from config import (
    IMG_SIZE, NUM_SHAPES_MIN, NUM_SHAPES_MAX,
    CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX,
    TRIANGLE_SIDE_MIN, TRIANGLE_SIDE_MAX,
    MAX_PLACEMENT_ATTEMPTS, DTYPE,
)


def _draw_filled_circle(canvas, cx, cy, r):
    """Draw a filled circle on a numpy canvas."""
    yy, xx = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    canvas[mask] = 1.0


def _draw_filled_triangle(canvas, cx, cy, side):
    """Draw a filled equilateral triangle (pointing up) centered at (cx, cy)."""
    h = int(side * np.sqrt(3) / 2)
    half_s = side // 2

    # Vertices of equilateral triangle centered at (cx, cy)
    # Top vertex, bottom-left, bottom-right
    top = (cx, cy - 2 * h // 3)
    bl = (cx - half_s, cy + h // 3)
    br = (cx + half_s, cy + h // 3)

    rows, cols = canvas.shape
    yy, xx = np.mgrid[:rows, :cols]

    # Point-in-triangle using barycentric / edge function method
    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = sign(xx, yy, top[0], top[1], bl[0], bl[1])
    d2 = sign(xx, yy, bl[0], bl[1], br[0], br[1])
    d3 = sign(xx, yy, br[0], br[1], top[0], top[1])

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    inside = ~(has_neg & has_pos)

    canvas[inside] = 1.0


def _circles_overlap(c1, c2):
    cx1, cy1, r1 = c1
    cx2, cy2, r2 = c2
    dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    return dist_sq < (r1 + r2) ** 2


def _bboxes_overlap(b1, b2):
    """Check if two axis-aligned bounding boxes overlap. Each is (x, y, w, h)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def generate_circle_image(rng_key, img_size=IMG_SIZE):
    """Generate a binary image with N non-overlapping circles.

    Returns: (image as jnp array [H, W, 1], count N)
    """
    canvas = np.zeros((img_size, img_size), dtype=np.float32)

    key_n, key_place = jax.random.split(rng_key)
    n = int(jax.random.randint(key_n, (), NUM_SHAPES_MIN, NUM_SHAPES_MAX + 1))

    placed = []  # list of (cx, cy, r)
    subkeys = jax.random.split(key_place, n * MAX_PLACEMENT_ATTEMPTS + 1)
    ki = 0

    for _ in range(n):
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            if ki >= len(subkeys):
                break
            k = subkeys[ki]; ki += 1
            kr, kx, ky = jax.random.split(k, 3)
            r = int(jax.random.randint(kr, (), CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX + 1))
            cx = int(jax.random.randint(kx, (), r, img_size - r))
            cy = int(jax.random.randint(ky, (), r, img_size - r))

            if not any(_circles_overlap((cx, cy, r), p) for p in placed):
                _draw_filled_circle(canvas, cx, cy, r)
                placed.append((cx, cy, r))
                break

    return jnp.array(canvas).reshape(img_size, img_size, 1), len(placed)


def generate_triangle_image(rng_key, img_size=IMG_SIZE):
    """Generate a binary image with N non-overlapping equilateral triangles.

    Returns: (image as jnp array [H, W, 1], count N)
    """
    canvas = np.zeros((img_size, img_size), dtype=np.float32)

    key_n, key_place = jax.random.split(rng_key)
    n = int(jax.random.randint(key_n, (), NUM_SHAPES_MIN, NUM_SHAPES_MAX + 1))

    placed_bboxes = []  # list of (x, y, w, h) bounding boxes
    subkeys = jax.random.split(key_place, n * MAX_PLACEMENT_ATTEMPTS + 1)
    ki = 0

    for _ in range(n):
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            if ki >= len(subkeys):
                break
            k = subkeys[ki]; ki += 1
            ks, kx, ky = jax.random.split(k, 3)
            side = int(jax.random.randint(ks, (), TRIANGLE_SIDE_MIN, TRIANGLE_SIDE_MAX + 1))
            h = int(side * np.sqrt(3) / 2)
            half_s = side // 2

            # Triangle bounding box: width = side, height = h
            margin_x = half_s + 2
            margin_y_top = 2 * h // 3 + 2
            margin_y_bot = h // 3 + 2

            if margin_x >= img_size - margin_x or margin_y_top >= img_size - margin_y_bot:
                continue

            cx = int(jax.random.randint(kx, (), margin_x, img_size - margin_x))
            cy = int(jax.random.randint(ky, (), margin_y_top, img_size - margin_y_bot))

            bbox = (cx - half_s, cy - 2 * h // 3, side, h)
            if not any(_bboxes_overlap(bbox, pb) for pb in placed_bboxes):
                _draw_filled_triangle(canvas, cx, cy, side)
                placed_bboxes.append(bbox)
                break

    return jnp.array(canvas).reshape(img_size, img_size, 1), len(placed_bboxes)


def generate_training_pair(rng_key, img_size=IMG_SIZE):
    """Generate a (circle_image, triangle_image) pair with the SAME count N.

    Positions and sizes are independent between the two images.
    Returns: (circle_img, triangle_img, count_N)
    """
    key_n, key_c, key_t = jax.random.split(rng_key, 3)
    target_n = int(jax.random.randint(key_n, (), NUM_SHAPES_MIN, NUM_SHAPES_MAX + 1))

    circle_img, c_count = _generate_with_target_count(key_c, target_n, 'circle', img_size)
    triangle_img, t_count = _generate_with_target_count(key_t, target_n, 'triangle', img_size)

    return circle_img, triangle_img, target_n


def _generate_with_target_count(rng_key, target_n, shape_type, img_size):
    """Generate an image trying to place exactly target_n shapes."""
    canvas = np.zeros((img_size, img_size), dtype=np.float32)

    subkeys = jax.random.split(rng_key, target_n * MAX_PLACEMENT_ATTEMPTS + 1)
    ki = 0
    count = 0

    if shape_type == 'circle':
        placed = []
        for _ in range(target_n):
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                if ki >= len(subkeys):
                    break
                k = subkeys[ki]; ki += 1
                kr, kx, ky = jax.random.split(k, 3)
                r = int(jax.random.randint(kr, (), CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX + 1))
                if r >= img_size - r:
                    continue
                cx = int(jax.random.randint(kx, (), r, img_size - r))
                cy = int(jax.random.randint(ky, (), r, img_size - r))
                if not any(_circles_overlap((cx, cy, r), p) for p in placed):
                    _draw_filled_circle(canvas, cx, cy, r)
                    placed.append((cx, cy, r))
                    count += 1
                    break
    else:  # triangle
        placed_bboxes = []
        for _ in range(target_n):
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                if ki >= len(subkeys):
                    break
                k = subkeys[ki]; ki += 1
                ks, kx, ky = jax.random.split(k, 3)
                side = int(jax.random.randint(ks, (), TRIANGLE_SIDE_MIN, TRIANGLE_SIDE_MAX + 1))
                h = int(side * np.sqrt(3) / 2)
                half_s = side // 2
                margin_x = half_s + 2
                margin_y_top = 2 * h // 3 + 2
                margin_y_bot = h // 3 + 2
                if margin_x >= img_size - margin_x or margin_y_top >= img_size - margin_y_bot:
                    continue
                cx = int(jax.random.randint(kx, (), margin_x, img_size - margin_x))
                cy = int(jax.random.randint(ky, (), margin_y_top, img_size - margin_y_bot))
                bbox = (cx - half_s, cy - 2 * h // 3, side, h)
                if not any(_bboxes_overlap(bbox, pb) for pb in placed_bboxes):
                    _draw_filled_triangle(canvas, cx, cy, side)
                    placed_bboxes.append(bbox)
                    count += 1
                    break

    return jnp.array(canvas).reshape(img_size, img_size, 1), count


def generate_training_batch(rng_key, num_devices, img_size=IMG_SIZE):
    """Generate num_devices training pairs, stacked for pmap.

    Returns:
        circle_batch:   jnp array [num_devices, 1, H, W, 1]
        triangle_batch: jnp array [num_devices, 1, H, W, 1]
        counts:         list of int (for logging only)
    """
    subkeys = jax.random.split(rng_key, num_devices)
    circles, triangles, counts = [], [], []
    for i in range(num_devices):
        c_img, t_img, n = generate_training_pair(subkeys[i], img_size)
        circles.append(c_img[None, ...])       # [1, H, W, 1]
        triangles.append(t_img[None, ...])     # [1, H, W, 1]
        counts.append(n)

    circle_batch = jnp.stack(circles, axis=0)      # [num_devices, 1, H, W, 1]
    triangle_batch = jnp.stack(triangles, axis=0)   # [num_devices, 1, H, W, 1]
    return circle_batch, triangle_batch, counts
