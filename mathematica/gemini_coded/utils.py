# utils.py

import numpy as np
from skimage.measure import label
from functools import partial
import jax

def count_objects(image_array, threshold=0.5):
    """
    Counts distinct objects (blobs) in a binary-like image.
    Uses skimage.measure.label, which is based on NumPy, not JAX.
    This function should be wrapped with jax.lax.stop_gradient when used in a JAX loss.
    """
    binary_img_np = (np.array(image_array) > threshold).astype(np.uint8)

    if np.sum(binary_img_np) == 0:
        return 0

    labeled_img = label(binary_img_np)
    num_objects = labeled_img.max()
    return num_objects

# JIT compile the stop_gradient version for usage in loss
# We don't want to JIT count_objects directly, as it uses numpy/skimage.
# Instead, we define a JAX-friendly wrapper.
@partial(jax.jit, static_argnums=(0,))
def jax_count_objects_stopped_gradient(image_array_jax, threshold=0.5):
    """
    Wrapper for count_objects that uses jax.lax.stop_gradient.
    This allows the value to be used in the JAX computation graph for loss,
    but prevents gradients from flowing back through the counting operation.
    """
    return jax.lax.stop_gradient(jax.pure_callback(
        partial(count_objects, threshold=threshold),
        jax.ShapeDtypeStruct((), jax.numpy.int32), # Output is a single integer
        image_array_jax
    ))