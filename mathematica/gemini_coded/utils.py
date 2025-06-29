import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from config import DTYPE

def count_objects(image_array_np):
    """
    Counts distinct objects in a binary image using NumPy/SciPy.
    Expected image_array_np: 2D numpy array (binary or float where 0/1 is clear).
    This function operates purely on NumPy arrays.
    """
    # Ensure it's a NumPy array (it should be, as pure_callback converts JAX arrays)
    if not isinstance(image_array_np, np.ndarray):
        image_array_np = np.asarray(image_array_np)

    # Threshold and convert to uint8 binary image
    binary_image = (image_array_np > 0.5).astype(np.uint8)
    
    # Define an 8-connectivity structure for 2D images
    s = generate_binary_structure(2, 2)
    
    # Label connected components and count them
    labeled_array, num_features = label(binary_image, structure=s)
    return num_features

def jax_count_objects_stopped_gradient(jax_image_array):
    """
    Wraps the NumPy object counting function for use in JAX.
    It takes a JAX array, passes it to the Python function, and returns a JAX array.
    The gradient through this operation is stopped implicitly.
    """
    # Define the shape and dtype of the output that 'count_objects' returns.
    # It returns a single integer.
    # --- FIXED: Use jax.ShapeDtypeStruct instead of jax.ShapedArray ---
    result_shape_dtype = jax.ShapeDtypeStruct((), dtype=jnp.int32) 
    
    # Call pure_callback. The first argument is the Python function.
    # The second argument is the expected shape and dtype of the result.
    # Subsequent arguments are the *dynamic* JAX arrays to pass to the Python function.
    count = jax.pure_callback(
        count_objects, 
        result_shape_dtype, 
        jax_image_array, # This is the dynamic JAX array input
    )
    
    return count
