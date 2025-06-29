# calculate_gradient.py

import jax
import jax.numpy as jnp

def scalar_function(matrix):
  """
  A simple function that takes a matrix and returns a scalar value.
  In this case, it's the sum of all the elements in the matrix.
  The gradient of this function with respect to the matrix should be a
  matrix of ones with the same shape.
  """
  return jnp.sum(matrix)

def scalar_function_squared(matrix):
  """
  Another example function: sum of the squares of the elements.
  The gradient of f(X) = sum(X^2) is 2*X.
  """
  return jnp.sum(matrix**2)

# 1. Define a sample input matrix.
sample_matrix = jnp.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]])

# 2. Use `jax.grad` to create a function that calculates the gradient.
#    `jax.grad` transforms a function into a new function that computes its gradient.
gradient_fn_simple = jax.grad(scalar_function)
gradient_fn_squared = jax.grad(scalar_function_squared)


# 3. Calculate the gradient for our sample matrix using the new functions.
gradient_matrix_simple = gradient_fn_simple(sample_matrix)
gradient_matrix_squared = gradient_fn_squared(sample_matrix)


# 4. Print the results.
print("--- Example 1: Gradient of sum(X) ---")
print("Original Matrix (X):")
print(sample_matrix)
print("\nFunction: f(X) = sum(X)")
print(f"Result of f(X): {scalar_function(sample_matrix)}")
print("\nCalculated Gradient (should be a matrix of ones):")
print(gradient_matrix_simple)

print("\n" + "="*40 + "\n")

print("--- Example 2: Gradient of sum(X^2) ---")
print("Original Matrix (X):")
print(sample_matrix)
print("\nFunction: f(X) = sum(X^2)")
print(f"Result of f(X): {scalar_function_squared(sample_matrix)}")
print("\nCalculated Gradient (should be 2*X):")
print(gradient_matrix_squared)
