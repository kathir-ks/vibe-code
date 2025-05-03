import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import random
import numpy as np
import matplotlib.pyplot as plt
from flax.training import train_state
import time

# --- Configuration ---
IMAGE_SIZE = (32, 32) # Small image size for faster training
MAX_OBJECTS = 10     # Maximum number of objects in an image
NUM_TRAIN_IMAGES = 10000 # Reduced for faster simulation, can increase
NUM_TEST_IMAGES = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10      # Increased epochs slightly for better training
ABSTRACT_REPR_DIM = 32 # Dimension of the abstract representation vector

# --- Data Generation ---

def generate_single_image(count, image_size):
    """
    Generates a single grayscale image with 'count' black squares on a white background.
    """
    img = np.ones(image_size, dtype=np.float32) # White background (1.0)
    obj_size = 2 # Size of the black squares

    for _ in range(count):
        # Randomly place a black square, ensure it's within bounds
        x = random.randint(0, image_size[0] - obj_size)
        y = random.randint(0, image_size[1] - obj_size)
        img[y:y+obj_size, x:x+obj_size] = 0.0 # Black (0.0)

    # Add a channel dimension for CNN input (Height, Width, Channels)
    return img[..., np.newaxis]

def generate_dataset(num_images, image_size, max_objects):
    """
    Generates a dataset of images and corresponding object counts.
    """
    images = []
    counts = []
    for _ in range(num_images):
        count = random.randint(0, max_objects)
        img = generate_single_image(count, image_size)
        images.append(img)
        counts.append(count)
    return np.array(images), np.array(counts, dtype=np.float32) # Counts as float for regression

print(f"Generating {NUM_TRAIN_IMAGES} training images and {NUM_TEST_IMAGES} test images...")
train_images, train_counts = generate_dataset(NUM_TRAIN_IMAGES, IMAGE_SIZE, MAX_OBJECTS)
test_images, test_counts = generate_dataset(NUM_TEST_IMAGES, IMAGE_SIZE, MAX_OBJECTS)
print("Data generation complete.")

# --- Unified Model Definition (Flax CNN + Dense) ---

class UnifiedModel(nn.Module):
    """
    A single model architecture capable of both encoding images and decoding counts.
    """
    abstract_repr_dim: int

    @nn.compact
    def encode(self, x):
        """ Encodes an image into an abstract representation vector. """
        # CNN layers
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten the output
        x = x.reshape((x.shape[0], -1))

        # Dense layer to produce the abstract representation
        x = nn.Dense(features=self.abstract_repr_dim)(x)
        x = nn.relu(x) # Keep ReLU for the abstract representation
        return x

    @nn.compact
    def decode(self, x):
        """ Decodes the abstract representation vector to predict the object count. """
        # Dense layers
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        x = nn.Dense(features=32)(x)
        x = nn.relu(x)

        # Output layer: single neuron for regression (predicting count)
        # No activation here, as we want a linear output for the count.
        x = nn.Dense(features=1)(x)
        return x.squeeze(-1) # Remove the last dimension of size 1

    # The __call__ method can be used for initialization purposes
    def __call__(self, x):
         # This method is primarily for initializing parameters based on input shape
         # We will use encode and decode methods directly in the training loop
         return self.encode(x) # Or self.decode(x) depending on intended init path


# --- Training Setup ---

# Initialize two instances of the UnifiedModel with different random keys
key = jax.random.PRNGKey(0)
key_a, key_b = jax.random.split(key)

model_a = UnifiedModel(abstract_repr_dim=ABSTRACT_REPR_DIM)
model_b = UnifiedModel(abstract_repr_dim=ABSTRACT_REPR_DIM)

# Initialize parameters for both models
dummy_image = jnp.ones((1, *IMAGE_SIZE, 1)) # Batch size of 1
params_a = model_a.init(key_a, dummy_image)['params']
params_b = model_b.init(key_b, dummy_image)['params']

# Combine parameters into a single dictionary for optimization
all_params = {'model_a': params_a, 'model_b': params_b}

# Define the optimizer
optimizer = optax.adam(LEARNING_RATE)

# Create the training state to hold parameters and optimizer state
state = train_state.TrainState.create(
    apply_fn=None, # apply_fn is not used directly here as we call model_a/b.apply
    params=all_params,
    tx=optimizer
)

# Define the loss function (Mean Squared Error) considering both communication paths
def loss_fn(all_params, images, counts):
    params_a = all_params['model_a']
    params_b = all_params['model_b']

    # Path 1: Model A encodes, Model B decodes
    abstract_repr_a = model_a.apply({'params': params_a}, images, method=model_a.encode)
    predicted_count_b = model_b.apply({'params': params_b}, abstract_repr_a, method=model_b.decode)
    loss_ab = jnp.mean((predicted_count_b - counts)**2)

    # Path 2: Model B encodes, Model A decodes
    abstract_repr_b = model_b.apply({'params': params_b}, images, method=model_b.encode)
    predicted_count_a = model_a.apply({'params': params_a}, abstract_repr_b, method=model_a.decode)
    loss_ba = jnp.mean((predicted_count_a - counts)**2)

    # Total loss is the sum of losses from both paths
    total_loss = loss_ab + loss_ba
    return total_loss

# Define the training step
@jax.jit
def train_step(state, images, counts):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, images, counts)
    state = state.apply_gradients(grads=grads)
    return state, loss, grads # Return grads for potential debugging/analysis

# Define the evaluation function
@jax.jit
def eval_fn(all_params, images, counts):
    params_a = all_params['model_a']
    params_b = all_params['model_b']

    # Evaluate Path A->B
    abstract_repr_a = model_a.apply({'params': params_a}, images, method=model_a.encode)
    predicted_count_b = model_b.apply({'params': params_b}, abstract_repr_a, method=model_b.decode)
    loss_ab = jnp.mean((predicted_count_b - counts)**2)
    mae_ab = jnp.mean(jnp.abs(predicted_count_b - counts))

    # Evaluate Path B->A
    abstract_repr_b = model_b.apply({'params': params_b}, images, method=model_b.encode)
    predicted_count_a = model_a.apply({'params': params_a}, abstract_repr_b, method=model_a.decode)
    loss_ba = jnp.mean((predicted_count_a - counts)**2)
    mae_ba = jnp.mean(jnp.abs(predicted_count_a - counts))

    # Report combined metrics
    total_loss = loss_ab + loss_ba
    avg_mae = (mae_ab + mae_ba) / 2.0

    return total_loss, avg_mae, loss_ab, loss_ba, mae_ab, mae_ba

# --- Training Loop ---

print("Starting training...")
train_losses = []
test_total_losses = []
test_avg_maes = []
test_loss_ab_history = []
test_loss_ba_history = []
test_mae_ab_history = []
test_mae_ba_history = []


for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    # Shuffle training data
    permutation = np.random.permutation(NUM_TRAIN_IMAGES)
    shuffled_train_images = train_images[permutation]
    shuffled_train_counts = train_counts[permutation]

    batch_loss = 0
    for i in range(0, NUM_TRAIN_IMAGES, BATCH_SIZE):
        images_batch = shuffled_train_images[i:i+BATCH_SIZE]
        counts_batch = shuffled_train_counts[i:i+BATCH_SIZE]
        state, loss, grads = train_step(state, images_batch, counts_batch) # Capture grads if needed
        batch_loss += loss

    avg_train_loss = batch_loss / (NUM_TRAIN_IMAGES / BATCH_SIZE)
    train_losses.append(avg_train_loss)

    # Evaluate on test data
    test_total_loss, test_avg_mae, test_loss_ab, test_loss_ba, test_mae_ab, test_mae_ba = eval_fn(state.params, test_images, test_counts)
    test_total_losses.append(test_total_loss)
    test_avg_maes.append(test_avg_mae)
    test_loss_ab_history.append(test_loss_ab)
    test_loss_ba_history.append(test_loss_ba)
    test_mae_ab_history.append(test_mae_ab)
    test_mae_ba_history.append(test_mae_ba)


    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Test Total Loss: {test_total_loss:.4f}, Test Avg MAE: {test_avg_mae:.4f}, Time: {epoch_time:.2f}s")
    print(f"  (A->B Loss: {test_loss_ab:.4f}, A->B MAE: {test_mae_ab:.4f} | B->A Loss: {test_loss_ba:.4f}, B->A MAE: {test_mae_ba:.4f})")


print("Training finished.")

# --- Visualization ---

# Plot training and test total loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Total Loss')
plt.plot(test_total_losses, label='Test Total Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Total Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plot Test Average MAE
plt.figure(figsize=(12, 6))
plt.plot(test_avg_maes, label='Test Average MAE', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Test Average Mean Absolute Error over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plot individual path losses
plt.figure(figsize=(12, 6))
plt.plot(test_loss_ab_history, label='Test Loss (A->B)')
plt.plot(test_loss_ba_history, label='Test Loss (B->A)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Test Loss for Individual Communication Paths')
plt.legend()
plt.grid(True)
plt.show()

# Plot individual path MAEs
plt.figure(figsize=(12, 6))
plt.plot(test_mae_ab_history, label='Test MAE (A->B)')
plt.plot(test_mae_ba_history, label='Test MAE (B->A)')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Test MAE for Individual Communication Paths')
plt.legend()
plt.grid(True)
plt.show()


# Example predictions on a few test images
print("\nExample Predictions (A->B Path):")
params_a = state.params['model_a']
params_b = state.params['model_b']

abstract_reprs_a = model_a.apply({'params': params_a}, test_images[:10], method=model_a.encode)
predicted_counts_b = model_b.apply({'params': params_b}, abstract_reprs_a, method=model_b.decode)

for i in range(10):
    print(f"Image {i}: True Count = {test_counts[i]}, Predicted Count (A->B) = {predicted_counts_b[i]:.2f}")

print("\nExample Predictions (B->A Path):")
abstract_reprs_b = model_b.apply({'params': params_b}, test_images[:10], method=model_b.encode)
predicted_counts_a = model_a.apply({'params': params_a}, abstract_reprs_b, method=model_a.decode)

for i in range(10):
    print(f"Image {i}: True Count = {test_counts[i]}, Predicted Count (B->A) = {predicted_counts_a[i]:.2f}")

