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
NUM_EPOCHS = 5       # Reduced epochs for faster simulation, can increase
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

# --- Model Definitions (Flax CNNs) ---

class ImageEncoder(nn.Module):
    """
    Model A: Encodes an image into an abstract representation vector.
    """
    abstract_repr_dim: int

    @nn.compact
    def __call__(self, x):
        # Simple CNN layers
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten the output for the dense layer
        x = x.reshape((x.shape[0], -1))

        # Dense layer to produce the abstract representation
        x = nn.Dense(features=self.abstract_repr_dim)(x)
        # Using ReLU here, but could experiment with others or no activation
        x = nn.relu(x)
        return x

class CountDecoder(nn.Module):
    """
    Model B: Decodes the abstract representation vector to predict the object count.
    """
    @nn.compact
    def __call__(self, x):
        # Dense layers to process the abstract representation
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        x = nn.Dense(features=32)(x)
        x = nn.relu(x)

        # Output layer: single neuron for regression (predicting count)
        # No activation here, as we want a linear output for the count.
        x = nn.Dense(features=1)(x)
        return x.squeeze(-1) # Remove the last dimension of size 1

# --- Training Setup ---

# Combine the two models into a single forward pass for training
class CommunicationModel(nn.Module):
    abstract_repr_dim: int

    @nn.compact
    def __call__(self, image):
        encoder = ImageEncoder(abstract_repr_dim=self.abstract_repr_dim)
        decoder = CountDecoder()

        abstract_repr = encoder(image)
        predicted_count = decoder(abstract_repr)
        return predicted_count

# Initialize the model and parameters
key = jax.random.PRNGKey(0)
dummy_image = jnp.ones((1, *IMAGE_SIZE, 1)) # Batch size of 1
model = CommunicationModel(abstract_repr_dim=ABSTRACT_REPR_DIM)
params = model.init(key, dummy_image)['params']

# Define the optimizer
optimizer = optax.adam(LEARNING_RATE)

# Create the training state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

# Define the loss function (Mean Squared Error)
def loss_fn(params, images, counts):
    predicted_counts = model.apply({'params': params}, images)
    # Ensure predicted_counts and counts have the same shape for MSE
    loss = jnp.mean((predicted_counts - counts)**2)
    return loss

# Define the training step
@jax.jit
def train_step(state, images, counts):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, images, counts)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Define the evaluation function
@jax.jit
def eval_fn(params, images, counts):
    predicted_counts = model.apply({'params': params}, images)
    loss = jnp.mean((predicted_counts - counts)**2)
    # Calculate Mean Absolute Error for a more interpretable metric
    mae = jnp.mean(jnp.abs(predicted_counts - counts))
    return loss, mae

# --- Training Loop ---

print("Starting training...")
train_losses = []
test_losses = []
test_maes = []

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
        state, loss = train_step(state, images_batch, counts_batch)
        batch_loss += loss

    avg_train_loss = batch_loss / (NUM_TRAIN_IMAGES / BATCH_SIZE)
    train_losses.append(avg_train_loss)

    # Evaluate on test data
    test_loss, test_mae = eval_fn(state.params, test_images, test_counts)
    test_losses.append(test_loss)
    test_maes.append(test_mae)

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Time: {epoch_time:.2f}s")

print("Training finished.")

# --- Visualization (Optional) ---

# Plot training and test loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plot Test MAE
plt.figure(figsize=(10, 5))
plt.plot(test_maes, label='Test MAE', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Test Mean Absolute Error over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Example predictions on a few test images
print("\nExample Predictions:")
predicted_counts = model.apply({'params': state.params}, test_images[:10])
for i in range(10):
    print(f"Image {i}: True Count = {test_counts[i]}, Predicted Count = {predicted_counts[i]:.2f}")

# --- Abstract Representation Visualization (Conceptual) ---
# To truly visualize the abstract representation, you would need dimensionality reduction
# techniques like PCA or t-SNE applied to the output of the encoder for a set of images.
# This is more complex and not included in this basic simulation.
# However, the idea is that the encoder learns to embed images with the same count
# closer together in the abstract representation space, allowing the decoder to
# easily map these representations to the correct count.

