import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import random
import numpy as np
import matplotlib.pyplot as plt
from flax.training import train_state
import time
from sklearn.decomposition import PCA # Import PCA for visualization

# --- Configuration ---
IMAGE_SIZE = (32, 32) # Small image size for faster training
MAX_OBJECTS = 10     # Maximum number of objects in an image
NUM_TRAIN_IMAGES = 10000 # Reduced for faster simulation, can increase
NUM_TEST_IMAGES = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10      # Increased epochs slightly for better training
ABSTRACT_REPR_DIM = 32 # Dimension of the abstract representation vector
VIS_IMAGES_COUNT = 100 # Number of test images to use for visualization

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

# --- Prepare data for visualization ---
# Use a fixed subset of test images for consistent visualization across epochs
vis_images = test_images[:VIS_IMAGES_COUNT]
vis_counts = test_counts[:VIS_IMAGES_COUNT]

# --- Model Definitions (Flax CNN + Dense) ---

class Encoder(nn.Module):
    """ Encodes an image into an abstract representation vector. """
    abstract_repr_dim: int

    @nn.compact
    def __call__(self, x):
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

class Decoder(nn.Module):
    """ Decodes the abstract representation vector to predict the object count. """
    @nn.compact
    def __call__(self, x):
        # Dense layers
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        x = nn.Dense(features=32)(x)
        x = nn.relu(x)

        # Output layer: single neuron for regression (predicting count)
        x = nn.Dense(features=1)(x)
        return x.squeeze(-1) # Remove the last dimension of size 1


# --- Training Setup ---

# Instantiate Encoder and Decoder modules
encoder_module = Encoder(abstract_repr_dim=ABSTRACT_REPR_DIM)
decoder_module = Decoder()

# Initialize parameters for two sets of Encoder-Decoder pairs (for model A and model B)
key = jax.random.PRNGKey(0)
key_a_enc, key_a_dec, key_b_enc, key_b_dec = jax.random.split(key, 4)

dummy_image = jnp.ones((1, *IMAGE_SIZE, 1)) # Dummy input for encoder initialization
dummy_abstract_repr = jnp.ones((1, ABSTRACT_REPR_DIM)) # Dummy input for decoder initialization

# Initialize parameters for Model A's encoder and decoder
params_a_enc = encoder_module.init(key_a_enc, dummy_image)['params']
params_a_dec = decoder_module.init(key_a_dec, dummy_abstract_repr)['params']

# Initialize parameters for Model B's encoder and decoder
params_b_enc = encoder_module.init(key_b_enc, dummy_image)['params']
params_b_dec = decoder_module.init(key_b_dec, dummy_abstract_repr)['params']


# Combine parameters into a single dictionary for optimization
# Model A's parameters include its encoder and decoder parts
# Model B's parameters include its encoder and decoder parts
all_params = {
    'model_a': {'encoder': params_a_enc, 'decoder': params_a_dec},
    'model_b': {'encoder': params_b_enc, 'decoder': params_b_dec}
}

# Define the optimizer
optimizer = optax.adam(LEARNING_RATE)

# Create the training state
state = train_state.TrainState.create(
    apply_fn=None, # apply_fn is not used directly here as we call module.apply
    params=all_params,
    tx=optimizer
)

# Define the loss function (Mean Squared Error) considering both communication paths
def loss_fn(all_params, images, counts):
    params_a_enc = all_params['model_a']['encoder']
    params_a_dec = all_params['model_a']['decoder']
    params_b_enc = all_params['model_b']['encoder']
    params_b_dec = all_params['model_b']['decoder']

    # Path 1: Model A encodes, Model B decodes
    abstract_repr_a = encoder_module.apply({'params': params_a_enc}, images)
    predicted_count_b = decoder_module.apply({'params': params_b_dec}, abstract_repr_a)
    loss_ab = jnp.mean((predicted_count_b - counts)**2)

    # Path 2: Model B encodes, Model A decodes
    abstract_repr_b = encoder_module.apply({'params': params_b_enc}, images)
    predicted_count_a = decoder_module.apply({'params': params_a_dec}, abstract_repr_b)
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
    params_a_enc = all_params['model_a']['encoder']
    params_a_dec = all_params['model_a']['decoder']
    params_b_enc = all_params['model_b']['encoder']
    params_b_dec = all_params['model_b']['decoder']

    # Evaluate Path A->B
    abstract_repr_a = encoder_module.apply({'params': params_a_enc}, images)
    predicted_count_b = decoder_module.apply({'params': params_b_dec}, abstract_repr_a)
    loss_ab = jnp.mean((predicted_count_b - counts)**2)
    mae_ab = jnp.mean(jnp.abs(predicted_count_b - counts))

    # Evaluate Path B->A
    abstract_repr_b = encoder_module.apply({'params': params_b_enc}, images)
    predicted_count_a = decoder_module.apply({'params': params_a_dec}, abstract_repr_b)
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

# Store abstract representations for visualization across epochs
abstract_reprs_a_history = []
abstract_reprs_b_history = []


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

    # --- Record abstract representations for visualization ---
    params_a_enc = state.params['model_a']['encoder']
    params_b_enc = state.params['model_b']['encoder']

    # Compute abstract representations for the fixed visualization subset
    abstract_reprs_a = encoder_module.apply({'params': params_a_enc}, vis_images)
    abstract_reprs_b = encoder_module.apply({'params': params_b_enc}, vis_images)

    # Store the representations (convert to numpy for easier handling later)
    abstract_reprs_a_history.append(np.array(abstract_reprs_a))
    abstract_reprs_b_history.append(np.array(abstract_reprs_b))


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


# --- Abstract Representation Visualization ---

print("\nVisualizing Abstract Representations (PCA to 2D)...")

# Choose which epoch's representations to visualize (e.g., the last epoch)
epoch_to_visualize = NUM_EPOCHS - 1 # Index of the last epoch

abstract_reprs_a_final = abstract_reprs_a_history[epoch_to_visualize]
abstract_reprs_b_final = abstract_reprs_b_history[epoch_to_visualize]

# Apply PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)

# Fit and transform the representations for Model A's encoder
abstract_reprs_a_2d = pca.fit_transform(abstract_reprs_a_final)

# Fit and transform the representations for Model B's encoder
# Note: We fit PCA separately for each model's representations
pca_b = PCA(n_components=2)
abstract_reprs_b_2d = pca_b.fit_transform(abstract_reprs_b_final)


# Plot the 2D representations
plt.figure(figsize=(14, 6))

# Plot for Model A's Encoder
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
scatter_a = plt.scatter(abstract_reprs_a_2d[:, 0], abstract_reprs_a_2d[:, 1], c=vis_counts, cmap='viridis', alpha=0.7)
plt.colorbar(scatter_a, label='Object Count')
plt.title(f'Model A Encoder Abstract Repr (Epoch {epoch_to_visualize + 1}, PCA to 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)

# Plot for Model B's Encoder
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
scatter_b = plt.scatter(abstract_reprs_b_2d[:, 0], abstract_reprs_b_2d[:, 1], c=vis_counts, cmap='viridis', alpha=0.7)
plt.colorbar(scatter_b, label='Object Count')
plt.title(f'Model B Encoder Abstract Repr (Epoch {epoch_to_visualize + 1}, PCA to 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)

plt.tight_layout() # Adjust layout to prevent overlap
plt.show()

# --- Example predictions on a few test images ---
# (This part remains the same as before)
print("\nExample Predictions (A->B Path):")
params_a_enc = state.params['model_a']['encoder']
params_b_dec = state.params['model_b']['decoder']

abstract_reprs_a = encoder_module.apply({'params': params_a_enc}, test_images[:10])
predicted_counts_b = decoder_module.apply({'params': params_b_dec}, abstract_reprs_a)

for i in range(10):
    print(f"Image {i}: True Count = {test_counts[i]}, Predicted Count (A->B) = {predicted_counts_b[i]:.2f}")

print("\nExample Predictions (B->A Path):")
params_b_enc = state.params['model_b']['encoder']
params_a_dec = state.params['model_a']['decoder']

abstract_reprs_b = encoder_module.apply({'params': params_b_enc}, test_images[:10])
predicted_counts_a = decoder_module.apply({'params': params_a_dec}, abstract_reprs_b)

for i in range(10):
    print(f"Image {i}: True Count = {test_counts[i]}, Predicted Count (B->A) = {predicted_counts_a[i]:.2f}")

