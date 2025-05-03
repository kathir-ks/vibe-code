import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
from functools import partial
import os
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
key = jax.random.PRNGKey(42)

# Constants
IMG_SIZE = 64
MAX_OBJECTS = 10
NUM_TRAIN_IMAGES = 100000
NUM_TEST_IMAGES = 1000
BATCH_SIZE = 64
EMBEDDING_DIM = 32  # Size of the abstract representation
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

# Image generation function
def generate_random_image(key, img_size=IMG_SIZE, max_objects=MAX_OBJECTS):
    """Generate a random image with N objects (1 to max_objects)"""
    key, subkey = jax.random.split(key)
    num_objects = jax.random.randint(subkey, (1,), 1, max_objects + 1)[0]
    
    # Create a white background image
    image = np.ones((img_size, img_size), dtype=np.float32)
    
    # Add black objects (circles)
    for _ in range(num_objects):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        # Random position and size
        x = jax.random.randint(subkey1, (1,), 5, img_size - 5)[0]
        y = jax.random.randint(subkey2, (1,), 5, img_size - 5)[0]
        radius = jax.random.randint(subkey3, (1,), 3, 6)[0]
        
        # Create a temporary image with PIL for drawing the circle
        temp_img = Image.fromarray((image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(temp_img)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=0)
        image = np.array(temp_img).astype(np.float32) / 255.0
    
    return image, num_objects

# Generate dataset
def generate_dataset(key, num_images, img_size=IMG_SIZE, max_objects=MAX_OBJECTS):
    images = []
    counts = []
    
    for i in range(num_images):
        key, subkey = jax.random.split(key)
        img, count = generate_random_image(subkey, img_size, max_objects)
        images.append(img)
        counts.append(count)
    
    # Convert to JAX arrays
    images = jnp.array(images).reshape(-1, img_size, img_size, 1)  # Add channel dimension
    counts = jnp.array(counts)
    
    return images, counts

# Define the CNN model for image analysis
class ImageEncoder(nn.Module):
    """CNN model that encodes an image into an abstract representation"""
    embedding_dim: int = EMBEDDING_DIM
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not train)(x)
        
        # Output abstract representation
        embedding = nn.Dense(features=self.embedding_dim)(x)
        return embedding

# Define the decoder model for interpreting abstract representations
class AbstractDecoder(nn.Module):
    """Model that decodes abstract representation into object count"""
    max_objects: int = MAX_OBJECTS
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not train)(x)
        
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        
        # Output a probability distribution over possible counts (0 to max_objects)
        logits = nn.Dense(features=self.max_objects + 1)(x)
        return logits

# Combined model for each agent
class CountingAgent(nn.Module):
    """Combined model with encoder and decoder"""
    embedding_dim: int = EMBEDDING_DIM
    max_objects: int = MAX_OBJECTS
    
    def setup(self):
        self.encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.decoder = AbstractDecoder(max_objects=self.max_objects)
    
    def __call__(self, image, train: bool = True):
        # Encode image to abstract representation
        embedding = self.encoder(image, train=train)
        # Decode abstract representation to count
        logits = self.decoder(embedding, train=train)
        return logits, embedding
    
    def decode_embedding(self, embedding, train: bool = False):
        # Just use the decoder part
        return self.decoder(embedding, train=train)

# Create TrainState for model training
def create_train_state(model, key, learning_rate=LEARNING_RATE):
    """Creates an initial `TrainState` for training"""
    params = model.init(key, jnp.ones([1, IMG_SIZE, IMG_SIZE, 1]))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

# Loss functions
def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=MAX_OBJECTS + 1)
    return optax.softmax_cross_entropy(logits, one_hot).mean()

# Training step
@jax.jit
def train_step(state_a, state_b, images, labels):
    """Train step for both models with communication"""
    
    def loss_fn(params_a, params_b, images, labels):
        # Forward pass through model A
        logits_a, embeddings_a = state_a.apply_fn({'params': params_a}, images, train=True)
        loss_a = cross_entropy_loss(logits_a, labels)
        
        # Forward pass through model B, using embeddings from A
        logits_b_from_a = state_b.apply_fn({'params': params_b}, embeddings_a, train=True, method=CountingAgent.decode_embedding)
        loss_b_from_a = cross_entropy_loss(logits_b_from_a, labels)
        
        # Forward pass through model B
        logits_b, embeddings_b = state_b.apply_fn({'params': params_b}, images, train=True)
        loss_b = cross_entropy_loss(logits_b, labels)
        
        # Forward pass through model A, using embeddings from B
        logits_a_from_b = state_a.apply_fn({'params': params_a}, embeddings_b, train=True, method=CountingAgent.decode_embedding)
        loss_a_from_b = cross_entropy_loss(logits_a_from_b, labels)
        
        # Total loss
        total_loss_a = loss_a + loss_a_from_b
        total_loss_b = loss_b + loss_b_from_a
        
        return total_loss_a + total_loss_b, (total_loss_a, total_loss_b)
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    (total_loss, (loss_a, loss_b)), (grads_a, grads_b) = grad_fn(
        state_a.params, state_b.params, images, labels)
    
    # Update parameters
    state_a = state_a.apply_gradients(grads=grads_a)
    state_b = state_b.apply_gradients(grads=grads_b)
    
    return state_a, state_b, total_loss, loss_a, loss_b

# Evaluation function
@jax.jit
def evaluate(state_a, state_b, images, labels):
    """Evaluate both models with communication"""
    # Direct predictions
    logits_a, embeddings_a = state_a.apply_fn({'params': state_a.params}, images, train=False)
    logits_b, embeddings_b = state_b.apply_fn({'params': state_b.params}, images, train=False)
    
    # Cross predictions
    logits_b_from_a = state_b.apply_fn({'params': state_b.params}, embeddings_a, train=False, method=CountingAgent.decode_embedding)
    logits_a_from_b = state_a.apply_fn({'params': state_a.params}, embeddings_b, train=False, method=CountingAgent.decode_embedding)
    
    # Calculate accuracy
    pred_a = jnp.argmax(logits_a, axis=1)
    pred_b = jnp.argmax(logits_b, axis=1)
    pred_b_from_a = jnp.argmax(logits_b_from_a, axis=1)
    pred_a_from_b = jnp.argmax(logits_a_from_b, axis=1)
    
    acc_a = jnp.mean(pred_a == labels)
    acc_b = jnp.mean(pred_b == labels)
    acc_b_from_a = jnp.mean(pred_b_from_a == labels)
    acc_a_from_b = jnp.mean(pred_a_from_b == labels)
    
    # Calculate loss
    loss_a = cross_entropy_loss(logits_a, labels)
    loss_b = cross_entropy_loss(logits_b, labels)
    loss_b_from_a = cross_entropy_loss(logits_b_from_a, labels)
    loss_a_from_b = cross_entropy_loss(logits_a_from_b, labels)
    
    return {
        'acc_a': acc_a, 
        'acc_b': acc_b,
        'acc_b_from_a': acc_b_from_a,
        'acc_a_from_b': acc_a_from_b,
        'loss_a': loss_a,
        'loss_b': loss_b,
        'loss_b_from_a': loss_b_from_a,
        'loss_a_from_b': loss_a_from_b
    }

# Main training function
def train_models():
    # Generate datasets
    key, subkey1, subkey2 = jax.random.split(key, 3)
    train_images, train_counts = generate_dataset(subkey1, NUM_TRAIN_IMAGES)
    test_images, test_counts = generate_dataset(subkey2, NUM_TEST_IMAGES)
    
    # Initialize models
    model_a = CountingAgent(embedding_dim=EMBEDDING_DIM, max_objects=MAX_OBJECTS)
    model_b = CountingAgent(embedding_dim=EMBEDDING_DIM, max_objects=MAX_OBJECTS)
    
    # Create train states
    key, subkey1, subkey2 = jax.random.split(key, 3)
    state_a = create_train_state(model_a, subkey1)
    state_b = create_train_state(model_b, subkey2)
    
    # Training loop
    steps_per_epoch = NUM_TRAIN_IMAGES // BATCH_SIZE
    
    for epoch in range(NUM_EPOCHS):
        # Shuffle training data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, train_images.shape[0])
        train_images_shuffled = train_images[perm]
        train_counts_shuffled = train_counts[perm]
        
        # Train for one epoch
        epoch_loss = 0.0
        epoch_loss_a = 0.0
        epoch_loss_b = 0.0
        
        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            batch_start = step * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, train_images.shape[0])
            batch_images = train_images_shuffled[batch_start:batch_end]
            batch_counts = train_counts_shuffled[batch_start:batch_end]
            
            state_a, state_b, loss, loss_a, loss_b = train_step(
                state_a, state_b, batch_images, batch_counts)
            
            epoch_loss += loss
            epoch_loss_a += loss_a
            epoch_loss_b += loss_b
        
        # Calculate average epoch loss
        epoch_loss /= steps_per_epoch
        epoch_loss_a /= steps_per_epoch
        epoch_loss_b /= steps_per_epoch
        
        # Evaluate on test set
        eval_metrics = evaluate(state_a, state_b, test_images, test_counts)
        
        print(f"Epoch {epoch+1} completed:")
        print(f"  Train Loss: {epoch_loss:.4f} (A: {epoch_loss_a:.4f}, B: {epoch_loss_b:.4f})")
        print(f"  Test Accuracy: Direct A: {eval_metrics['acc_a']:.4f}, Direct B: {eval_metrics['acc_b']:.4f}")
        print(f"                 B from A: {eval_metrics['acc_b_from_a']:.4f}, A from B: {eval_metrics['acc_a_from_b']:.4f}")
    
    return state_a, state_b, model_a, model_b

# Analyze communication patterns
def analyze_communication(state_a, state_b, model_a, model_b):
    # Generate images with 1 to MAX_OBJECTS objects
    images = []
    counts = []
    
    # Generate one image for each possible count
    for count in range(1, MAX_OBJECTS + 1):
        key, subkey = jax.random.split(key)
        image = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        # Add exactly count black circles
        for i in range(count):
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            x = jax.random.randint(subkey1, (1,), 5, IMG_SIZE - 5)[0]
            y = jax.random.randint(subkey2, (1,), 5, IMG_SIZE - 5)[0]
            radius = jax.random.randint(subkey3, (1,), 3, 6)[0]
            
            temp_img = Image.fromarray((image * 255).astype(np.uint8))
            draw = ImageDraw.Draw(temp_img)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=0)
            image = np.array(temp_img).astype(np.float32) / 255.0
        
        images.append(image)
        counts.append(count)
    
    # Convert to JAX arrays
    images = jnp.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    counts = jnp.array(counts)
    
    # Get abstract representations from each model
    _, embeddings_a = model_a.apply({'params': state_a.params}, images, train=False)
    _, embeddings_b = model_b.apply({'params': state_b.params}, images, train=False)
    
    # Analyze the embeddings
    # 1. PCA to visualize the embeddings
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    embeddings_a_pca = pca.fit_transform(np.array(embeddings_a))
    
    plt.figure(figsize=(10, 8))
    for i, count in enumerate(counts):
        plt.scatter(embeddings_a_pca[i, 0], embeddings_a_pca[i, 1], s=100, label=f"{count} objects")
    
    plt.title("PCA of Model A's Abstract Representations")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_a_embeddings_pca.png")
    
    # 2. Check if the abstract representation is "counting" in any way
    # Calculate pairwise distances between embeddings
    from sklearn.metrics.pairwise import euclidean_distances
    
    distances = euclidean_distances(np.array(embeddings_a))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(distances, cmap='viridis')
    plt.colorbar(label="Euclidean Distance")
    plt.title("Pairwise Distances Between Model A's Representations")
    plt.xticks(range(len(counts)), counts)
    plt.yticks(range(len(counts)), counts)
    plt.xlabel("Number of Objects")
    plt.ylabel("Number of Objects")
    plt.savefig("model_a_embedding_distances.png")
    
    # 3. Check if there's a correlation between embedding features and object count
    correlations = []
    for i in range(embeddings_a.shape[1]):
        feature_values = embeddings_a[:, i]
        corr = np.corrcoef(counts, feature_values)[0, 1]
        correlations.append((i, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("Top 5 features by correlation with object count:")
    for i, corr in correlations[:5]:
        print(f"Feature {i}: correlation = {corr:.4f}")
    
    # Plot the top feature against object count
    top_feature_idx = correlations[0][0]
    top_feature_vals = embeddings_a[:, top_feature_idx]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(counts, top_feature_vals, s=100)
    plt.title(f"Top Correlated Feature (Feature {top_feature_idx}) vs. Object Count")
    plt.xlabel("Number of Objects")
    plt.ylabel(f"Feature {top_feature_idx} Value")
    plt.grid(True)
    plt.savefig("top_feature_correlation.png")
    
    return embeddings_a, embeddings_b

# Visualization function
def visualize_model_predictions(state_a, state_b, model_a, model_b):
    # Generate test images with known object counts
    key, subkey = jax.random.split(key)
    test_images = []
    test_counts = []
    
    for i in range(5):  # Generate 5 test cases
        for count in [1, 3, 5, 7, 10]:  # Different object counts
            key, subkey = jax.random.split(key)
            image = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            
            # Add exactly count black circles
            for j in range(count):
                key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
                x = jax.random.randint(subkey1, (1,), 5, IMG_SIZE - 5)[0]
                y = jax.random.randint(subkey2, (1,), 5, IMG_SIZE - 5)[0]
                radius = jax.random.randint(subkey3, (1,), 3, 6)[0]
                
                temp_img = Image.fromarray((image * 255).astype(np.uint8))
                draw = ImageDraw.Draw(temp_img)
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=0)
                image = np.array(temp_img).astype(np.float32) / 255.0
            
            test_images.append(image)
            test_counts.append(count)
    
    # Convert to JAX arrays
    test_images = jnp.array(test_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_counts = jnp.array(test_counts)
    
    # Get predictions
    logits_a, embeddings_a = model_a.apply({'params': state_a.params}, test_images, train=False)
    logits_b, embeddings_b = model_b.apply({'params': state_b.params}, test_images, train=False)
    
    # Cross predictions
    logits_b_from_a = model_b.apply({'params': state_b.params}, embeddings_a, train=False, method=CountingAgent.decode_embedding)
    logits_a_from_b = model_a.apply({'params': state_a.params}, embeddings_b, train=False, method=CountingAgent.decode_embedding)
    
    # Get predicted counts
    pred_a = jnp.argmax(logits_a, axis=1)
    pred_b = jnp.argmax(logits_b, axis=1)
    pred_b_from_a = jnp.argmax(logits_b_from_a, axis=1)
    pred_a_from_b = jnp.argmax(logits_a_from_b, axis=1)
    
    # Visualize some examples
    plt.figure(figsize=(15, 10))
    for i in range(5):  # Show first 5 examples
        plt.subplot(1, 5, i+1)
        plt.imshow(test_images[i, :, :, 0], cmap='gray')
        plt.title(f"True: {test_counts[i]}\nA: {pred_a[i]}, B: {pred_b[i]}\nB←A: {pred_b_from_a[i]}, A←B: {pred_a_from_b[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("model_predictions.png")
    
    # Confusion matrices for the communication
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Convert to numpy for sklearn
    pred_b_from_a_np = np.array(pred_b_from_a)
    test_counts_np = np.array(test_counts)
    
    # Create confusion matrix
    cm = confusion_matrix(test_counts_np, pred_b_from_a_np, labels=range(MAX_OBJECTS+1))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(MAX_OBJECTS+1),
                yticklabels=range(MAX_OBJECTS+1))
    plt.title("Confusion Matrix: Model B's Predictions from Model A's Embeddings")
    plt.xlabel("Predicted Count")
    plt.ylabel("True Count")
    plt.savefig("confusion_matrix_b_from_a.png")

# Run the training process
if __name__ == "__main__":
    print("Starting model training...")
    state_a, state_b, model_a, model_b = train_models()
    
    print("\nAnalyzing communication patterns...")
    embeddings_a, embeddings_b = analyze_communication(state_a, state_b, model_a, model_b)
    
    print("\nVisualizing model predictions...")
    visualize_model_predictions(state_a, state_b, model_a, model_b)
    
    print("\nDone! Check the generated visualization files for results.")