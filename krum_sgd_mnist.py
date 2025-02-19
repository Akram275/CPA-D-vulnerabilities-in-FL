import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to generate worker data from MNIST dataset with Dirichlet sampling on class labels
def generate_worker_data(num_workers, num_samples_per_worker, alpha):
    (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0  # Normalize pixel values

    # One-hot encode the labels
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)

    worker_data = []
    num_classes = 10

    # Generate class distributions for each worker
    for _ in range(num_workers):
        class_sizes = np.random.dirichlet(alpha * np.ones(num_classes)) * len(y_train)
        class_indices = [np.where(y_train == i)[0] for i in range(num_classes)]

        worker_indices = []
        for class_index, class_size in zip(class_indices, class_sizes):
            class_indices_sample = np.random.choice(class_index, int(class_size), replace=True)
            worker_indices.extend(class_indices_sample)

        np.random.shuffle(worker_indices)
        worker_X = X_train[worker_indices[:num_samples_per_worker]]
        worker_y = y_train_one_hot[worker_indices[:num_samples_per_worker]]

        worker_data.append((worker_X, worker_y))

    return worker_data

# Define a simple neural network model
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    return model

# Loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.legacy.Adam()

# Function to train the model on worker's data and return gradients
def train_model_and_get_gradients(model, X_train, y_train, replace_gradients=False):
    with tf.GradientTape() as tape:
        logits = model(X_train, training=True)
        print(logits.shape)
        print(y_train.shape)
        #loss = model.compiled_loss(y_train, logits)
        loss = loss_fn(y_train, logits)

    gradients = tape.gradient(loss, model.trainable_variables)

    if replace_gradients:
        # Replace gradients with normally distributed values for demonstration
        for i in range(1):
            gradients[i] = tf.random.normal(gradients[i].shape)

    return gradients

# Krum aggregation
def krum_aggregation(gradients, num_workers, m=4):
    distances = []
    num_variables = len(gradients[0])  # Number of variables (weights and biases)

    # Calculate distances between gradients
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            distance = 0.0
            for k in range(num_variables):
                grad_i_flat = tf.reshape(gradients[i][k], [-1])
                grad_j_flat = tf.reshape(gradients[j][k], [-1])
                distance += tf.norm(grad_i_flat - grad_j_flat)
            distances.append((distance.numpy(), i, j))

    # Sort distances and find the indices of the m nearest gradients
    print('distances : ', distances)
    distances.sort(key=lambda x: x[0])
    krum_indices = [distances[i][1] for i in range(num_workers - m)]
    # Select the gradients from workers not in krum_indices
    selected_gradients = [gradients[idx] for idx in range(num_workers) if idx not in krum_indices]
    print('selected gradients for aggregation are : ', krum_indices)
    return selected_gradients, distances

# Function to perform Krum aggregation on gradients from multiple workers
def perform_krum_aggregation(workers_data, num_workers, m=4, replace_gradients_index=None):
    # Initialize model and optimizer
    model = create_model()

    # Train models on each worker's data and compute gradients
    gradients = []
    for i, worker_data in enumerate(workers_data):
        X_train = worker_data[0]
        y_train = worker_data[1]
        print('X_train shape = ', X_train.shape)
        print('y_train shape = ', y_train.shape)
        #model.compile(optimizer=optimizer, loss=loss_fn)
        model.fit(X_train, np.argmax(y_train, axis=1), epochs=1, batch_size=32)
        replace_gradients = (replace_gradients_index is not None and i == replace_gradients_index)
        worker_gradients = train_model_and_get_gradients(model, X_train, y_train, replace_gradients)
        gradients.append(worker_gradients)

    # Perform Krum aggregation
    aggregated_gradients, distances = krum_aggregation(gradients, num_workers=num_workers, m=m)

    return aggregated_gradients, distances

# Generate data for 10 workers with Dirichlet sampling on class labels
num_samples_per_worker = 2000
alpha = [0.001] * 10  # Dirichlet parameters for class label distribution
#workers_data = [generate_worker_data(10, num_samples_per_worker, alpha) for _ in range(10)]
workers_data = generate_worker_data(10, num_samples_per_worker, alpha)

# Perform Krum aggregation without replacing gradients
print("Performing Krum aggregation without replaced gradients:")
aggregated_gradients = perform_krum_aggregation(workers_data, num_workers=10, m=4)
# Print aggregated gradients shapes



# Perform Krum aggregation with replaced gradients in one worker (for example, at index 0)
replace_gradients_index = 0
print("\nPerforming Krum aggregation with replaced gradients at worker index", replace_gradients_index)
aggregated_gradients_replaced, distances = perform_krum_aggregation(workers_data, num_workers=10, m=4, replace_gradients_index=replace_gradients_index)



for k in range(10):
    worker_index = k
    krum_score = 0.0
    count = 0
    for dist, i, j in distances:
        if i == worker_index:
            krum_score += dist
            count += 1
        elif j == worker_index:
            krum_score += dist
            count += 1

    print(f"Krum score for worker {worker_index}: {krum_score}")
