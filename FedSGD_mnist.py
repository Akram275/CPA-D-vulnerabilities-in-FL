import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import clone_model
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from sklearn.svm import SVC
from collections import Counter
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, f1_score
import keras


def Le_NET(distrib='glorot_uniform'):
    """
    Creates a LeNET CNN model architecture for (FE)MNIST with customizable kernel initializer.

    Parameters:
        distrib (str): Kernel initializer to use (e.g., 'uniform', 'glorot_uniform').

    Returns:
        model (Sequential): Compiled CNN model for MNIST.
    """
    model = Sequential([
        #First Conv block
        Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=distrib, input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        #Second Conv block
        Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer=distrib),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),

        #Final Dense block
        Dense(128, activation='relu', kernel_initializer=distrib),
        BatchNormalization(),
        Dense(10, activation='softmax', kernel_initializer=distrib)
    ])
    #tensorflow_privacy.DPKerasSGDOptimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    return model



def get_heterogeneous_data_mnist(n_clients, alpha=0.5):
    """
    Returns MNIST datasets split among clients with parametric data heterogeneity.

    Args:
        n_clients (int): Number of clients.
        alpha (float): Concentration parameter for Dirichlet distribution
                       (lower values increase heterogeneity).

    Returns:
        training_data (list): List of tuples (x_train_client, y_train_client) for each client.
        test_data (list): List of tuples (x_test_client, y_test_client) for each client.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten the labels (1D array)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # One-hot encode the labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Generate Dirichlet-distributed label proportions for clients
    label_distribution = np.random.dirichlet([alpha] * 10, n_clients)

    # Allocate training data to clients based on label proportions
    train_indices_by_class = [np.where(y_train == c)[0] for c in range(10)]
    client_train_indices = [[] for _ in range(n_clients)]
    for c, indices in enumerate(train_indices_by_class):
        np.random.shuffle(indices)
        proportions = label_distribution[:, c]
        splits = (proportions * len(indices)).astype(int)
        splits[-1] = len(indices) - sum(splits[:-1])  # Adjust to use all samples
        current_idx = 0
        for client_id, split in enumerate(splits):
            client_train_indices[client_id].extend(indices[current_idx:current_idx + split])
            current_idx += split

    # Allocate test data evenly among clients
    test_indices_by_class = [np.where(y_test == c)[0] for c in range(10)]
    client_test_indices = [[] for _ in range(n_clients)]
    for c, indices in enumerate(test_indices_by_class):
        np.random.shuffle(indices)
        splits = np.array_split(indices, n_clients)
        for client_id, split in enumerate(splits):
            client_test_indices[client_id].extend(split)

    # Prepare training and test data for each client
    training_data = []
    test_data = []

    for i in range(n_clients):
        x_train_client = x_train[client_train_indices[i]]
        y_train_client = y_train_cat[client_train_indices[i]]

        x_test_client = x_test[client_test_indices[i]]
        y_test_client = y_test_cat[client_test_indices[i]]

        training_data.append((x_train_client, y_train_client))
        test_data.append((x_test_client, y_test_client))


    return training_data, test_data
    #training_data[0] --> client 0 --> training_data[0][0] (images) training_data[0][1]

def update_local_model(agg_model, input_shape, lr=0.001) :
    #update the local models from the aggregated one received from server
    local_model = tf.keras.models.clone_model(agg_model)
    local_model.build(input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    local_model.compile(
            optimizer=optimizer,
            loss=keras.losses.categorical_crossentropy,
            metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision')
                ]
            )

    local_model.set_weights(agg_model.get_weights())
    return local_model



def compute_gradients(model, x_train, y_train):
    """Compute gradients for a client using their local data."""
    with tf.GradientTape() as tape:
        #tape.watch(model.trainable_variables)
        y_pred = model(x_train, training=True)
        loss = keras.losses.categorical_crossentropy(y_train, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients



def aggregate_gradients(client_gradients):
    """Aggregate gradients by averaging them across clients."""
    aggregated_gradients = []
    for grads in zip(*client_gradients):
        #aggregated_gradients.append(np.mean(grads, axis=0))
        aggregated_gradients.append(tf.reduce_mean(tf.stack(grads), axis=0))
    return aggregated_gradients



def create_batches(X, y, batch_size):
    """
    Splits X and y into batches of a given size.

    Args:
        X (np.ndarray): Input data (features).
        y (np.ndarray): Labels corresponding to X.
        batch_size (int): The batch size to split the data into.

    Returns:
        batches (list): A list of tuples, where each tuple contains a batch of (X_batch, y_batch).
    """
    batches = []
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split into batches
    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        batches.append((X_batch, y_batch))

    return batches



def average_cosine_similarity(gradients_list):
    """
    Computes the average cosine similarity between all pairs of client gradients.

    Args:
        gradients_list (list of list of tf.Tensor):
            A list where each entry is a list of gradient tensors for a client.

    Returns:
        float: The average cosine similarity.
    """
    n = len(gradients_list)

    # Flatten and convert each client's gradients into a single 1D numpy array
    flattened_gradients = [np.concatenate([g.numpy().flatten() for g in client_grads]) for client_grads in gradients_list]

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine(flattened_gradients[i], flattened_gradients[j])
            similarities.append(1 - sim)
            #print((i, j), sim)
    return np.mean(similarities), np.max(similarities), np.std(similarities)




def clip_and_noise(grad, norm, noise_std):
    """
    Clips the gradient norm and adds Gaussian noise for differential privacy.

    Parameters:
    - grad: List of tensors representing gradients.
    - norm: The L2 norm threshold for clipping.
    - noise_std: Standard deviation of the Gaussian noise to be added.

    Returns:
    - List of clipped and noisy gradients as tensors.
    """
    clipped_grad = [tf.clip_by_norm(g, norm) for g in grad]  # Clip gradients
    noisy_grad = [g + tf.random.normal(tf.shape(g), mean=0.0, stddev=noise_std, dtype=g.dtype) for g in clipped_grad]  # Add noise
    return noisy_grad


def compute_gradients_microbatch(model, X, y, batch_size=32):
    """
    Computes per-microbatch gradients with clipping and noise addition.

    Parameters:
    - model: The model for which gradients are computed.
    - X: Input data.
    - y: Labels.
    - batch_size: Size of microbatches.

    Returns:
    - Aggregated noisy gradients.
    """
    microbatches = create_batches(X, y, batch_size)  # Create microbatches
    microbatch_gradients = []

    for batch in microbatches:
        grad = compute_gradients(model, batch[0], batch[1])   # Compute raw gradients
        grad = clip_and_noise(grad, norm=3.0, noise_std=1.5)  # Clip and add noise
        microbatch_gradients.append(grad)

    aggregated_gradients = aggregate_gradients(microbatch_gradients)
    return aggregated_gradients


def run_training(data, rounds, seed, lr_decay=True, use_dp=False, batch_size=64) :
    scores = []

    training_data, test_data = data

    n_workers = len(training_data)

    x_test = np.concatenate([data[0] for data in test_data], axis=0)
    y_test = np.concatenate([data[1] for data in test_data], axis=0)

    global_model = Le_NET('lecun_uniform')
    comm_round = 0
    for t in range(rounds):
        batches = []
        #Split and reshuffle data for workers
        for i in range(n_workers):
            batches.append(create_batches(training_data[i][0], training_data[i][1], batch_size))
        for i, batch_list in enumerate(zip(*batches)):
            comm_round +=1
            print(f'\nstep {i}/{len(batches[0])} round {t+1} comm. round {comm_round}')
            client_gradients = []
            #Workers contribute their respective gradients from current global model on synchronized batches.
            #parallelizable
            for i in range(n_workers) :
                if use_dp :
                    grad = compute_gradients_microbatch(global_model, batch_list[i][0], batch_list[i][1])
                else :
                    grad = compute_gradients(global_model, batch_list[i][0], batch_list[i][1])
                client_gradients.append(grad)

            aggregated_gradients = aggregate_gradients(client_gradients)
            measures = average_cosine_similarity(client_gradients)
            print(f'average cosine : {measures[0]} max directional shift {measures[1]} std {measures[2]}')

            # Update the global model with the aggregated gradients using apply_gradients
            for layer_idx, grad in enumerate(aggregated_gradients):
                #global_model.optimizer.apply_gradients(zip(grad, global_model.trainable_variables[layer_idx:layer_idx+1]))
                global_model.optimizer.apply_gradients(zip(aggregated_gradients, global_model.trainable_variables))
                #\theta^t+1 = \theta^t - learning_rate * aggregated_gradients
            eval = global_model.evaluate(x_test, y_test, verbose=0)
            print(f'loss {eval[0]}, accuracy {eval[1]}')
            #learning rate decay after each round for better stability
            if lr_decay :
                tf.keras.backend.set_value(global_model.optimizer.lr, tf.keras.backend.get_value(global_model.optimizer.lr) * 0.75)

    pass #Do stuff with the model and the measures


if __name__ =='__main__' :
      max_seeds=10
      for seed in range(0, max_seeds) :
        n_clients = 11
        alphas = [0.1, 1, 10, 100]
        training_data, test_data = get_heterogeneous_data_mnist(n_clients, alpha=alphas[0])
        print("Training Data Label Distribution:")
        for i, (x_train_client, y_train_client) in enumerate(training_data):
            label_counts = Counter(np.argmax(y_train_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        print("\nTest Data Label Distribution:")
        for i, (x_test_client, y_test_client) in enumerate(test_data):
            label_counts = Counter(np.argmax(y_test_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        data = (training_data, test_data)
        run_training(data, rounds=30, seed=seed, lr_decay=False, use_dp=True, batch_size=64)
