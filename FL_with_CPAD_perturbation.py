import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import clone_model
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from collections import Counter
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, f1_score
import csv


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

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum([tf.convert_to_tensor(grad_list_tuple[i]) for i in range(len(scaled_weight_list))] , axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def FedAvg(models, n, clients_weights) :
    scaled_weights = []

    global_model = Le_NET('zeros')
    for i in range(n) :
        scaled_weights.append(scale_model_weights(models[i].get_weights(), clients_weights[i]))
    avg_weights = sum_scaled_weights(scaled_weights)
    global_model.set_weights([avg_weight_layer.numpy() for avg_weight_layer in avg_weights])
    #global_model.set_weights(avg_weights)
    return global_model


def update_local_model(agg_model, input_shape, lr=0.001) :
    #update the local models from the aggregated one received from server
    local_model = tf.keras.models.clone_model(agg_model)
    local_model.build(input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    local_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision')
                ]
            )

    local_model.set_weights(agg_model.get_weights())
    return local_model


def perturb_model(model, layer_idx, noise_std=0.1, lr=0.0001):

    # Clone the model structure
    noisy_model = clone_model(model)
    noisy_model.build(model.input_shape)

    # Add noise to the weights of the cloned model
    original_layer = model.layers[layer_idx]
    if hasattr(original_layer, 'weights'):
        original_weights = original_layer.get_weights()
        # Add noise to each weight
        noisy_weights = [
            w + (np.random.normal(0, noise_std, w.shape)) for w in original_weights
        ]
        noisy_model.layers[layer_idx].set_weights(noisy_weights)
    #Recompile model
    noisy_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return noisy_model



# Function to load data from an npz file for a client
def load_client_data(npz_file):
    data = np.load(npz_file)
    x, y = data["x"], data["y"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
    return (x_train, y_train), (x_test, y_test)


def get_honest_data(n_clients) :
    training_data = []
    test_data = []

    for i in range(n_clients) :
        client_data_path = f"/home/akram/MetaClassifier/FL_simulations/data/femnist/honest_data/client_{i}.npz"
        (x_train, y_train), (x_test, y_test) = load_client_data(client_data_path)

        x_train = x_train.astype('float32') / 1.0
        x_test = x_test.astype('float32') / 1.0

        y_train_cat = to_categorical(y_train, 10)
        y_test_cat = to_categorical(y_test, 10)

        training_data.append((x_train, y_train_cat))
        test_data.append((x_test, y_test_cat))

    return training_data, test_data

def get_honest_data_mnist(n_clients):

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Split the training data among clients
    train_splits = np.array_split(np.arange(x_train.shape[0]), n_clients)
    test_splits = np.array_split(np.arange(x_test.shape[0]), n_clients)

    training_data = []
    test_data = []

    for i in range(n_clients):
        x_train_client = x_train[train_splits[i]]
        y_train_client = y_train_cat[train_splits[i]]

        x_test_client = x_test[test_splits[i]]
        y_test_client = y_test_cat[test_splits[i]]

        training_data.append((x_train_client, y_train_client))
        test_data.append((x_test_client, y_test_client))

    return training_data, test_data


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



def run_training(data, local_epochs, rounds, seed, lr_decay=True, perturb=False, perturbation_time=None) :

    scores = []

    training_data, test_data = data

    n_workers = len(training_data)

    x_test = np.concatenate([data[0] for data in test_data], axis=0)
    y_test = np.concatenate([data[1] for data in test_data], axis=0)

    global_model = Le_NET('lecun_uniform')

    #Log file names according to the experimental setting :
    if perturb :
        if iid_data :
            if perturbation_time < 5 :
                conv_file_name = f"conv_results/iid-data/perturbed/early/convergence_seed_{seed}.csv"
            if perturbation_time == 15 :
                conv_file_name = f"conv_results/iid-data/perturbed/intermediate/convergence_seed_{seed}.csv"
            else :
                conv_file_name = f"conv_results/iid-data/perturbed/late/convergence_seed_{seed}.csv"
        else :
            if perturbation_time < 5 :
                conv_file_name = f"conv_results/non-iid-data/perturbed/early/convergence_seed_{seed}.csv"
            if perturbation_time == 15 :
                conv_file_name = f"conv_results/non-iid-data/perturbed/intermediate/convergence_seed_{seed}.csv"
            else :
                conv_file_name = f"conv_results/non-iid-data/perturbed/late/convergence_seed_{seed}.csv"
    else :
        if iid_data :
            conv_file_name = f"conv_results/iid-data/regular/convergence_seed_{seed}.csv"
        else :
            conv_file_name = f"conv_results/non-iid-data/regular/convergence_seed_{seed}.csv"

    with open(conv_file_name, mode="a", newline='') as f1 :
        writer1 = csv.writer(f1)
        #Main training loop
        for t in range(rounds):
            perturbation = 0
            print(f'\nRound {t} ...')
            models = []
            if lr_decay :               #Local learning rate for this round
                lr=0.001/(10*(t+1))     #If clients data is IID : Start strong and decay quick
            else :
                lr=0.0001               #Else : A stable learning rate
            #Update clients
            for i in range(n_workers):
                model = update_local_model(global_model, (28, 28, 1), lr=lr)
                model.fit(training_data[i][0], training_data[i][1], epochs=local_epochs, batch_size=8, verbose=0)
                print(f'Updating client {i} ... ', model.evaluate(test_data[i][0], test_data[i][1], verbose=0))
                models.append(model)

            global_model = FedAvg(models, n_workers, [np.round(1 / n_workers, 5) for i in range(n_workers)])
            if perturb and t == perturbation_time :
                global_model = Le_NET('he_normal')
                global_model = perturb_model(global_model, layer_idx=8, noise_std=3.19)
                perturbation = 1
            score = global_model.evaluate(x_test, y_test, verbose=0)
            score.append(perturbation)
            scores.append([score])
            print(r'Global score: ', score)
            writer1.writerows([score])


if __name__ =='__main__' :
    #setup a data heterogeneity setting
    iid_data = True

    if iid_data :
        alpha = 10
    else :
        alpha = 0.1


    max_seeds = 10
    for seed in range(3, max_seeds) :
        training_data, test_data = get_heterogeneous_data_mnist(5, alpha=alpha)
        print("Training Data Label Distribution:")
        for i, (x_train_client, y_train_client) in enumerate(training_data):
            label_counts = Counter(np.argmax(y_train_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        print("\nTest Data Label Distribution:")
        for i, (x_test_client, y_test_client) in enumerate(test_data):
            label_counts = Counter(np.argmax(y_test_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        data = (training_data, test_data)
        run_training(data, local_epochs=1, rounds=30, seed=seed, lr_decay=False)
    """
    #Early perturbation
    for seed in range(0, max_seeds) :
        training_data, test_data = get_heterogeneous_data_mnist(5, alpha=alpha)

        print("Training Data Label Distribution:")
        for i, (x_train_client, y_train_client) in enumerate(training_data):
            label_counts = Counter(np.argmax(y_train_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        print("\nTest Data Label Distribution:")
        for i, (x_test_client, y_test_client) in enumerate(test_data):
            label_counts = Counter(np.argmax(y_test_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        data = (training_data, test_data)
        run_training(data, local_epochs=1, rounds=30, seed=seed, lr_decay=False, perturb=True, perturbation_time=2)

    #Intermediate perturbation
    for seed in range(0, max_seeds) :
        training_data, test_data = get_heterogeneous_data_mnist(5, alpha=alpha)

        print("Training Data Label Distribution:")
        for i, (x_train_client, y_train_client) in enumerate(training_data):
            label_counts = Counter(np.argmax(y_train_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        print("\nTest Data Label Distribution:")
        for i, (x_test_client, y_test_client) in enumerate(test_data):
            label_counts = Counter(np.argmax(y_test_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        data = (training_data, test_data)
        run_training(data, local_epochs=1, rounds=30, seed=seed, lr_decay=False, perturb=True, perturbation_time=15)


    #Late perturbation
    for seed in range(0, max_seeds) :
        training_data, test_data = get_heterogeneous_data_mnist(5, alpha=alpha)

        print("Training Data Label Distribution:")
        for i, (x_train_client, y_train_client) in enumerate(training_data):
            label_counts = Counter(np.argmax(y_train_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        print("\nTest Data Label Distribution:")
        for i, (x_test_client, y_test_client) in enumerate(test_data):
            label_counts = Counter(np.argmax(y_test_client, axis=1))
            print(f"Client {i}: {dict(label_counts)}")

        data = (training_data, test_data)
        run_training(data, local_epochs=1, rounds=30, seed=seed, lr_decay=False, perturb=True, perturbation_time=25)
    """
