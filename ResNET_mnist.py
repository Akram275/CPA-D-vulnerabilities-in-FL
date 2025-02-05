import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define the residual block
def residual_block(x, filters, kernel_size, strides=(1, 1)):
    y = layers.Conv2D(filters, kernel_size, padding='same', strides=strides)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)
    y = layers.BatchNormalization()(y)

    # Apply projection shortcut if the dimensions change
    if strides != (1, 1) or x.shape[-1] != filters:
        x = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', strides=strides)(x)

    y = layers.Add()([x, y])  # Skip connection
    y = layers.Activation('relu')(y)
    return y


# The ResNet architecture
def ResNet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Stack residual blocks
    for _ in range(3):
        x = residual_block(x, 64, 3)

    #Intermediate maxPooling2D
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Stack residual blocks again
    for _ in range(3):
        x = residual_block(x, 128, 3)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x)
    return model




# Load and preprocess MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

# Build the ResNet model
input_shape = x_train.shape[1:]
num_classes = 10
model = ResNet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)



names = []
acc = []

#RLWE dtandard deviation to perturb the model
RLWE_stdv = 3.19

for i, layer in enumerate(model.layers) :
   t_p = len(layer.trainable_variables)
   if t_p > 0 :
     print(f"Layer index: {i}, Name: {layer.name}, Trainable parameters: {t_p}, weights.size = {layer.get_weights()[0].size}, bias.size = {layer.get_weights()[1].size}")

     names.append(layer.name)
     #create a clone model
     perturb_model = tf.keras.models.clone_model(model)
     w = perturb_model.layers[i].get_weights()
     curr_acc = []
     for j in range(len(w)) :
         print(f"\n\nperturabting model weights at layer {i}, Name : {layer.name}, vector {j}")
         perturb_model.set_weights(model.get_weights())
         shape = w[j].shape
         w[j] = np.random.normal(0, RLWE_stdv, size=shape)
         #Replace with perturbed layer weights
         perturb_model.layers[i].set_weights(w)
         #Compile model after perturabtions
         perturb_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

         test_loss, test_acc = perturb_model.evaluate(x_test, y_test)
         print("Test accuracy (after replacing weights):", test_acc)
         curr_acc.append(test_acc)
     acc.append(curr_acc)


def plot(acc, names) :
    # Width of each bar
    bar_width = 0.2
    # Set positions for each group of bars
    x = np.arange(len(names))
    # Plotting
    plt.figure(figsize=(12, 6))
    # Plot bars for each layer
    for i, layer_name in enumerate(names):
        #Avoid having layers numbers for a better plot
        if layer_name[0] == 'b' :
            names[i] = 'BN'
        if layer_name[0] == 'c' :
            names[i] = 'Conv'
        num_experiments = len(acc[i])
        for j in range(num_experiments):
            plt.bar(x[i] + j * bar_width, acc[i][j], width=bar_width, color='skyblue', edgecolor='black', align='center', zorder=2)

    plt.xlabel('Layers/Positions', fontsize=18)
    plt.ylabel('Validation Accuracy', fontsize=18)
    plt.xticks(x + ((num_experiments - 0) * bar_width) / 2, names, rotation=45, fontsize=18)  # Position x-ticks at center of each group
    plt.tight_layout()
    plt.ylim(0.0, 1.0)
    plt.axhline(y=0.93, color='skyblue', linestyle='--', label='Pre perturabtion accuracy')
    plt.grid(True, linestyle='--', color='gray')
    plt.legend(['Pre perturabtion accuracy', 'Post perturabtion accuracy'], fontsize=18)
    plt.show()

plot(acc, names)
