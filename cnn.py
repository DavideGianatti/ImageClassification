#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from extract_features import get_images, save_images
import os

"""
This script implements, trains, and saves different architectures of Convolutional Neural Networks (CNNs).
The trained models are saved in the "networks" directory.
"""

def build_kinda_big_cnn():
    """
    Builds a relatively larger CNN model with more layers and larger filter sizes.

    Returns:
        tensorflow.keras.models.Sequential: The built CNN model.
    """
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(32, (3,3), activation='relu', strides=(3,3), padding='same', input_shape=(100,100,3)))
    cnn.add(layers.MaxPooling2D((2,2)))
    cnn.add(layers.Conv2D(64, (3,3), activation='relu', strides=(2,2)))
    cnn.add(layers.MaxPooling2D((2,2)))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(16, activation='softmax'))

    return cnn

def build_cnn():
    """
    Builds a standard CNN model with smaller filter sizes and fewer layers.

    Returns:
        tensorflow.keras.models.Sequential: The built CNN model.
    """
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(8, (3,3), activation='relu', strides=(5,5), padding='same', input_shape=(100,100,3)))
    cnn.add(layers.MaxPooling2D((2,2)))
    cnn.add(layers.Conv2D(16, (3,3), activation='relu', strides=(3,3)))
    cnn.add(layers.MaxPooling2D((2,2)))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(16, activation='softmax'))

    return cnn

def build_mini_cnn():
    """
    Builds a smaller CNN model with fewer filters and layers.

    Returns:
        tensorflow.keras.models.Sequential: The built CNN model.
    """
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(3, (3,3), activation='relu', strides=(5,5), padding='same', input_shape=(100,100,3)))
    cnn.add(layers.MaxPooling2D((2,2)))
    cnn.add(layers.Conv2D(6, (2,2), activation='relu', strides=(2,2)))
    cnn.add(layers.AveragePooling2D((5,5)))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(16, activation='softmax'))  # Adjust output units if needed

    return cnn

def add_bottleneck(cnn):
    """
    Adds a bottleneck layer to the CNN to reduce the number of parameters.

    Args:
        cnn (tensorflow.keras.models.Sequential): The CNN model to modify.
    """
    cnn.pop()  # Remove the last layer
    cnn.add(layers.Dense(2, activation='relu'))  # Add a bottleneck layer
    cnn.add(layers.Dense(16, activation='softmax'))  # Add the final classification layer

def mini_lr_l2(X_train, Y_train, X_validation, Y_validation):
    """
    Performs grid search for learning rates and L2 regularization coefficients.

    Args:
        X_train (numpy.ndarray): Training feature set.
        Y_train (numpy.ndarray): Training labels.
        X_validation (numpy.ndarray): Validation feature set.
        Y_validation (numpy.ndarray): Validation labels.
    """
    n_epochs = 250
    eta_trial = [0.0001, 0.001, 0.01, 0.1]  # Learning rates to try
    lamda_trial = [0, 0.001, 0.01, 0.1]  # L2 regularization coefficients to try
    eta_lamda = np.zeros((len(eta_trial), len(lamda_trial)))

    for i, eta in enumerate(eta_trial):
        for j, lamda in enumerate(lamda_trial):
            cnn = build_mini_cnn()
            opt = tf.keras.optimizers.Adam(weight_decay=lamda, learning_rate=eta)
            cnn.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        steps_per_execution=1600, metrics=['accuracy'])
            history = cnn.fit(X_train, Y_train, epochs=n_epochs, validation_data=(X_validation, Y_validation))

            eta_lamda[i, j] = history.history["val_accuracy"][-1]  # Store final validation accuracy

    print("Learning Rates: ", eta_trial)
    print("L2 Regularization Coefficients: ", lamda_trial)
    print("Grid Search Results: ", eta_lamda)

def mini_batchsize(X_train, Y_train, X_validation, Y_validation):
    """
    Performs grid search for batch sizes.

    Args:
        X_train (numpy.ndarray): Training feature set.
        Y_train (numpy.ndarray): Training labels.
        X_validation (numpy.ndarray): Validation feature set.
        Y_validation (numpy.ndarray): Validation labels.
    """
    n_epochs = 250
    batch_trial = [960, 1600, 2400]  # Batch sizes to try
    val_accuracies = np.zeros(len(batch_trial))

    for i, batch in enumerate(batch_trial):
        print("Processing batch size: ", 4800 // batch)
        cnn = build_mini_cnn()
        cnn.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    steps_per_execution=batch, metrics=['accuracy'])
        history = cnn.fit(X_train, Y_train, epochs=n_epochs, validation_data=(X_validation, Y_validation))

        val_accuracies[i] = history.history["val_accuracy"][-1]

    print("Batch Sizes: ", batch_trial)
    print("Validation Accuracies: ", val_accuracies)

def test_accuracy(X_test, Y_test):
    """
    Tests the accuracy of a saved CNN model on the test set.

    Args:
        X_test (numpy.ndarray): Test feature set.
        Y_test (numpy.ndarray): Test labels.
    """
    cnn_name = "mini_cnn"
    cnn = tf.keras.models.load_model("networks/" + cnn_name)
    score, acc = cnn.evaluate(X_test, Y_test)

    print("Accuracy on the Test Set: ", acc)

if __name__ == "__main__":

    # Save pre-processed dataset if needed
    for path in ["train", "validation", "test"]:
        if not os.path.isfile(f"data/image_X_{path}.dat"):
            save_images(path)

    # Load datasets
    X_train, Y_train = get_images("train")
    X_test, Y_test = get_images("test")
    X_validation, Y_validation = get_images("validation")

    # Build and train CNN model
    cnn_name = "mini_cnn"
    cnn = build_mini_cnn()
    opt = tf.keras.optimizers.Adam(weight_decay=0.001, learning_rate=0.01)
    cnn.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                steps_per_execution=1600, metrics=['accuracy'])
    history = cnn.fit(X_train, Y_train, epochs=250, validation_data=(X_validation, Y_validation))

    # Save the trained model
    os.makedirs('networks', exist_ok=True)
    cnn.save("networks/" + cnn_name)
