#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from extract_features import black_contour, get_images, save_images
from classifiers import linear_svm
import timeit

"""
Training of different classifiers using the last hidden layer of a CNN
"""

def get_lastlayer(X_train, net):
    """
    Extracts features from the second last layer of a CNN for each image in X_train.

    Parameters
    ----------
    X_train : ndarray
        Array of training images.
    net : str
        Filename of the CNN model to load.

    Returns
    -------
    list
        List of features extracted from the second last layer of the CNN.
    """
    # Load the CNN model
    cnn = tf.keras.models.load_model(f"networks/{net}")
    # Create a new model with all layers except the last one
    last_layer = tf.keras.models.Sequential(cnn.layers[:-1])

    m = X_train.shape[0]
    features = []

    # Extract features for each image
    for i in range(m):
        image = X_train[i]
        features.append(last_layer(image[None, :,:,:])[0])

    return features

def how_fast(cnn_name, model):
    """
    Measures the time required to extract features and classify them.

    Parameters
    ----------
    cnn_name : str
        Filename of the CNN model to load.
    model : sklearn classifier
        Classifier model for prediction.

    Returns
    -------
    float
        Average time taken for feature extraction and classification.
    """
    N = 1000  # Number of experiments
    t_wasted = np.zeros(N)

    # Load the CNN model
    cnn = tf.keras.models.load_model(f"networks/{cnn_name}")
    last_layer = tf.keras.models.Sequential(cnn.layers[:-1])
    image = plt.imread("fruits/train/apple/apple-001.jpg")
    image = black_contour(image)  # Pre-process the image
    image = image / 255

    for i in range(N):
        start = timeit.default_timer()

        # Extract features
        features = np.array(last_layer(image[None, :,:,:])[0])
        model.predict(features.reshape(1, -1))

        end = timeit.default_timer()

        t_wasted[i] = (end - start)

    avg_time = np.mean(t_wasted)
    std_time = np.std(t_wasted)
    print(f"Low level feature, Time Wasted: {avg_time:.4f} +/- {std_time:.4f} s")

    return avg_time

if __name__ == "__main__":

    # Save pre-processed dataset if needed
    for path in ["train", "validation", "test"]:
        if not os.path.isfile(f"data/image_X_{path}.dat"):
            save_images(path)

    # Load image datasets
    X_train, Y_train = get_images("train")
    X_test, Y_test = get_images("test")
    X_validation, Y_validation = get_images("validation")

    # Extract features from the last hidden layer of the CNN
    cnn_name = "mini_cnn"
    features_train = get_lastlayer(X_train, cnn_name)
    features_val = get_lastlayer(X_validation, cnn_name)
    features_test = get_lastlayer(X_test, cnn_name)

    # Train and evaluate a classifier from classifiers.py
    linear_svm(features_train, features_val, features_test, Y_train, Y_validation, Y_test)
