#!/usr/bin/env python3

import numpy as np
import timeit
from sklearn.decomposition import PCA
from extract_features import extract_lowlevel, black_contour, bright_saturation_histogram, color_histogram, edge_direction_histogram, rgb_cooccurrence_matrix
from classifiers import linear_svm
import os

"""
Training of a classifier based on low-level features.
"""

def save_feature(path):
    """
    Save low-level features and labels to disk.

    Parameters
    ----------
    path : str
        The dataset path (train/validation/test).
    """
    X, Y = extract_lowlevel(path)

    os.makedirs('data', exist_ok=True)
    np.save(f"data/low_X_{path}.dat", X)
    np.save(f"data/low_Y_{path}.dat", Y)

def load_feature(path):
    """
    Load low-level features and labels from disk.

    Parameters
    ----------
    path : str
        The dataset path (train/validation/test).

    Returns
    -------
    tuple
        (X, Y) - Low-level features and corresponding labels.
    """
    X = np.load(f"data/low_X_{path}.dat.npy")
    Y = np.load(f"data/low_Y_{path}.dat.npy")
    return X, Y

def how_fast(model):
    """
    Measure the time taken to extract features and classify an image.

    Parameters
    ----------
    model : sklearn classifier
        The classifier model to use for prediction.

    Returns
    -------
    float
        Average time taken for feature extraction and classification.
    """
    N = 100  # Number of experiments
    t_wasted = np.zeros(N)
    image = plt.imread("fruits/train/apple/apple-001.jpg")
    image = black_contour(image)  # Pre-process the image
    image = image / 255

    for i in range(N):
        start = timeit.default_timer()

        # Extract features
        BS_hist = bright_saturation_histogram(image)
        color_hist = color_histogram(image).reshape(-1)
        edge_direct = edge_direction_histogram(image)
        cooccurrence = rgb_cooccurrence_matrix(image)
        features = np.concatenate((color_hist, edge_direct, cooccurrence.reshape(-1), BS_hist), axis=None)

        # Predict using the model
        model.predict(features.reshape(1, -1))

        end = timeit.default_timer()
        t_wasted[i] = end - start

    avg_time = np.mean(t_wasted)
    std_time = np.std(t_wasted)
    print(f"Low level feature, Time Wasted: {avg_time:.4f} +/- {std_time:.4f} s")

    return avg_time

if __name__ == "__main__":

    # Save features dataset if needed
    for path in ["train", "validation", "test"]:
        if not os.path.isfile(f"data/image_X_{path}.dat"):
            save_feature(path)

    # Load features and labels
    X_train, Y_train = load_feature("train")
    X_validation, Y_validation = load_feature("validation")
    X_test, Y_test = load_feature("test")

    # Train and evaluate a classifier from classifiers.py
    linear_svm(X_train, X_validation, X_test, Y_train, Y_validation, Y_test, lamda=0.03)
