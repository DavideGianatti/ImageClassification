#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import timeit

"""
Different methods for the extraction of features are implemented.
"""

def black_contour(image):
    """
    Converts white background to black in an image.

    Parameters
    ----------
    image : ndarray
        Input image array.

    Returns
    -------
    ndarray
        Image with black background.
    """
    aux = np.copy(image)
    # Threshold for white background
    threshold = 0.95
    for i in range(100):
        for j in range(100):
            if np.mean(image[i][j]) > threshold:
                aux[i][j] = 0
    return aux

def noise(image):
    """
    Adds random noise to the image.

    Parameters
    ----------
    image : ndarray
        Input image array.

    Returns
    -------
    ndarray
        Image with added noise.
    """
    aux = np.copy(image)
    mean = 0
    delta = 0.1
    # Add noise to each pixel if it's not black
    for i in range(100):
        for j in range(100):
            if np.sum(image[i][j]) != 0:
                aux[i][j] += delta * (2 * np.random.rand(3) - 1)
    # Clip values to be between 0 and 1
    aux = np.clip(aux, 0, 1)
    return aux

def mirror(image):
    """
    Reflects the image around the diagonal.

    Parameters
    ----------
    image : ndarray
        Input image array.

    Returns
    -------
    ndarray
        Reflected image.
    """
    aux = np.zeros_like(image)
    for i in range(100):
        for j in range(100):
            aux[i][j] = image[j][i]
    return aux

def blur(image):
    """
    Applies a mean filter to blur the image.

    Parameters
    ----------
    image : ndarray
        Input image array.

    Returns
    -------
    ndarray
        Blurred image.
    """
    aux = np.zeros_like(image)
    for i in range(1, 99):
        for j in range(1, 99):
            aux[i][j] = np.mean(image[i-1:i+2, j-1:j+2], axis=(0, 1))
    return aux

def cooling(image):
    """
    Adds a cooling effect by increasing the blue component.

    Parameters
    ----------
    image : ndarray
        Input image array.

    Returns
    -------
    ndarray
        Image with increased blue component.
    """
    aux = np.copy(image)
    c = 0.1
    # Increase blue component if it's not already close to maximum
    aux[..., 2] = np.clip(aux[..., 2] + c, 0, 1)
    return aux

def warming(image):
    """
    Adds a warming effect by increasing the red component.

    Parameters
    ----------
    image : ndarray
        Input image array.

    Returns
    -------
    ndarray
        Image with increased red component.
    """
    aux = np.copy(image)
    c = 0.1
    # Increase red component if it's not already close to maximum
    aux[..., 0] = np.clip(aux[..., 0] + c, 0, 1)
    return aux

def bright_saturation_histogram(image, bins=64):
    """
    Computes the histogram of brightness and saturation.

    Parameters
    ----------
    image : ndarray
        Input image array.
    bins : int
        Number of bins for the histogram.

    Returns
    -------
    ndarray
        Histogram of brightness and saturation.
    """
    B = np.zeros((100, 100))
    S = np.zeros((100, 100))
    # Calculate brightness and saturation
    for i in range(100):
        for j in range(100):
            max_val = np.max(image[i, j, :])
            min_val = np.min(image[i, j, :])
            B[i, j] = (max_val + min_val) / 2
            if B[i, j] != 0:
                S[i, j] = (max_val - min_val) / (1 - np.abs(2 * B[i, j] - 1))

    B = (B * (bins - 1)).astype(int)
    S = (S * (bins - 1)).astype(int)
    Bhist = np.bincount(B.ravel(), minlength=bins)
    Shist = np.bincount(S.ravel(), minlength=bins)
    hist = np.stack([Bhist, Shist], axis=0)
    hist = hist / np.sum(hist)
    return hist

def color_histogram(img, bins=64):
    """
    Computes the RGB marginal histograms of a color image.

    Parameters
    ----------
    img : ndarray
        Input image array.
    bins : int
        Number of bins for the histograms.

    Returns
    -------
    ndarray
        Marginal histograms for RGB channels.
    """
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)  # Convert grayscale to RGB
    if img.max() > 1:
        img = img / 255.0
    img = (img * (bins - 1)).astype(int)
    rhist = np.bincount(img[:, :, 0].ravel(), minlength=bins)
    ghist = np.bincount(img[:, :, 1].ravel(), minlength=bins)
    bhist = np.bincount(img[:, :, 2].ravel(), minlength=bins)
    hist = np.stack([rhist, ghist, bhist], axis=0)
    hist = hist / (img.shape[0] * img.shape[1])
    return hist

def edge_direction_histogram(img, bins=64):
    """
    Computes the edge direction histogram of an image.

    Parameters
    ----------
    img : ndarray
        Input image array.
    bins : int
        Number of direction bins.

    Returns
    -------
    ndarray
        Edge direction histogram.
    """
    if img.ndim == 3:
        img = img.mean(axis=2)  # Convert to grayscale
    img = img.astype(float)
    gx = img[:, 2:] - img[:, :-2]
    gx = gx[:-2, :] + 2 * gx[1:-1, :] + gx[2:, :]
    gy = img[2:, :] - img[:-2, :]
    gy = gy[:, :-2] + 2 * gy[:, 1:-1] + gy[:, 2:]
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gx, -gy)
    dirs = (bins * angle / (2 * np.pi)).astype(int) % bins
    hist = np.bincount(dirs.ravel(), weights=magnitude.ravel(), minlength=bins)
    hist = hist / max(1e-16, hist.sum())
    return hist

def cooccurrence_matrix(img, bins=8, distance=10):
    """
    Computes the gray level co-occurrence matrix (GLCM) of an image.

    Parameters
    ----------
    img : ndarray
        Input image array.
    bins : int
        Number of gray levels.
    distance : int
        Distance between neighboring pixels.

    Returns
    -------
    ndarray
        Normalized co-occurrence matrix.
    """
    if img.ndim == 3:
        img = img.mean(axis=2)  # Convert to grayscale
    if img.max() > 1:
        img = img / 255.0
    img = (img * (bins - 1)).astype(int)
    mat = _cooccurrence_matrix_dir(img, bins, distance, 0)  # Pixels below
    mat += _cooccurrence_matrix_dir(img, bins, 0, distance)  # Pixels to the right
    mat += mat.T  # Count pixels above and to the left
    mat = mat / mat.sum()
    return mat

def rgb_cooccurrence_matrix(img, quantization=3, distance=10):
    """
    Computes the RGB co-occurrence matrix.

    Parameters
    ----------
    img : ndarray
        Input image array.
    quantization : int
        Number of quantization levels for each color channel.
    distance : int
        Distance between neighboring pixels.

    Returns
    -------
    ndarray
        Normalized RGB co-occurrence matrix.
    """
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)  # Convert grayscale to RGB
    if img.max() > 1:
        img = img / 255.0
    img = (img * (quantization - 1)).astype(int)
    bins = quantization ** 3
    img = (img * np.array([[[1, quantization, quantization ** 2]]])).sum(axis=2)
    mat = _cooccurrence_matrix_dir(img, bins, distance, 0)  # Pixels below
    mat += _cooccurrence_matrix_dir(img, bins, 0, distance)  # Pixels to the right
    mat += mat.T  # Count pixels above and to the left
    mat = mat / mat.sum()
    return mat

def _cooccurrence_matrix_dir(values, bins, di, dj):
    """
    Helper function to compute the co-occurrence matrix for a given direction.

    Parameters
    ----------
    values : ndarray
        Image array with integer values.
    bins : int
        Number of bins.
    di : int
        Row displacement.
    dj : int
        Column displacement.

    Returns
    -------
    ndarray
        Co-occurrence matrix for the specified direction.
    """
    m, n = values.shape
    codes = values[:m - di, :n - dj] + bins * values[di:, dj:]
    entries = np.bincount(codes.ravel(), minlength=bins ** 2)
    return entries.reshape(bins, bins)

def data_augmentation():
    """
    Applies various augmentations to the images and saves them in "fruits/train_extended".
    """
    classes = os.listdir("fruits/train")
    classes.sort()

    for klass in classes:
        class_path = f"fruits/train/{klass}"
        image_files = os.listdir(class_path)
        for imagename in image_files:
            print(imagename)
            image = plt.imread(f"{class_path}/{imagename}")
            image = image / 255
            image = black_contour(image)

            # Save original and augmented images
            plt.imsave(f"fruits/train_extended/{klass}/{imagename}", image)
            plt.imsave(f"fruits/train_extended/{klass}/n_{imagename}", noise(image))
            plt.imsave(f"fruits/train_extended/{klass}/m_{imagename}", mirror(image))
            plt.imsave(f"fruits/train_extended/{klass}/b_{imagename}", blur(image))
            plt.imsave(f"fruits/train_extended/{klass}/c_{imagename}", cooling(image))
            plt.imsave(f"fruits/train_extended/{klass}/w_{imagename}", warming(image))

def save_images(path):
    """
    Saves images and their labels from the specified directory to data files.

    Parameters
    ----------
    path : str
        Subdirectory path under "fruits".
    """
    classes = os.listdir(f"fruits/{path}")
    classes.sort()

    all_features = []
    all_labels = []
    class_label = 0

    for klass in classes:
        class_path = f"fruits/{path}/{klass}"
        image_files = os.listdir(class_path)
        for imagename in image_files:
            print(imagename)
            image = plt.imread(f"{class_path}/{imagename}")
            image = image / 255
            image = black_contour(image)

            all_features.append(image)
            all_labels.append(class_label)
        class_label += 1

    X = np.stack(all_features, axis=0)
    Y = np.array(all_labels)

    os.makedirs('data', exist_ok=True)
    np.save(f"data/image_X_{path}.dat", X)
    np.save(f"data/image_Y_{path}.dat", Y)

def get_images(path):
    """
    Loads images and labels from saved data files.

    Parameters
    ----------
    path : str
        Subdirectory path under "data".

    Returns
    -------
    tuple
        Loaded images and labels.
    """
    X = np.load(f"data/image_X_{path}.dat.npy")
    Y = np.load(f"data/image_Y_{path}.dat.npy")

    return X, Y

def extract_lowlevel(path):
    """
    Extracts low-level features from images in the specified directory.

    Parameters
    ----------
    path : str
        Subdirectory path under "fruits".

    Returns
    -------
    tuple
        Feature matrix and labels.
    """
    classes = os.listdir(f"fruits/{path}")
    classes.sort()

    all_features = []
    all_labels = []
    class_label = 0

    for klass in classes:
        class_path = f"fruits/{path}/{klass}"
        image_files = os.listdir(class_path)
        for imagename in image_files:
            print(imagename)
            image = plt.imread(f"{class_path}/{imagename}")
            image = image / 255

            image = black_contour(image)

            BS_hist = bright_saturation_histogram(image)
            color_hist = color_histogram(image).reshape(-1)
            edge_direct = edge_direction_histogram(image)
            cooccurrence = cooccurrence_matrix(image)
            features = np.concatenate((color_hist, edge_direct, cooccurrence.ravel(), BS_hist), axis=None)
            all_features.append(features)
            all_labels.append(class_label)

        class_label += 1

    X = np.stack(all_features, axis=0)
    Y = np.array(all_labels)

    return X, Y

def neural_features(im, net):
    """
    Extracts features from the last hidden layer of a trained network.

    Parameters
    ----------
    im : ndarray
        Input image.
    net : pvml.CNN
        Pre-trained neural network.

    Returns
    -------
    ndarray
        Feature vector from the network's last hidden layer.
    """
    im = im[None, :, :, :]  # Add batch dimension
    activations = net.forward(im)
    features = activations[-3].reshape(-1)
    return features

def extract_neural_features(path):
    """
    Extracts neural network features from images in the specified directory.

    Parameters
    ----------
    path : str
        Subdirectory path under "fruits".

    Returns
    -------
    tuple
        Feature matrix and labels.
    """
    net = pvml.CNN.load("pvmlnet.npz")  # Load pre-trained network

    classes = os.listdir(f"fruits/{path}")
    classes.sort()

    all_features = []
    all_labels = []
    class_label = 0

    start = timeit.default_timer()

    for klass in classes:
        class_path = f"fruits/{path}/{klass}"
        image_files = os.listdir(class_path)
        for imagename in image_files:
            print(imagename)
            image = plt.imread(f"{class_path}/{imagename}")
            image = image / 255

            image = black_contour(image)
            image = cv2.resize(image, (224, 224))  # Resize for the network
            features = neural_features(image, net)
            all_features.append(features)
            all_labels.append(class_label)

        class_label += 1

    end = timeit.default_timer()
    time_per_image = (end - start) / (len(classes) * len(image_files))
    print(f"Time needed for the extraction of the feature of an image = {time_per_image:.2f} s")

    X = np.stack(all_features, axis=0)
    Y = np.array(all_labels)

    return X, Y
