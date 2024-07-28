import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

"""
This script implements several classifiers and evaluates their performance on training, validation, and test sets.
It includes:
- Linear SVM
- Polynomial SVM
- Radial Basis Function (RBF) SVM
- Minimum Distance (MD) classifier
- Regularization coefficient search for SVM
"""

def accuracy(labels, Y):
    """
    Calculate the accuracy of predictions.

    Args:
        labels (np.ndarray): Predicted labels.
        Y (np.ndarray): True labels.

    Returns:
        float: Accuracy of the predictions.
    """
    return (labels == Y).mean()

def confusion_matrix(labels, Y):
    """
    Compute the confusion matrix for classification.

    Args:
        labels (np.ndarray): Predicted labels.
        Y (np.ndarray): True labels.

    Returns:
        np.ndarray: Confusion matrix normalized by the number of samples.
    """
    cm = np.zeros((16, 16))

    for i in range(len(labels)):
        cm[Y[i], labels[i]] += 1

    cm /= len(labels)

    return cm

def linear_svm(features_train, features_val, features_test, Y_train, Y_validation, Y_test, lamda=1):
    """
    Train and evaluate a linear SVM classifier.

    Args:
        features_train (np.ndarray): Training feature matrix.
        features_val (np.ndarray): Validation feature matrix.
        features_test (np.ndarray): Test feature matrix.
        Y_train (np.ndarray): Training labels.
        Y_validation (np.ndarray): Validation labels.
        Y_test (np.ndarray): Test labels.
        lamda (float): Regularization parameter (default is 1).

    Returns:
        svm.SVC: Trained linear SVM model.
    """
    model = svm.SVC(C=1/lamda, kernel='linear', decision_function_shape='ovr')
    model.fit(features_train, Y_train)

    print("Number of Support Vectors: ", len(model.support_vectors_))

    labels_train = model.predict(features_train)
    labels_val = model.predict(features_val)
    labels_test = model.predict(features_test)

    print("Accuracy on the Training set: ", accuracy(labels_train, Y_train))
    print("Accuracy on the Validation set: ", accuracy(labels_val, Y_validation))
    print("Accuracy on the Test set: ", accuracy(labels_test, Y_test))

    cm = confusion_matrix(labels_val.astype(int), Y_validation)
    plt.title("Confusion Matrix of the Validation Set")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()

    return model

def poly_svm(features_train, features_val, features_test, Y_train, Y_validation, Y_test, d=3):
    """
    Train and evaluate a polynomial SVM classifier.

    Args:
        features_train (np.ndarray): Training feature matrix.
        features_val (np.ndarray): Validation feature matrix.
        features_test (np.ndarray): Test feature matrix.
        Y_train (np.ndarray): Training labels.
        Y_validation (np.ndarray): Validation labels.
        Y_test (np.ndarray): Test labels.
        d (int): Degree of the polynomial kernel (default is 3).

    Returns:
        svm.SVC: Trained polynomial SVM model.
    """
    model = svm.SVC(kernel='poly', degree=d, decision_function_shape='ovr')
    model.fit(features_train, Y_train)

    print("Number of Support Vectors: ", len(model.support_vectors_))

    labels_train = model.predict(features_train)
    labels_val = model.predict(features_val)
    labels_test = model.predict(features_test)

    print("Accuracy on the Training set: ", accuracy(labels_train, Y_train))
    print("Accuracy on the Validation set: ", accuracy(labels_val, Y_validation))
    print("Accuracy on the Test set: ", accuracy(labels_test, Y_test))

    cm = confusion_matrix(labels_val.astype(int), Y_validation)
    plt.title("Confusion Matrix of the Validation Set")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()

    return model

def rbf_svm(features_train, features_val, features_test, Y_train, Y_validation, Y_test, g=1):
    """
    Train and evaluate an RBF SVM classifier.

    Args:
        features_train (np.ndarray): Training feature matrix.
        features_val (np.ndarray): Validation feature matrix.
        features_test (np.ndarray): Test feature matrix.
        Y_train (np.ndarray): Training labels.
        Y_validation (np.ndarray): Validation labels.
        Y_test (np.ndarray): Test labels.
        g (float): Gamma parameter for the RBF kernel (default is 1).

    Returns:
        svm.SVC: Trained RBF SVM model.
    """
    model = svm.SVC(kernel='rbf', gamma=g, decision_function_shape='ovr')
    model.fit(features_train, Y_train)

    print("Number of Support Vectors: ", len(model.support_vectors_))

    labels_train = model.predict(features_train)
    labels_val = model.predict(features_val)
    labels_test = model.predict(features_test)

    print("Accuracy on the Training set: ", accuracy(labels_train, Y_train))
    print("Accuracy on the Validation set: ", accuracy(labels_val, Y_validation))
    print("Accuracy on the Test set: ", accuracy(labels_test, Y_test))

    cm = confusion_matrix(labels_val.astype(int), Y_validation)
    plt.title("Confusion Matrix of the Validation Set")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()

    return model

def minimum_train(features):
    """
    Train a minimum distance classifier by computing the mean of each class.

    Args:
        features (np.ndarray): Feature matrix where each class is assumed to be contiguous.

    Returns:
        np.ndarray: Mean feature vector for each class.
    """
    m = features.shape[0]
    m_class = m // 16
    n = features.shape[1]
    mu = np.zeros((16, n))

    for i in range(16):
        mu[i] = np.mean(features[i*m_class:(i+1)*m_class], axis=0)

    return mu

def md_classifier(features, mu):
    """
    Classify features using the minimum distance classifier.

    Args:
        features (np.ndarray): Feature matrix to classify.
        mu (np.ndarray): Mean feature vectors for each class.

    Returns:
        np.ndarray: Predicted labels.
    """
    m = features.shape[0]
    scores = np.zeros((16, m))

    for i in range(16):
        scores[i] = np.linalg.norm(features - mu[i], axis=1)

    labels = np.argmin(scores, axis=0)

    return labels

def minimum_distance(features_train, features_val, features_test, Y_train, Y_validation, Y_test):
    """
    Train and evaluate a minimum distance classifier.

    Args:
        features_train (np.ndarray): Training feature matrix.
        features_val (np.ndarray): Validation feature matrix.
        features_test (np.ndarray): Test feature matrix.
        Y_train (np.ndarray): Training labels.
        Y_validation (np.ndarray): Validation labels.
        Y_test (np.ndarray): Test labels.
    """
    mu = minimum_train(features_train)

    labels_train = md_classifier(features_train, mu)
    labels_val = md_classifier(features_val, mu)
    labels_test = md_classifier(features_test, mu)

    print("Accuracy on the Training set: ", accuracy(labels_train, Y_train))
    print("Accuracy on the Validation set: ", accuracy(labels_val, Y_validation))
    print("Accuracy on the Test set: ", accuracy(labels_test, Y_test))

    cm = confusion_matrix(labels_val.astype(int), Y_validation)
    plt.title("Confusion Matrix of the Validation Set")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()

def regularization_coeff(X_train, X_validation, X_test, Y_train, Y_validation, Y_test):
    """
    Search for the best regularization coefficient for linear SVM by evaluating different lambda values.

    Args:
        X_train (np.ndarray): Training feature matrix.
        X_validation (np.ndarray): Validation feature matrix.
        X_test (np.ndarray): Test feature matrix.
        Y_train (np.ndarray): Training labels.
        Y_validation (np.ndarray): Validation labels.
        Y_test (np.ndarray): Test labels.
    """
    lamda_trial = [0.01, 0.03, 0.1, 0.3, 1]
    for lamda in lamda_trial:
        print("Evaluating the accuracy for lambda =", lamda)
        linear_svm(X_train, X_validation, X_test, Y_train, Y_validation, Y_test, lamda)
