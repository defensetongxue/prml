import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

import json
import os
def load_data(filename, n_components=None):
    """
    Load the dataset and optionally apply PCA dimensionality reduction.
    
    :param filename: Filename of the dataset.
    :param n_components: Number of dimensions after PCA reduction. If None, PCA is not applied.
    :return: Training data, training labels, testing data, testing labels.
    """
    dataset = loadmat(filename)
    X_test = dataset['test0']
    y_test = np.zeros(X_test.shape[0])

    for i in range(1, 10):
        X_test = np.vstack((X_test, dataset['test' + str(i)]))
        y_test = np.hstack((y_test, np.full(dataset['test' + str(i)].shape[0], i)))

    X_train = dataset['train0']
    y_train = np.zeros(X_train.shape[0])

    for i in range(1, 10):
        X_train = np.vstack((X_train, dataset['train' + str(i)]))
        y_train = np.hstack((y_train, np.full(dataset['train' + str(i)].shape[0], i)))

    # Apply PCA dimensionality reduction
    if n_components is not None:
        print("Down the dim with pca to {}".format(str(n_components)))
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, y_train, X_test, y_test

def update_results_json(dim, accuracy, file_path='./experiments/result_tree.json'):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump({}, file)

    with open(file_path, 'r') as file:
        results = json.load(file)

    results[dim] = accuracy

    with open(file_path, 'w') as file:
        json.dump(results, file)
