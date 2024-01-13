# customDataset.py
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA

class CustomMNISTDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(filename, n_components=None):
    dataset = loadmat(filename)
    X_train = np.vstack([dataset[f'train{i}'] for i in range(10)])
    y_train = np.hstack([np.full(dataset[f'train{i}'].shape[0], i) for i in range(10)])
    X_test = np.vstack([dataset[f'test{i}'] for i in range(10)])
    y_test = np.hstack([np.full(dataset[f'test{i}'].shape[0], i) for i in range(10)])

    if n_components is not None:
        print("Applying PCA to reduce dimensions to", n_components)
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Reshape and scale
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255

    return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test)
