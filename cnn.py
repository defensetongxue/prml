# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from util.customDataset import CustomMNISTDataset, load_data
import os
from util.model import SimpleCNN
# Load data
X_train, y_train, X_test, y_test = load_data('mnist_all.mat')
# X_train, y_train, X_test, y_test = load_data('mnist_all.mat', n_components=250)


# Create dataset
train_dataset = CustomMNISTDataset(X_train, y_train)
test_dataset = CustomMNISTDataset(X_test, y_test)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load ResNet50 model
model = SimpleCNN()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
losses = []  # List to store loss values
for epoch in range(num_epochs):
    model.train()
    running_loss=0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss+=(loss.item())  # Collect loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(running_loss/len(test_loader))
# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')


# Plot and save the loss graph
os.makedirs('./figure/',exist_ok=True)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.savefig('./figure/cnn.png')
