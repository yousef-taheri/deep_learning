# -*- coding: utf-8 -*-
"""regularization_mnist.py: Regularization Techniques on MNIST

This script explores various regularization techniques on the MNIST dataset
using a simple feedforward neural network. The focus is on L1/L2
regularization, Dropout, BatchNorm, and LayerNorm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data Loading and Preprocessing
# MNIST dataset
transform = transforms.ToTensor()

# Full training dataset
full_dataset = datasets.MNIST(root='/data', train=True, download=True, transform=transform)

# Take only 5% of the training data
subset_size = int(0.05 * len(full_dataset))
train_dataset = Subset(full_dataset, range(subset_size))
test_dataset = datasets.MNIST(root='/data', train=False, download=True, transform=transform)

# Data loaders
batch_size = 300
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. Model Definition
class SimpleNN(nn.Module):
    def __init__(self, dropout_rate=0.0, use_batchnorm=False, use_layernorm=False, l1_lambda=0.0, l2_lambda=0.0): # l1 and l2 lambdas
        super(SimpleNN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        self.l1_lambda = l1_lambda # l1 lambda for store. This is only required for regularization
        self.l2_lambda = l2_lambda # l2 lambda for store. This is only required for regularization
        # Linear layers
        self.fc1 = nn.Linear(28 * 28, 100)
        self.bn1 = nn.BatchNorm1d(100) if use_batchnorm else nn.Identity() # Use nn.Identity for conditional
        self.ln1 = nn.LayerNorm(100) if use_layernorm else nn.Identity() # Use nn.Identity for conditional
        self.dropout1 = nn.Dropout(dropout_rate) # use a parameter.

        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100) if use_batchnorm else nn.Identity()
        self.ln2 = nn.LayerNorm(100) if use_layernorm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(100, 100)
        self.bn3 = nn.BatchNorm1d(100) if use_batchnorm else nn.Identity()
        self.ln3 = nn.LayerNorm(100) if use_layernorm else nn.Identity()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(100, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = self.relu(self.bn1(self.ln1(self.fc1(x)))) # Batchnorm/layernorm before activation
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.ln2(self.fc2(x))))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.ln3(self.fc3(x))))
        x = self.dropout3(x)

        x = self.fc4(x) # Linear layer
        return x

# 4. Training Setup
# Hyperparameters
learning_rate = 0.001
num_epochs = 10 # Reduced for demonstration

# Model instantiation (choose your configuration)
model = SimpleNN(dropout_rate=0.5, use_batchnorm=False, use_layernorm=False,l1_lambda=0.0001,l2_lambda=0.0001).to(device) # All the flags can be tested
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard setup
writer = SummaryWriter()

# Store gradients (as requested in the PDF instructions)
def store_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        layer.weight.register_hook(store_grad(layer.weight))

# 5. Training Loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters()) # calculate L1 norm
        loss = loss + model.l1_lambda * l1_norm # add l1 norm with its corresponding lambda

        # L2 regularization
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()) # calculate L2 norm
        loss = loss + model.l2_lambda * l2_norm # add l2 norm with its corresponding lambda


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

    # 6. Evaluation
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # 7. TensorBoard Logging (Selected Examples)
    # log the gradients into the tensorboard
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            if hasattr(layer.weight, 'grad') and layer.weight.grad is not None: # Check if gradient exists
                writer.add_histogram(f'{name}.weight_grad', layer.weight.grad, epoch) # Add histogram
            writer.add_histogram(f'{name}.weight', layer.weight, epoch) # Add histogram

    # log loss
    writer.add_scalar('Loss/train', loss.item(), epoch) # Log loss
writer.close()

print("Finished Training, check TensorBoard")
