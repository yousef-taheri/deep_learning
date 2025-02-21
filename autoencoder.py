# -*- coding: utf-8 -*-
Autoencoder Implementation with MNIST Dataset

This script implements an autoencoder for the MNIST dataset using PyTorch.
It demonstrates:

-   Loading and preparing the MNIST dataset using torchvision.
-   Defining a custom dataset class.
-   Implementing an autoencoder model with linear layers.
-   Training the autoencoder with GPU support.
-   Checkpointing the model for resuming training.
-   Using TensorBoard for visualization.

The implementation follows the guidelines and exercises outlined in the provided autoencoder.md file.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torch.nn as nn
import os
from google.colab import drive  # Only needed in Colab
from torch.utils.tensorboard import SummaryWriter

# 1. Dataset Loading and Preparation
# Load the MNIST dataset using torchvision.  It downloads automatically.
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor()) #download set to true, transform added

# 2. Custom Dataset Class (as described in the PDF)
# This class adapts the MNIST dataset to provide image vectors and labels.
class MyDataset(Dataset):
    def __init__(self, mnist_dataset):
        """
        Initializes the dataset with MNIST data.

        Args:
            mnist_dataset: The MNIST dataset object from torchvision.
        """
        self.data = mnist_dataset.data
        self.targets = mnist_dataset.targets
        self.len = len(self.data)

    def __getitem__(self, index):
        """
        Retrieves an item (image, label) from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image (as a flattened tensor) and the label.
        """
        image = self.data[index].float() / 255.0  # Normalize pixel values to [0, 1]
        image = image.view(-1)  # Flatten the image into a vector (28*28)
        label = self.targets[index]
        return image, label

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return self.len

# 3. Autoencoder Model (as described in the PDF)
# This model consists of an encoder and a decoder with linear layers.
class Autoencoder(nn.Module):
    def __init__(self):
        """
        Initializes the Autoencoder model.
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(128, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 28*28).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 28*28).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 4. State Class for Checkpointing (as described in the PDF)
# This class stores the model, optimizer, epoch, and iteration for checkpointing.
class State:
    def __init__(self, model, optim, epoch=0, iteration=0): #add epoch and interation init to zero
        """
        Initializes the State object.

        Args:
            model: The model to be saved.
            optim: The optimizer to be saved.
            epoch (int): The starting epoch number.
            iteration (int): The starting iteration number.
        """
        self.model = model
        self.optim = optim
        self.epoch = epoch
        self.iteration = iteration

# 5. Training Setup
# Determine if CUDA (GPU) is available and set the device accordingly.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Mount Google Drive (Only needed if running on Google Colab)
try:
    drive.mount('/content/gdrive')
    IN_COLAB = True
except:
    IN_COLAB = False

# Create the dataset and data loader.
minist_dataset = MyDataset(mnist_trainset)
BATCH_SIZE = 150
train_loader = DataLoader(minist_dataset, shuffle=True, batch_size=BATCH_SIZE)

# Define the loss function and learning rate.
loss_func = nn.MSELoss()  # Mean Squared Error loss
lr = 1e-3  # Increased learning rate to 1e-3 for faster convergence. Original was 1e-4
momentum = 0.9
ITERATIONS = 1000

# 6. Checkpointing (as described in the PDF)
# Define the path to save the model.

model_save_name = 'autoencod.pt'
if IN_COLAB:
    savepath = os.path.join("/content/gdrive/My Drive/Colab Notebooks/saved Model", model_save_name)
else:
    savepath = model_save_name  # Save in the current directory if not in Colab


# Load the model from the checkpoint if it exists.
if os.path.isfile(savepath):
    with open(savepath, "rb") as fp:
        state = torch.load(fp, map_location=device)  # Load to the correct device
    autoencoder = state.model
    optim = state.optim
    print("Using previous model from epoch:", state.epoch)
else:
    autoencoder = Autoencoder().to(device) #model to device before creating the optimizer.
    optim = torch.optim.SGD(params=autoencoder.parameters(), lr=lr, momentum=momentum)
    state = State(autoencoder, optim)
    print("Initializing new model")

# 7. Training Loop
# Create a TensorBoard writer.
writer = SummaryWriter()

# Start the training loop.
for epoch in range(state.epoch, ITERATIONS):
    # Define 'Losses' to calculate the average loss in each epoch.
    losses = []

    for x, _ in train_loader:
        # Zero the gradients.
        state.optim.zero_grad()

        # Move data to the device.
        x = x.to(device)

        # Forward pass.
        xhat = autoencoder(x)

        # Calculate the loss.
        loss = loss_func(xhat, x)

        # Backward pass and optimization.
        loss.backward()
        state.optim.step()

        # Record the loss.
        losses.append(loss.item())

        # Update the iteration counter.
        state.iteration += 1

    # Calculate the average loss for the epoch.
    avg_loss = sum(losses) / len(losses)

    # Save the model checkpoint.
    with open(savepath, "wb") as fp:
        state.epoch = epoch + 1
        state.model = autoencoder
        state.optim = optim #save the optimizer
        torch.save(state, fp)

    # Print the average loss and epoch number.
    print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f}")

    # Write the loss to TensorBoard.
    writer.add_scalar('Loss/train', avg_loss, epoch)

# Close the TensorBoard writer.
writer.close()

print("Training complete.  Run 'tensorboard --logdir=runs' in your terminal and open the provided URL in your browser to view the TensorBoard visualizations.")
