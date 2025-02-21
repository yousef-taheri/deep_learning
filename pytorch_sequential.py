# -*- coding: utf-8 -*-
"""
Implements a linear regression model using autograd and then explores neural networks 
using PyTorch's nn module. Includes solutions related to linear layer implementation,
optimization, and sequential model building.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F  # Importing functional interface for nn

# Set device to CUDA if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# Load and preprocess the Boston Housing dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df = pd.read_csv(url, delim_whitespace=True, header=None)
x_mat = df.iloc[:, 0:13].values  # Features (NumPy array)
y_res = df.iloc[:, 13].values    # Target (NumPy array)

# Add a bias (intercept) term to the feature matrix
aug_x = np.concatenate([np.ones((x_mat.shape[0], 1)), x_mat], axis=1)

# Convert NumPy arrays to PyTorch tensors and move to the selected device
x_train_tensor = torch.from_numpy(aug_x).float().to(device)
y_train_tensor = torch.from_numpy(y_res).float().to(device).view(-1, 1) # Reshape target to be [N, 1]

# --- Custom Linear Layer Implementation using autograd ---

class LinearFunction(torch.autograd.Function):
    """
    Implements a linear layer (y = xW^T) with custom forward and backward passes 
    using torch.autograd.Function. This allows for manual control of the gradient calculation.
    """

    @staticmethod
    def forward(ctx, x, w):
        """
        Forward pass: Computes the output y = xW^T.

        Args:
            ctx: Context object to save tensors needed for the backward pass.
            x: Input tensor (N, in_features).
            w: Weight tensor (out_features, in_features).

        Returns:
            Output tensor (N, out_features).
        """
        ctx.save_for_backward(x, w)  # Save x and w for backward pass
        output = x.mm(w.t())  # Matrix multiplication (x @ w.T)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Computes gradients with respect to x and w.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            grad_output: Gradient of the loss with respect to the output (N, out_features).

        Returns:
            Gradients with respect to x and w.
        """
        x, w = ctx.saved_tensors
        grad_x = grad_w = None

        grad_x = grad_output.mm(w)     # Gradient w.r.t. x: dL/dx = dL/dy @ W
        grad_w = grad_output.t().mm(x) # Gradient w.r.t. w: dL/dW = (dL/dy).T @ x

        return grad_x, grad_w


# Verify the implementation using gradcheck

x_check = torch.randn(506, 14, requires_grad=True, dtype=torch.float64, device=device)
w_check = torch.randn(1, 14, requires_grad=True, dtype=torch.float64, device=device)
torch.autograd.gradcheck(LinearFunction.apply, (x_check, w_check)) # Function must be .apply


# --- Linear Regression Training with Custom Linear Layer and Gradient Descent ---
def train_linear_regression(x_train, y_train, w_initial, learning_rate=1e-9, num_epochs=10000, print_every=50):
    """
    Trains a linear regression model using gradient descent and the custom LinearFunction.

    Args:
        x_train: Training data features (N, in_features).
        y_train: Training data target (N, 1).
        w_initial: Initial weight tensor (out_features, in_features).
        learning_rate: Learning rate for gradient descent.
        num_epochs: Number of training epochs.
        print_every: Frequency of printing the loss.
    """
    w = w_initial.clone().detach().requires_grad_(True)  # Create a trainable copy of initial weights
    ctx = None # no need for Context()
    for t in range(num_epochs):
        y_pred = LinearFunction.apply(x_train, w)  # Forward pass using custom linear layer
        loss = (y_pred - y_train).pow(2).mean()    # Mean squared error loss

        if t % print_every == 0:
            print(f"Epoch {t}, Loss: {loss.item()}")

        loss.backward()  # Compute gradients

        with torch.no_grad():
            w -= learning_rate * w.grad # Update weights

        w.grad.zero_()  # Reset gradients to zero after each update


# Initialize weights and train the model
torch.random.manual_seed(3360)
w_initial = torch.randn(1, 14, requires_grad=False, dtype=torch.float, device=device)
train_linear_regression(x_train_tensor, y_train_tensor, w_initial, learning_rate=1e-9, num_epochs=10000)


# --- Neural Network Implementation using torch.nn ---

class MyNeuralNet(nn.Module):
    """
    A simple neural network with one hidden layer, implemented using torch.nn.Module.
    """

    def __init__(self):
        """
        Initializes the layers of the network.
        """
        super().__init__()
        self.hidden = nn.Linear(14, 10) # Linear layer: in_features=14, out_features=10
        self.output = nn.Linear(10, 1) # Linear layer: in_features=10, out_features=1

    def forward(self, x):
        """
        Forward pass: Defines how the input is transformed through the layers.

        Args:
            x: Input tensor (N, in_features).

        Returns:
            Output tensor.
        """
        x = self.hidden(x)      # Pass through the hidden layer
        x = torch.tanh(x)       # Apply tanh activation
        x = self.output(x)      # Pass through the output layer
        return x


# --- Training the Neural Network ---

def train_nn(model, x_train, y_train, learning_rate=1e-3, num_epochs=5000, print_every=100):
    """
    Trains a neural network model.

    Args:
        model: The neural network model to train.
        x_train: Training data features.
        y_train: Training data target.
        learning_rate: Learning rate for the optimizer.
        num_epochs: Number of training epochs.
        print_every: Frequency of printing the loss.
    """
    loss_fn = nn.MSELoss(reduction='mean') # Mean squared error loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) # Stochastic gradient descent optimizer

    for t in range(num_epochs):
        model.train()             # Set the model to training mode
        yhat = model(x_train)     # Forward pass: compute the output
        loss = loss_fn(y_train, yhat) # Compute the loss

        if t % print_every == 0:
            print(f"Epoch {t}, Loss: {loss.item()}")

        optimizer.zero_grad()   # Clear the gradients
        loss.backward()         # Compute the gradients
        optimizer.step()          # Update the parameters


# Initialize and train the neural network
torch.manual_seed(42)
model = MyNeuralNet().to(device)   # Create the model and move it to the device
train_nn(model, x_train_tensor, y_train_tensor, learning_rate=1e-4, num_epochs=5000)


# --- Sequential Model Implementation ---

def create_sequential_model(in_features, hidden_units, out_features):
  """
  Creates a neural network using `nn.Sequential`.

  Args:
      in_features: Number of input features.
      hidden_units: Number of hidden units in the single hidden layer.
      out_features: Number of output features.

  Returns:
      A `nn.Sequential` model.
  """
  model = nn.Sequential(
      nn.Linear(in_features, hidden_units),
      nn.Tanh(),
      nn.Linear(hidden_units, out_features)
  )
  return model

# Create the sequential model
torch.manual_seed(42) # set the random seed.
sequential_model = create_sequential_model(in_features=14, hidden_units=10, out_features=1).to(device)

# --- Training the Sequential Model ---
train_nn(sequential_model, x_train_tensor, y_train_tensor, learning_rate=1e-4, num_epochs=5000)
