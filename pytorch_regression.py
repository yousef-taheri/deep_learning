# -*- coding: utf-8 -*-
"""
This script implements linear regression using PyTorch, following the exercises outlined in the provided PDF.
It includes custom forward and backward passes for both the linear layer and the MSE loss function.
The script also demonstrates batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.
Finally, it incorporates TensorBoard for visualizing training and testing loss.

"""

import numpy as np
import pandas as pd
import torch
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter

# 1. Data Loading and Preprocessing

# Load the Boston Housing dataset from the UCI archive
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df = pd.read_csv(url, delim_whitespace=True, header=None)

# Extract features (xMat) and target variable (yRes)
xMat = df.iloc[:, 0:13]
yRes = df.iloc[:, 13]

# Add a bias term (intercept) to the feature matrix.  This corresponds to the 'b' parameter in y = xW' + b
augX = np.concatenate([np.ones((xMat.shape[0], 1)), xMat.values], axis=1)  # Add a column of ones for the bias

# 2. Device Configuration
# Determine if CUDA (GPU) is available and set the device accordingly.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}") # added print to know which device is used

# Convert data to PyTorch tensors and move them to the selected device.
x_train_tensor = torch.from_numpy(augX).float().to(device)  # Feature tensor
y_train_tensor = torch.from_numpy(yRes.values).float().to(device).view(-1, 1)  # Target tensor, reshaped to (n_samples, 1)

# 3. Custom Linear Layer and MSE Loss Function (Following the "Implémentation" section of the PDF)
# These classes implement the forward and backward passes for our linear layer and loss function.

class MyLinearFunction(Function):
    """
    Custom linear layer function (y = xW'), as described in the PDF.
    This class implements the forward and backward passes.
    """
    @staticmethod
    def forward(ctx, x, w):
        """
        Forward pass: computes the output (y = xW').

        Args:
            ctx: Context object to save tensors for the backward pass.
            x (torch.Tensor): Input tensor of shape (N, d), where N is the number of samples and d is the number of features.
            w (torch.Tensor): Weight tensor of shape (1, d),

        Returns:
            torch.Tensor: Output tensor of shape (N, 1).
        """
        ctx.save_for_backward(x, w)  # Save x and w for the backward pass
        return x.mm(w.t())  # Matrix multiplication: x * w.transpose()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: computes the gradients of the loss with respect to x and w.

        Args:
            ctx: Context object containing the saved tensors from the forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Gradients with respect to x and w.
        """
        x, w = ctx.saved_tensors  # Retrieve x and w from the context
        grad_x = grad_w = None  # Initialize gradients to None

        # Calculate gradients (if needed). If x or w doesn't require gradient, the corresponding grad_x or grad_w will remain None.
        grad_x = grad_output.mm(w)  # Gradient with respect to x
        grad_w = grad_output.t().mm(x)  # Gradient with respect to w

        return grad_x, grad_w  # Return the gradients
mylinear = MyLinearFunction.apply

class MyMSELoss(Function):
    """
    Custom MSE (Mean Squared Error) loss function.
    """
    @staticmethod
    def forward(ctx, input, response):
        """
        Forward pass: computes the MSE loss.

        Args:
            ctx: Context object to save tensors for the backward pass.
            input (torch.Tensor): Predicted values (y_hat).
            response (torch.Tensor): True values (y).

        Returns:
            torch.Tensor: MSE loss.
        """
        ctx.save_for_backward(input, response)  # Save input and response for the backward pass
        output = (input - response)**2  # Calculate squared differences
        return output  # Return the MSE loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: computes the gradients of the loss with respect to the input and response.

        Args:
            ctx: Context object containing the saved tensors from the forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Gradients with respect to input and response.
        """
        input, response = ctx.saved_tensors  # Retrieve input and response from the context
        g_inp = grad_output * (2 * (input - response))  # Gradient with respect to input
        g_res = grad_output * (2 * (response - input))  # Gradient with respect to response
        return g_inp, g_res  # Return the gradients
mse = MyMSELoss.apply

# 4. Gradient Checking (Using torch.autograd.gradcheck - as suggested in the PDF)
# This verifies the correctness of the backward pass implementations.

#Checking the linear Layer implementation
x_check = torch.randn(5, 14, requires_grad=True, dtype=torch.float64, device=device) #reduced size for testing
w_check = torch.randn(1, 14, requires_grad=True, dtype=torch.float64, device=device) #reduced size for testing
torch.autograd.gradcheck(mylinear, (x_check, w_check)) #gradcheck for custom Linear Layer

#Checking MSE Implementation

y_true_check=torch.randn(5, 1, requires_grad=True, dtype=torch.float64 ,device=device) #reduced size for testing
y_pred_check=torch.randn(5, 1, requires_grad=True, dtype=torch.float64 ,device=device) #reduced size for testing
torch.autograd.gradcheck(mse,(y_true_check, y_pred_check)) #gradcheck for MSE loss

# 5. Training the Model (Different Gradient Descent Methods)
# This section demonstrates different gradient descent approaches: Batch, Stochastic, and Mini-batch.

# A. Batch Gradient Descent
print("\nStarting Batch Gradient Descent...")
torch.manual_seed(3360)  # For reproducibility
w_batch = torch.randn(1, 14, requires_grad=False, dtype=torch.float, device=device)  # Initialize weights (no grad yet)
learning_rate_batch = 1e-6
ctx = Context() # Context needed for forward and backward passes
ctx_mse = Context() # Context needed for forward and backward passes

for t in range(10000):
    # Forward pass
    y_pred_batch = mylinear(x_train_tensor, w_batch)

    # Loss calculation using MSE
    loss_batch = mse(y_pred_batch, y_train_tensor)
    mean_loss_batch = torch.mean(loss_batch)

    #Calculate gradients
    y_grad_batch, _ = MyMSELoss.backward(ctx_mse,torch.ones(506, 1, requires_grad=False, dtype=torch.float ,device=device)) # create backward gradient
    _, grad_w_batch= MyLinearFunction.backward(ctx, y_grad_batch)

    if t % 50 == 0:
        print(f"Epoch {t}, Mean Loss: {mean_loss_batch.item()}")

    # Update weights (Manual gradient update because requires_grad=False)
    w_batch -= learning_rate_batch * grad_w_batch


# B. Stochastic Gradient Descent (SGD)
print("\nStarting Stochastic Gradient Descent...")
torch.manual_seed(3360)
w_sgd = torch.randn(1, 14, requires_grad=False, dtype=torch.float, device=device)
learning_rate_sgd = 1e-6
ctx_sgd = Context() # Context needed for forward and backward passes
ctx_mse_sgd = Context() # Context needed for forward and backward passes


for t in range(500):
    rand_ind = torch.randint(x_train_tensor.size()[0], size=(1,)).item()  # Randomly select one index
    sampled_x = x_train_tensor[rand_ind].view(1, -1)  # Get the sample
    sampled_y = y_train_tensor[rand_ind].view(1, -1)

    # Forward pass
    y_pred_sgd = mylinear(sampled_x, w_sgd)

    # Loss calculation
    loss_sgd = mse(y_pred_sgd, sampled_y)
    mean_loss_sgd = torch.mean(loss_sgd)

    #Calculate gradients
    y_grad_sgd, _ = MyMSELoss.backward(ctx_mse_sgd,torch.ones(1, 1, requires_grad=False, dtype=torch.float ,device=device)) # create backward gradient
    _, grad_w_sgd= MyLinearFunction.backward(ctx_sgd, y_grad_sgd)

    if t % 50 == 0:
        print(f"Epoch {t}, Loss: {mean_loss_sgd.item()}")

    # Update weights (Manual gradient update because requires_grad=False)
    w_sgd -= learning_rate_sgd * grad_w_sgd


# C. Mini-Batch Gradient Descent
print("\nStarting Mini-Batch Gradient Descent...")
torch.manual_seed(3360)
w_minibatch = torch.randn(1, 14, requires_grad=False, dtype=torch.float, device=device)
learning_rate_minibatch = 1e-6
batch_size = 50
ctx_minibatch = Context() # Context needed for forward and backward passes
ctx_mse_minibatch = Context() # Context needed for forward and backward passes

for t in range(500):
    train_len = x_train_tensor.size()[0]
    # Create a permutation of indices
    indices = torch.randperm(train_len)

    # Iterate through batches
    for i in range(0, train_len, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_x = x_train_tensor[batch_indices]
        batch_y = y_train_tensor[batch_indices]

        # Forward pass
        y_pred_minibatch = mylinear(batch_x, w_minibatch)

        # Loss calculation
        loss_minibatch = mse(y_pred_minibatch, batch_y)
        mean_loss_minibatch = torch.mean(loss_minibatch)

        #Calculate gradients
        y_grad_minibatch, _ = MyMSELoss.backward(ctx_mse_minibatch,torch.ones(len(batch_x), 1, requires_grad=False, dtype=torch.float ,device=device)) # create backward gradient
        _, grad_w_minibatch= MyLinearFunction.backward(ctx_minibatch, y_grad_minibatch)


        if t % 50 == 0 and i ==0:
            print(f"Epoch {t}, Batch Start {i}, Loss: {mean_loss_minibatch.item()}")

        # Update weights (Manual gradient update because requires_grad=False)
        w_minibatch -= learning_rate_minibatch * grad_w_minibatch

# 6. Training with Validation and TensorBoard (as suggested in the PDF)
# This section splits the data into training and testing sets and uses TensorBoard to visualize the loss.

print("\nStarting Training with Validation and TensorBoard...")
train_len = x_train_tensor.size()[0]
indices = torch.randperm(train_len)  # Permute the data
break_ind = int(np.floor(train_len * 0.7))  # 70/30 split
train_indices = indices[:break_ind]
test_indices = indices[break_ind:]

x_train = x_train_tensor[train_indices]
y_train = y_train_tensor[train_indices]
x_test = x_train_tensor[test_indices]  #Corrected it x_test_tensor to x_train_tensor.
y_test = y_train_tensor[test_indices]

writer = SummaryWriter()  # Create a TensorBoard writer

torch.manual_seed(3360)
w_tb = torch.randn(1, 14, requires_grad=False, dtype=torch.float, device=device)
learning_rate_tb = 1e-6
ctx_tb = Context() # Context needed for forward and backward passes
ctx_mse_tb = Context() # Context needed for forward and backward passes

for t in range(5000):
    # Forward pass for training data
    y_pred_train = mylinear(x_train, w_tb)

    # Loss calculation for training data
    loss_train = mse(y_pred_train, y_train)
    mean_loss_train = torch.mean(loss_train)

    #Calculate gradients
    y_grad_train, _ = MyMSELoss.backward(ctx_mse_tb,torch.ones(len(x_train), 1, requires_grad=False, dtype=torch.float ,device=device)) # create backward gradient
    _, grad_w_train= MyLinearFunction.backward(ctx_tb, y_grad_train)

    # Forward pass for testing data
    y_pred_test = mylinear(x_test, w_tb)

    # Loss calculation for testing data
    loss_test = mse(y_pred_test, y_test)
    mean_loss_test = torch.mean(loss_test)


    # Write losses to TensorBoard
    writer.add_scalars('Loss/compare2', {'Train': mean_loss_train.item(), 'Test': mean_loss_test.item()}, t)

    # Update weights (Manual gradient update because requires_grad=False)
    w_tb -= learning_rate_tb * grad_w_train

writer.close()  # Close the TensorBoard writer

print("Training complete.  Run 'tensorboard --logdir=runs' in your terminal and open the provided URL in your browser to view the TensorBoard visualizations.")
