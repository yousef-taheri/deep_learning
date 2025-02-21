# PyTorch Linear Regression Implementation

## Overview

This script implements linear regression using the PyTorch deep learning framework. It provides a hands-on example of defining custom forward and backward passes for a linear layer and the Mean Squared Error (MSE) loss function. The script also demonstrates and compares different gradient descent optimization techniques:

*   **Batch Gradient Descent:** Computes the gradient over the entire training dataset.
*   **Stochastic Gradient Descent (SGD):** Computes the gradient using a single randomly selected training example.
*   **Mini-Batch Gradient Descent:** Computes the gradient using a small batch of training examples.

The script also incorporates TensorBoard for visualizing the training and testing loss, allowing for easy monitoring of the model's performance.

## Problem Solved

The primary problem being solved is to train a linear regression model to predict housing prices using the Boston Housing dataset. The script implements the linear regression model and its optimization from scratch, providing a deep understanding of the underlying principles of deep learning.

Specifically, the script addresses the following key aspects:

*   **Linear Model Implementation:**  Defining a linear model using PyTorch's `Function` class, which requires specifying both the forward and backward passes. This exercise reinforces understanding of the computational graph and automatic differentiation.
*   **MSE Loss Function:** Implementing the Mean Squared Error (MSE) loss function and its gradient. This is a fundamental loss function for regression problems.
*   **Gradient Descent Optimization:**  Implementing and comparing different variants of gradient descent to find the optimal weights for the linear model.
*   **Evaluation:**  Splitting the data into training and testing sets to evaluate the model's generalization performance.
*   **Visualization:**  Using TensorBoard to visualize the training and testing loss over time, enabling monitoring of the training process and identifying potential issues such as overfitting or slow convergence.

## Dataset

The script uses the Boston Housing dataset, which contains information about various attributes of houses in the Boston area and their corresponding prices. The dataset is loaded from the UCI Machine Learning Repository.

## Implementation Details

The script includes the following key implementation details:

*   **Custom Linear Layer (`MyLinearFunction`)**:
    *   **Forward Pass**: Computes the output of the linear layer (y = xW'), where x is the input features, W is the weight matrix.
    *   **Backward Pass**: Calculates the gradients of the loss with respect to the input features (x) and the weight matrix (W).

*   **Custom MSE Loss Function (`MyMSELoss`)**:
    *   **Forward Pass**: Computes the Mean Squared Error (MSE) between the predicted values and the true values.
    *   **Backward Pass**: Calculates the gradients of the loss with respect to the predicted values and the true values.

*   **Gradient Descent Variants**:
    *   **Batch Gradient Descent**: Iterates through the entire dataset to compute the gradient.
    *   **Stochastic Gradient Descent**: Uses a single randomly chosen data point per iteration to compute the gradient.
    *   **Mini-Batch Gradient Descent**: Computes the gradient over a small batch of data points in each iteration.

*   **TensorBoard Integration**:
    *   The script uses `torch.utils.tensorboard.SummaryWriter` to log the training and testing loss values during training.
    *   These logs can be visualized using TensorBoard by running the command `tensorboard --logdir=runs` in your terminal and opening the provided URL in your browser.

## Instructions

1.  **Install Dependencies:**
    ```bash
    pip install numpy pandas torch tensorboard
    ```
2.  **Run the Script:**
    ```bash
    python your_script_name.py
    ```
3.  **Visualize with TensorBoard:**
    ```bash
    tensorboard --logdir=runs
    ```
    Open the URL provided by TensorBoard in your browser to view the visualizations.

## Code Structure

The script is organized into the following main sections:

1.  **Data Loading and Preprocessing:** Loads the Boston Housing dataset and prepares the data for training.
2.  **Device Configuration:** Determines if CUDA (GPU) is available and sets the device accordingly.
3.  **Custom Linear Layer and MSE Loss Function:** Defines the custom PyTorch functions for the linear layer and MSE loss.
4.  **Gradient Checking:** Verifies the correctness of the backward pass implementations using `torch.autograd.gradcheck`.
5.  **Training the Model:** Implements and compares different gradient descent methods.
6.  **Training with Validation and TensorBoard:** Trains the model with a validation set and logs the training and testing loss to TensorBoard.

## References

*   **Boston Housing Dataset:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data)
*   **PyTorch Documentation:** [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
*   **TensorBoard Documentation:** [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)


This README provides a comprehensive overview of the script, its purpose, implementation details, and instructions for running and visualizing the results. It's designed to be clear and informative for anyone looking to understand and use the script.
