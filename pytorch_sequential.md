
# PyTorch Linear Regression and Neural Network Implementation

## Overview

This project demonstrates linear regression and neural network implementations using PyTorch. It includes a custom linear layer implementation using `torch.autograd.Function`, gradient descent optimization, and neural network construction using `torch.nn.Module` and `torch.nn.Sequential`. The project uses the Boston Housing dataset for training and evaluation.

## Problem Statement

The primary goal is to predict housing prices in the Boston area using machine learning models implemented in PyTorch. The project explores:

1.  **Linear Regression with Autograd:**  Implementing a linear regression model from scratch using PyTorch's automatic differentiation (`autograd`) to calculate gradients. This provides a deep understanding of the backpropagation process.

2.  **Neural Network with `torch.nn`:** Building and training a neural network with one hidden layer using PyTorch's `nn` module. This demonstrates the ease of constructing and training neural networks with PyTorch's built-in functionalities.

3.  **Sequential Model Construction:** Creating a neural network using `torch.nn.Sequential` for a more concise and organized model definition.

## Dataset

The project uses the [Boston Housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data) from the UCI Machine Learning Repository. The dataset contains information on housing values in suburbs of Boston.

*   **Features:** 13 features including crime rate, proportion of residential land zoned, accessibility to radial highways, etc.
*   **Target:** Median value of owner-occupied homes in $1000s.

## Implementation Details

### 1. Linear Regression with Custom Autograd Function

*   A custom `LinearFunction` is implemented using `torch.autograd.Function` to define the forward and backward passes for a linear layer. This provides a hands-on understanding of gradient calculation.
*   The `forward` method calculates the linear transformation `y = xW^T`.
*   The `backward` method calculates the gradients with respect to the input `x` and weight matrix `W`.
*   The implementation is verified using `torch.autograd.gradcheck` to ensure the correctness of the gradient calculation.
*   Gradient descent is used to train the linear regression model.

### 2. Neural Network with `torch.nn.Module`

*   A simple neural network `MyNeuralNet` is implemented using `torch.nn.Module`.
*   The network consists of one hidden layer with a `tanh` activation function and an output layer.
*   `torch.nn.MSELoss` is used as the loss function.
*   `torch.optim.SGD` is used as the optimizer.
*   The model is trained using a standard training loop.

### 3. Sequential Model Construction

*   A neural network is created using `torch.nn.Sequential` for a more compact and readable model definition.
*   The `create_sequential_model` function encapsulates the model construction process.

## Files

*   `tp2_pytorch.py`: The main Python script containing the linear regression and neural network implementations.

## Dependencies

*   Python 3.x
*   PyTorch
*   NumPy
*   Pandas

You can install the dependencies using pip:

```bash
pip install torch numpy pandas
```

## Usage

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Run the main script:

    ```bash
    python tp2_pytorch.py
    ```

The script will:

*   Download and preprocess the Boston Housing dataset.
*   Train a linear regression model using custom autograd and gradient descent.
*   Train a neural network using `torch.nn.Module` and `torch.optim.SGD`.
*   Train a sequential model using `torch.nn.Sequential`.
*   Print the loss during training for each model.

## Results

The script will output the training loss at specified intervals for each model. The goal is to minimize the loss, indicating that the model is learning to predict housing prices accurately.

## Further Exploration

*   Experiment with different learning rates, batch sizes, and network architectures.
*   Implement regularization techniques to prevent overfitting.
*   Evaluate the models on a separate test dataset.
*   Compare the performance of the custom autograd implementation with the built-in PyTorch functionalities.

## License

This project is licensed under the [MIT License](LICENSE).
