# regularization_mnist.py: Exploring Regularization Techniques on MNIST

## Overview

This script explores various regularization techniques commonly used in deep learning to improve the generalization performance of neural networks. It focuses on training a simple feedforward neural network on the MNIST dataset and observing the effects of:

*   L1 and L2 Regularization
*   Dropout
*   Batch Normalization
*   Layer Normalization

The script provides a practical demonstration of how these techniques can be implemented in PyTorch and how they impact the training process and model performance.

## Problem Solved

The script addresses the problem of overfitting in neural networks. Overfitting occurs when a model learns the training data too well, resulting in poor performance on unseen data. Regularization techniques are used to combat overfitting by:

*   **Simplifying the model:** Encouraging the model to learn simpler, more generalizable representations.
*   **Reducing the variance of the model:** Making the model less sensitive to the specific details of the training data.

## Dataset

The script uses the MNIST handwritten digit dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is 28x28 pixels in grayscale.  The dataset is loaded using `torchvision.datasets.MNIST`. Only 5 percent of the training data is used.

## Implementation Details

*   **Model Architecture:**
    *   The script implements a simple feedforward neural network with three linear layers and ReLU activations.
    *   The network includes options for applying Dropout, BatchNorm, and LayerNorm.
*   **Regularization Techniques:**
    *   **L1 and L2 Regularization:** L1 and L2 regularization are implemented by adding a penalty term to the loss function that is proportional to the sum of the absolute values (L1) or the sum of the squares (L2) of the model's weights.
    *   **Dropout:** Dropout is implemented by randomly setting a fraction of the neurons in each layer to zero during training. This helps to prevent the model from relying too heavily on any single neuron.
    *   **Batch Normalization:** BatchNorm is implemented by normalizing the activations of each layer within a mini-batch. This helps to stabilize the training process and allows for higher learning rates.
    *   **Layer Normalization:** LayerNorm is similar to BatchNorm, but it normalizes the activations across the features within a single sample rather than across the samples within a mini-batch. This can be useful when batch sizes are small or when dealing with recurrent neural networks.
*   **Training:**
    *   The model is trained using the CrossEntropyLoss function.
    *   The Adam optimizer is used to update the model's parameters.
    *   The training loss and test accuracy are logged to the console and TensorBoard.
*   **TensorBoard Logging:**
    *   The script logs the training loss, weights, and gradients of the linear layers to TensorBoard.

## Instructions

1.  **Prerequisites:**
    *   Python 3.6+
    *   PyTorch (>=1.0)
    *   Torchvision
    *   TensorBoard
    *   NumPy
2.  **Install Dependencies:**
    ```bash
    pip install torch torchvision numpy tensorboard
    ```
3.  **Run the Script:**
    ```bash
    python regularization_mnist.py
    ```
4.  **Monitor with TensorBoard:**
    ```bash
    tensorboard --logdir=runs
    ```
    Open the URL provided to view the training progress.

## Potential Extensions

*   Experiment with different hyperparameter values for the regularization techniques.
*   Compare the performance of different combinations of regularization techniques.
*   Implement data augmentation to further improve generalization performance.
*   Evaluate the model on a separate validation dataset to tune the hyperparameters.
*   Visualize the learned weights and activations of the model to gain insights into how the regularization techniques are affecting the model's behavior.
*  Implement early stopping.
