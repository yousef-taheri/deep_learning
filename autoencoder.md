
# PyTorch Autoencoder for MNIST Handwritten Digit Recognition

## Overview

This project implements an autoencoder using the PyTorch deep learning framework.  An autoencoder is a type of neural network used for unsupervised learning, particularly for dimensionality reduction, feature extraction, and data denoising. This implementation focuses on training an autoencoder to effectively compress and reconstruct images from the MNIST handwritten digit dataset.

The script includes:

*   Data handling with custom datasets and dataloaders.
*   Model definition with an encoder-decoder architecture.
*   Training loop with GPU support.
*   Checkpointing to save and restore training progress.
*   Visualization of training progress with TensorBoard.

## Problem Solved

The main problem addressed here is learning a compact, compressed representation of MNIST images using an autoencoder.  The autoencoder learns to:

1.  **Encode:**  Transform the high-dimensional image data (784 pixels) into a lower-dimensional latent space (128 dimensions in this example). This forces the network to learn the most important features of the data.
2.  **Decode:**  Reconstruct the original image from the compressed representation in the latent space.

By successfully training the autoencoder, we demonstrate:

*   The ability to learn a meaningful representation of the MNIST dataset without explicit labels.
*   The power of neural networks for unsupervised learning tasks.
*   A practical application of PyTorch for building and training deep learning models.

## Dataset

The project uses the MNIST handwritten digit dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is 28x28 pixels in grayscale. The dataset is readily available through the `torchvision.datasets` module.

## Solution Implementation

The solution consists of the following key components:

*   **`MyDataset` Class:** A custom PyTorch `Dataset` class that:
    *   Loads the MNIST images and labels.
    *   Flattens each 28x28 image into a 784-dimensional vector.
    *   Normalizes pixel values to the range \[0, 1] to improve training stability.
*   **`Autoencoder` Class:** A PyTorch `nn.Module` class that defines the autoencoder architecture:
    *   **Encoder:** A linear layer that maps the 784-dimensional input to a 128-dimensional latent space, followed by a ReLU activation function.
    *   **Decoder:** A linear layer that maps the 128-dimensional latent representation back to the 784-dimensional output space, followed by a Sigmoid activation function (to ensure pixel values are between 0 and 1).
*   **Training Loop:**
    *   Iterates over the training dataset in batches using a PyTorch `DataLoader`.
    *   Performs a forward pass through the autoencoder to obtain the reconstructed image.
    *   Calculates the Mean Squared Error (MSE) loss between the original image and the reconstructed image.
    *   Performs a backward pass to compute the gradients.
    *   Updates the model's parameters using Stochastic Gradient Descent (SGD) with momentum.
    *   Logs the training loss to TensorBoard for visualization.
*   **Checkpointing:** The `State` class saves the model's weights, optimizer state, epoch number, and iteration number at regular intervals during training. This allows training to be resumed from a saved checkpoint in case of interruption or for further experimentation.
*   **GPU Support:** Leverages the availability of a GPU (if present) for accelerated training using `torch.device`.

## Instructions

1.  **Prerequisites:**
    *   Python 3.6+
    *   PyTorch (>=1.0)
    *   Torchvision
    *   TensorBoard
    *   NumPy
    *   Pandas (for potential future analysis)

2.  **Installation:**
    ```bash
    pip install torch torchvision tensorboard numpy
    ```

3.  **Running the Script:**

    *   If running on Google Colab, ensure that you are connected to a GPU runtime.  The script will automatically detect and use the GPU if available. It will also automatically mount google drive.
    *   Execute the Python script:
        ```bash
        python your_script_name.py
        ```

4.  **TensorBoard Visualization:**

    *   After running the script, launch TensorBoard in your terminal:
        ```bash
        tensorboard --logdir=runs
        ```
    *   Open the URL provided by TensorBoard in your web browser to visualize the training progress, including the training loss over time.

## Code Structure

*   **`autoencoder.py`:** The main Python script containing the autoencoder implementation, training loop, and data loading logic.

## Key Improvements

*   **Clear organization:** Separated into well-defined functions.
*   **Corrected Dataset Loading:** Ensures data is correctly loaded and preprocessed.
*   **GPU Support:**  Efficient utilization of available GPUs.
*   **Checkpointing:** Allows to restart training from a specific state.

## Potential Extensions

*   Experiment with different autoencoder architectures, such as convolutional autoencoders or variational autoencoders.
*   Evaluate the quality of the learned representations by visualizing the reconstructed images.
*   Use the trained encoder as a feature extractor for downstream tasks, such as image classification.
*   Apply the autoencoder for data denoising by training it to reconstruct clean images from noisy inputs.
*   Explore different loss functions, such as binary cross-entropy.
