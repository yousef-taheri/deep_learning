# self_attention.py: Self-Attention Mechanisms for Sentiment Analysis

## Overview

This script implements a self-attention mechanism for sentiment analysis. It aims to classify text sequences into positive or negative sentiment categories. The script implements the base model (Exo 0) and positional embeddings (Exo 2) as described in the document.

**Important Note:** Due to the limited details in the provided document, the script makes certain assumptions. The core functionality is implemented, but further refinement and experimentation may be needed for optimal results.

## Problem Solved

The script addresses the problem of sentiment analysis using self-attention. Self-attention allows the model to weigh the importance of different words in a sentence when determining the overall sentiment. The goal is to improve the accuracy of sentiment classification by enabling the model to focus on the most relevant words.

## Implementation Details

*   **Data Generation:**
    *   The script generates random data to simulate text sequences and sentiment labels.  This allows for execution without requiring a specific dataset.
*   **Positional Encoding (Exo 2):**
    *   The `PositionalEncoding` class adds positional information to the word embeddings, allowing the model to attend to the order of words in the sequence.
*   **Self-Attention Model (Exo 0 & Exo 2):**
    *   The `SelfAttentionModel` class implements the self-attention mechanism:
        *   **Embedding Layer:** Maps word IDs to dense vector representations.
        *   **Positional Encoding:** Adds positional information to the word embeddings.
        *   **Attention Layers:** Self-attention is applied in the attention layers.
        *   **Residual connection**: A residual connection has been added.
        *   **Normalization function**: Normalization with ReLU has been applied.
        *   **Linear Layer:** Maps the attention-weighted representation to class scores.
        *   **LogSoftmax Layer:** Converts the class scores to probabilities.
*   **Training:**
    *   The model is trained using the Negative Log-Likelihood Loss (NLLLoss) function.
    *   The Adam optimizer is used to update the model's parameters.
*   **TensorBoard Logging:**
    *   The script logs the training loss and test accuracy to TensorBoard.

## Usage Instructions

1.  **Prerequisites:**
    *   Python 3.6+
    *   PyTorch (>=1.0)
    *   NumPy
    *   TensorBoard
2.  **Install Dependencies:**
    ```bash
    pip install torch numpy tensorboard
    ```
3.  **Run the Script:**
    ```bash
    python self_attention.py
    ```
4.  **Monitor with TensorBoard:**
    ```bash
    tensorboard --logdir=runs
    ```
    Open the URL provided to view the training progress.

## Assumptions and Limitations

*   **Simplified Data:** The script uses randomly generated data for demonstration purposes.
*   **Basic Implementation:** The self-attention mechanism is a basic implementation and may not achieve state-of-the-art results.
*   **Hyperparameters:** Hyperparameters are not tuned.
*   **Missing Details:** The document lacks precise details on certain aspects of the implementation.

## Potential Extensions

*   Implement more sophisticated attention mechanisms, such as multi-head attention.
*   Use pre-trained word embeddings to improve the model's performance.
*   Evaluate the model on a real-world sentiment analysis dataset (e.g., IMDB).
*   Experiment with different values for the number of self-attention layers.
*   Fine-tune the model and do hyperparameter search.
* Implement the clustering token that helps generate the text.
