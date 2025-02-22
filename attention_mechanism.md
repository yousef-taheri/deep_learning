# attention_mechanism.py: Attention Mechanisms for Sentiment Analysis

## Overview

This script implements a basic sentiment analysis model with and without a simple attention mechanism. It uses randomly initialized embeddings for demonstration purposes. The focus is on implementing the base model (Exo 0) and the simple attention mechanism (Exo 1) as described in the document.

## Problem Solved

The script addresses the task of sentiment analysis, which involves determining the overall sentiment (e.g., positive or negative) expressed in a given text. The script explores how attention mechanisms can be used to improve the performance of sentiment analysis models by allowing the model to focus on the most important words or phrases in the text. The two models implemented are:
* Base Model that gets the embedding from a simple mean of all the words
* Attetion Model that weights the words based on the attention of the model.

## Implementation Details

*   **Data Generation:**
    *   The script generates random data to simulate text sequences and sentiment labels.
    *   This allows the script to be run without requiring a specific dataset.
*   **Model Architecture:**
    *   **Base Model:** This model represents a text sequence as the average of the embeddings of its words. A linear layer and LogSoftmax are then used to classify the sentiment.
    *   **Simple Attention Model:** This model extends the base model by incorporating an attention mechanism.
        *   The attention mechanism learns to assign weights to each word in the sequence, indicating its importance to the sentiment.
        *   The weighted average of the word embeddings is then used as the text representation.
        *   A linear layer and LogSoftmax are then used to classify the sentiment.
*   **Training:**
    *   The models are trained using the Negative Log-Likelihood Loss (NLLLoss) function.
    *   The Adam optimizer is used to update the model's parameters.
*   **TensorBoard Logging:**
    *   The script logs the training loss and test accuracy to TensorBoard.

## Instructions

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
    python attention_mechanism.py
    ```
4.  **Monitor with TensorBoard:**
    ```bash
    tensorboard --logdir=runs
    ```
    Open the URL provided to view the training progress.

## Potential Extensions

*   Implement more sophisticated attention mechanisms, such as self-attention or multi-head attention.
*   Use pre-trained word embeddings to improve the model's performance.
*   Evaluate the model on a real-world sentiment analysis dataset.
*   Explore the use of contextual embeddings from models like BERT or RoBERTa.
*   Implement the other suggestions found in the PDF, like trying with different models or experimenting with the entropy of the attention.
