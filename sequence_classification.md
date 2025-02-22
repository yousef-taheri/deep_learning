## Overview

This script implements a Recurrent Neural Network (RNN) to classify temperature sequences and identify the corresponding city.  It utilizes PyTorch for model definition, training, and evaluation.

## Problem Solved

The script aims to build a model that, given a sequence of temperature data, can accurately determine which city that temperature data belongs to. This involves training an RNN to recognize patterns and features within the temperature sequences that are unique to each city.

## Dataset

The script uses a dataset of temperature readings for multiple cities. Each row represents a timestamp, and each column represents the temperature reading for a specific city at that timestamp.

## Implementation Details

*   **Data Preprocessing:**
    *   Missing values are filled with the mean temperature for each city.
    *   Temperature data is normalized to the range \[0, 1].
*   **RNN Model:**
    *   An RNN model is implemented using PyTorch's `nn.Module`.
    *   The model consists of a linear layer, an RNN layer, and a softmax output layer.
*   **Training:**
    *   The model is trained using the CrossEntropyLoss function.
    *   The Adam optimizer is used to update the model's parameters.
    *   Training progress is monitored using TensorBoard.

## Usage Instructions

1.  **Install Dependencies:**
    ```bash
    pip install torch pandas numpy tensorboard
    ```
2.  **Upload Data:** Upload the `tempAMAL_train.csv` file when prompted by the script.
3.  **Run the Script:**
    ```bash
    python rnn_cla.py
    ```
4.  **Monitor with TensorBoard:**
    ```bash
    tensorboard --logdir=runs
    ```
    Open the URL provided to view the training progress.

## Key Improvements

*   Correct CrossEntropyLoss to address the classification problem
*   Correct Normalization to address the classification problem
*   Clear comments and explanations.
*   Added input dimension to the RNN model

## Potential Extensions

*   Experiment with different RNN architectures (e.g., LSTM, GRU).
*   Explore different data preprocessing techniques.
*   Evaluate the model on a separate test dataset.
*   Implement techniques to handle variable-length sequences.
