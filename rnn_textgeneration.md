# rnn_textgeneration.py: RNN for Trump Speech Generation

## Overview

This script implements a Recurrent Neural Network (RNN) to generate text mimicking the style of Donald Trump's speeches.  It utilizes PyTorch for model definition, training, and text generation.

## Problem Solved

The script aims to train a model that can generate realistic-sounding Trump-esque text.  This involves:

1.  Training an RNN on a corpus of Trump's speeches.
2.  Using the trained model to generate new text sequences.

## Dataset

The script requires a text file containing Trump's speeches (`trump_full_speech.txt`).

## Implementation Details

*   **Text Preprocessing:**
    *   Text is normalized to remove non-standard characters.
    *   Characters are mapped to numerical IDs.
*   **RNN Model:**
    *   An RNN model is implemented using PyTorch's `nn.Module`.
    *   The model includes an embedding layer, an RNN layer, and a softmax output layer.
*   **Training:**
    *   The model is trained using the CrossEntropyLoss function.
    *   The Adam optimizer is used to update the model's parameters.
    *   Training progress is monitored using TensorBoard.

## Usage Instructions

1.  **Install Dependencies:**
    ```bash
    pip install torch pandas numpy tensorboard
    ```
2.  **Upload Data:** Upload the `trump_full_speech.txt` file when prompted by the script.
3.  **Run the Script:**
    ```bash
    python rnn_textgeneration.py
    ```
4.  **Monitor with TensorBoard:**
    ```bash
    tensorboard --logdir=runs
    ```
    Open the URL provided to view the training progress.

## Key Improvements

*   Addressed errors in loss function and model definition.
*   Corrected text generation function.
*   Clear comments and explanations.
*   Device agnostic - used GPU when available

## Potential Extensions

*   Experiment with different RNN architectures (e.g., LSTM, GRU).
*   Explore different text preprocessing techniques.
*   Implement techniques to improve the quality and coherence of the generated text.
*   Fine-tune the model on a larger dataset of Trump's speeches.
*   Compare the generated text to actual Trump speeches using metrics like perplexity.
