# lstm_rnn.py: LSTM, GRU, and RNN for Text Generation

## Overview

This script implements and compares three different types of recurrent neural networks (RNNs) – Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and a simple RNN – for the task of text generation. The models are trained on a corpus of text (specifically, Trump's speeches) and then used to generate new text sequences in a similar style.  The implementation utilizes PyTorch for model definition, training, and evaluation. It is designed for local use, without the need for Google Drive integration.

## Problem Solved

The script tackles the problem of generating text that mimics the style of a given author or corpus. This involves training a model that can:

1.  **Understand the structure and patterns of the input text:** Capture the dependencies between words, the typical sentence structure, and the overall style of the text.
2.  **Generate new text sequences:** Produce new sequences of words that adhere to the learned patterns and style, creating coherent and plausible text.

The script compares the performance of three different RNN architectures (LSTM, GRU, and simple RNN) to determine which one is best suited for this task.

## Dataset

The script uses a text file containing the speeches of Donald Trump (`trump_full_speech.txt`). The script can work with any `.txt` file with few changes to the code.

## Implementation Details

*   **Text Preprocessing:**
    *   The raw text is tokenized into sentences using `nltk.tokenize.sent_tokenize`.
    *   Each sentence is further tokenized into subword units using SentencePiece.
    *   The numericalized dataset is then padded so it has a fixed dimension size.
*   **Model Architectures:**
    *   **LSTM Model:** Implements an LSTM network using PyTorch's `nn.LSTM` module. LSTMs are designed to handle long-range dependencies in sequential data, making them well-suited for text generation.
    *   **GRU Model:** Implements a GRU network using PyTorch's `nn.GRU` module. GRUs are a simplified version of LSTMs that can often achieve similar performance with fewer parameters.
    *   **Simple RNN Model:** Implements a basic RNN network using PyTorch's `nn.RNN` module. This model serves as a baseline for comparison.
*   **Training:**
    *   The models are trained using the Negative Log-Likelihood Loss (NLLLoss) function.
    *   The Adam optimizer is used to update the model's parameters.
    *   Training progress is monitored using TensorBoard.
*   **SentencePiece Tokenization:** Before passing strings to the models, the strings are passed to a SentencePiece model. The SentencePiece model has a vocab size of 2000, which is the output dimension of the models.

## Instructions

1.  **Prerequisites:**
    *   Python 3.6+
    *   PyTorch (>=1.0)
    *   SentencePiece
    *   NLTK
    *   TensorBoard
    *   NumPy
2.  **Install Dependencies:**
    ```bash
    pip install torch sentencepiece nltk numpy tensorboard
    ```
    Also, make sure to download nltk with:
    ```python
    import nltk
    nltk.download('punkt')
    ```
3.  **Data Preparation:**
    *   Ensure that the `trump_full_speech.txt` file is located in the `/data` directory. Also ensure that the folders have the correct permissions.
4.  **Run the Script:**
    ```bash
    python lstm_rnn.py
    ```
5.  **Monitor with TensorBoard:**
    ```bash
    tensorboard --logdir=runs
    ```
    Open the URL provided to view the training progress.

## Key Improvements

*   Fixed dimensions of the CNN and tensor shapes.
*   Added Softmax for stable results.
*   Added an ignore index to avoid calculating the loss of pad tokens.
*   Removed custom data processing, and streamlined the data loading with an internal data loader
*   Included a way to generate the SentencePiece Model if the model isn't found.

## Potential Extensions

*   Experiment with different model architectures, including deeper LSTMs and GRUs.
*   Explore techniques for improving the quality and coherence of the generated text, such as beam search.
*   Fine-tune the models on a larger dataset of text.
*   Implement techniques for generating text with specific attributes, such as sentiment or topic.
*   Use pre-trained word embeddings.
*   Add different ways to sample the output strings.
