# deep_learning: A Collection of Deep Learning Implementations in PyTorch

## Overview

This repository contains a collection of Python scripts implementing various deep learning models and techniques using the PyTorch framework. Each script focuses on a specific concept or application.

## Contents

The repository is organized with the following files:

*   Python scripts (`*.py`): Implementations of various deep learning models.
*   README files (`/readme/*.md`): Detailed explanations, instructions, and information for each script.

**Table of Contents:**

| Script Filename       | Description                                                              | README Link                                            |
| --------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------ |
| `seq2seq.py`          | Sequence-to-sequence model for tasks like machine translation.           | [readme/seq2seq.md](readme/seq2seq.md)                |
| `lstm_rnn.py`         | LSTM and RNN models for sequence processing tasks.                       | [readme/lstm_rnn.md](readme/lstm_rnn.md)               |
| `autoencoder.py`      | Autoencoder for dimensionality reduction or anomaly detection.              | [readme/autoencoder.md](readme/autoencoder.md)         |
| `cnn_sentiment.py`    | Convolutional Neural Network (CNN) for sentiment analysis.                 | [readme/cnn_sentiment.md](readme/cnn_sentiment.md)     |
| `rnn_timeseries.py`   | Recurrent Neural Network (RNN) for time series forecasting.               | [readme/rnn_timeseries.md](readme/rnn_timeseries.md)   |
| `self_attention.py`   | Self-attention mechanism for various tasks.                                | [readme/self_attention.md](readme/self_attention.md)   |
| `pytorch_regression.py`| Basic linear regression model using PyTorch.                              | [readme/pytorch_regression.md](readme/pytorch_regression.md) |
| `pytorch_sequential.py`| Simple sequential neural network using PyTorch.                           | [readme/pytorch_sequential.md](readme/pytorch_sequential.md) |
| `rnn_textgeneration.py`| Recurrent Neural Network (RNN) for text generation.                     | [readme/rnn_textgeneration.md](readme/rnn_textgeneration.md) |
| `attention_mechanism.py`| Attention mechanism implementation.   | [readme/attention_mechanism.md](readme/attention_mechanism.md)   |
| `regularization_mnist.py` | Demonstrates regularization techniques on the MNIST dataset.         | [readme/regularization_mnist.md](readme/regularization_mnist.md) |
| `sequence_classification.py`| Model for sequence classification tasks. | [readme/sequence_classification.md](readme/sequence_classification.md) |

## General Instructions

1.  **Prerequisites:**
    *   Python 3.6+
    *   PyTorch (>=1.0)
    *   NumPy
    *   (Other dependencies may be required depending on the specific script; see the corresponding `.md` file in the `/readme` directory.)

2.  **Installation:**
    ```bash
    pip install -r requirments.txt
    # Install any other dependencies as specified in the individual README files.
    ```

3.  **Usage:**
    *   Refer to the individual `.md` files in the `/readme` directory for detailed instructions on how to run each script and interpret the results.
    * The location of datasets should be verified in each program.
    * The programs are ready to run and can be executed independently, if each program can find its dataset.

## Contributing

Contributions to this repository are welcome! Please feel free to submit pull requests with bug fixes, improvements, or new implementations. Please provide a description with your additions.
