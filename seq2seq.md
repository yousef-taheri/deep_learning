# seq2seq.py: LSTM-Based Part-of-Speech Tagging

## Overview

This script implements a Part-of-Speech (POS) tagger using a Long Short-Term Memory (LSTM) network. It utilizes the Universal Dependencies French GSD dataset for training and evaluation. The script demonstrates:

*   Data loading and preprocessing from CoNLL-U format.
*   Vocabulary building for words and POS tags.
*   LSTM model definition with embedding and softmax layers.
*   Training loop with optimization and loss calculation.

The implementation adheres to the principles outlined in the provided PDF, focusing on sequence labeling with RNNs.

## Problem Solved

The script addresses the task of POS tagging, which involves assigning a grammatical category (e.g., noun, verb, adjective) to each word in a sentence. This is a fundamental task in natural language processing (NLP) and serves as a crucial step in many downstream applications, such as machine translation, information extraction, and question answering.

## Dataset

The script uses the Universal Dependencies French GSD dataset, which is a collection of sentences annotated with POS tags in the CoNLL-U format.  The data is expected to be located in the `/data` directory. You can find the dataset here: [https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131)

## Implementation Details

*   **Data Loading:**
    *   The script loads the training, development, and test datasets from local `.conllu` files using the `pyconll` library.
*   **Vocabulary Building:**
    *   The `dictbuilder` class creates vocabularies for words and POS tags.
    *   It assigns unique IDs to each word and tag encountered in the training data.
    *   An out-of-vocabulary (OOV) token is used to represent words not seen during training.
*   **Dataset Preparation:**
    *   The `myTextDataset` class prepares the data for training.
    *   It maps each word and tag in a sentence to its corresponding ID.
    *   It converts the sequences of word and tag IDs to PyTorch tensors.
*   **Model Architecture:**
    *   The `LSTMspeech` class defines the LSTM-based POS tagger:
        *   **Embedding Layer:** Maps word IDs to dense vector representations.
        *   **LSTM Layer:** Processes the embedded word sequences to capture sequential dependencies.
        *   **Linear Layer:** Maps the LSTM output to a vector of tag scores.
        *   **LogSoftmax Layer:** Converts the tag scores to probabilities.
*   **Training:**
    *   The model is trained using the Negative Log-Likelihood Loss (NLLLoss) function.
    *   The Adam optimizer is used to update the model's parameters.
    *   Training progress is monitored by printing the loss every 10 batches.

## Instructions

1.  **Prerequisites:**
    *   Python 3.6+
    *   PyTorch (>=1.0)
    *   pyconll
    *   Numpy
    *   Tensorboard
2.  **Install Dependencies:**
    ```bash
    pip install torch pyconll numpy tensorboard nltk
    ```
    Also, make sure to download nltk with:
    ```python
    import nltk
    nltk.download('punkt')
    ```
3.  **Data Preparation:**
    *   Download the French GSD dataset in CoNLL-U format and place the `.conllu` files (e.g., `fr_gsd-ud-train.conllu`, `fr_gsd-ud-dev.conllu`, `fr_gsd-ud-test.conllu`) into the `/data` directory.
4.  **Run the Script:**
    ```bash
    python seq2seq.py
    ```

## Potential Extensions

*   Implement techniques for handling OOV words more effectively, such as using subword embeddings.
*   Experiment with different LSTM architectures, such as stacked LSTMs or bidirectional LSTMs.
*   Add a Conditional Random Field (CRF) layer on top of the LSTM to improve tagging accuracy.
*   Evaluate the model on a separate test dataset.
*   Implement techniques for handling variable-length sequences more efficiently, such as using packed sequences.
*   Use TensorBoard to monitor the training process and visualize model performance.
