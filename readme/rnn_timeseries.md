# rnn_timeseries.py: RNN for Time Series Forecasting

## Overview

This script implements a Recurrent Neural Network (RNN) to forecast temperature values for a set of cities.  It demonstrates two approaches: treating all cities as a single multivariate time series and treating each city as an independent univariate time series.

## Problem Solved

The goal is to predict future temperature values given a history of past temperature data.  The script compares two methods for achieving this:

1.  **Multivariate Forecasting:**  Treats the temperature data for all cities as a single, correlated time series, aiming to capture dependencies between cities.
2.  **Univariate Forecasting:**  Treats the temperature data for each city independently, ignoring potential correlations between cities.

## Dataset

The script utilizes a dataset of temperature readings for multiple cities. Each row represents a timestamp, and each column represents the temperature reading for a specific city at that timestamp.

## Implementation Details

*   **Data Preprocessing:**
    *   Missing values are filled with the mean temperature for each city.
    *   Temperature data is normalized to the range \[0, 1].
*   **RNN Model:**
    *   An RNN model is implemented using PyTorch's `nn.Module`.
    *   The model consists of a linear layer, an RNN layer, and a linear output layer.
*   **Training:**
    *   The model is trained using the Mean Squared Error (MSE) loss function.
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
    python rnn_timeseries.py
    ```
4.  **Monitor with TensorBoard:**
    ```bash
    tensorboard --logdir=runs
    ```
    Open the URL provided to view the training progress.

## Key Improvements

*   Addressed shape mismatches in loss calculations.
*   Corrected tensor alignments
*   Explicitly stated the data types
*   Device agnostic - used GPU when available
*   Clear comments and explanations.

## Potential Extensions

*   Compare the performance of different RNN architectures (e.g., LSTM, GRU).
*   Experiment with different loss functions.
*   Investigate different methods for handling correlations between cities.
*   Evaluate the models on a separate test dataset.
*   Implement techniques for multi-step forecasting.
