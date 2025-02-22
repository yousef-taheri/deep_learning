# -*- coding: utf-8 -*-
"""self_attention: Self-Attention Mechanisms for Sentiment Analysis

This script implements a self-attention mechanism for sentiment analysis. It
includes the base model (Exo 0) and positional embeddings (Exo 2). Due to
incomplete details in the provided text, some assumptions were made.
See README.md for details.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import math

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data and Model Parameters (Simplified)
vocab_size = 1000  # Example vocabulary size
embedding_dim = 50
hidden_dim = 100
num_classes = 2  # Positive or negative sentiment
sequence_length = 20  # Example sequence length
batch_size = 32
num_layers = 3  # Number of self-attention layers (L=3 as suggested in the PDF)

# 3. Data Generation (Simplified - Random Data)
def generate_data(num_samples):
    data = torch.randint(0, vocab_size, (num_samples, sequence_length))
    labels = torch.randint(0, num_classes, (num_samples,))
    return data, labels

train_data, train_labels = generate_data(1000)
test_data, test_labels = generate_data(200)

# 4. Positional Encoding (Exo 2)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 5. Self-Attention Model (Exo 0 and Exo 2)
class SelfAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers,dropout_rate=0.1): #Added number of layers as parameter
        super(SelfAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim,dropout = dropout_rate,max_len=sequence_length) #Positional Encoding
        self.attention_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers) #Create a layer list
        ])
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.num_layers = num_layers
        self.g0 = nn.Linear(embedding_dim,embedding_dim) #g0
        self.f0 = nn.Linear(embedding_dim,embedding_dim) #f0
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x) #embedd
        x = self.positional_encoding(x.transpose(0,1)).transpose(0,1) #positional encoding

        # Self-attention layers
        for i in range(self.num_layers):
            #Attention
            attention_weights = torch.softmax(self.attention_layers[i](x), dim=1)
            x = x + self.f0(attention_weights * x) #Residual connection
            x = self.g0(x) #g0
            x = self.relu(x) #activation

        x = torch.mean(x, dim=1)  # Average embeddings (batch_size, embedding_dim)
        x = self.linear(x)  # (batch_size, num_classes)
        x = self.softmax(x) #Softmax to normalize

        return x

# 6. Training Setup
# Instantiate the model
model = SelfAttentionModel(vocab_size, embedding_dim, hidden_dim, num_classes, num_layers).to(device)

# Loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard setup
writer = SummaryWriter()

# 7. Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Get the batch
    data = train_data.to(device)
    labels = train_labels.to(device)

    # Pass it to the model
    outputs = model(data) #Run the model to get the output

    #Calculate the loss
    loss = criterion(outputs, labels) #calculate the loss function

    # Backpropagation
    loss.backward() #perform backpropagation
    optimizer.step() #optimize

    #Print info in steps of 10
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        writer.add_scalar('training loss', loss.item(), epoch)  # Tensorboard visualization

# 8. Evaluation
model.eval() #Set the model to eval mode

with torch.no_grad():
    data = test_data.to(device)
    labels = test_labels.to(device)
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1) #Check which output maximizes the probability

    accuracy = (predicted == labels).sum().item() / len(test_labels) #Calculate amount of correctly classified inputs

    print('Accuracy of the network on the test data: {} %'.format(100 * accuracy)) #Report

writer.close()
