# -*- coding: utf-8 -*-
"""attention_mechanism: Attention Mechanisms for Sentiment Analysis

This script implements a basic sentiment analysis model with and without a
simple attention mechanism. It uses randomly initialized embeddings for
demonstration purposes. The focus is on implementing the base model (Exo 0)
and the simple attention mechanism (Exo 1) as described in the document.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data and Model Parameters (Simplified)
vocab_size = 1000  # Example vocabulary size
embedding_dim = 50
hidden_dim = 100
num_classes = 2  # Positive or negative sentiment
sequence_length = 20 # Example sequence length
batch_size = 32

# 3. Data Generation (Simplified - Random Data)
def generate_data(num_samples):
    data = torch.randint(0, vocab_size, (num_samples, sequence_length))
    labels = torch.randint(0, num_classes, (num_samples,))
    return data, labels

train_data, train_labels = generate_data(1000)
test_data, test_labels = generate_data(200)

# 4. Base Model (Exo 0)
class BaseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BaseModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes) # Only one layer
        self.softmax = nn.LogSoftmax(dim=1) #add LogSoftmax

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = torch.mean(x, dim=1)  # Average embeddings (batch_size, embedding_dim)
        x = self.linear(x)  # (batch_size, num_classes)
        x = self.softmax(x)
        return x

# 5. Simple Attention Model (Exo 1)
class SimpleAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SimpleAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention_linear = nn.Linear(embedding_dim, 1)  # Attention weights
        self.linear = nn.Linear(embedding_dim, num_classes) # Only one layer
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        attention_weights = torch.softmax(self.attention_linear(x).squeeze(2), dim=1)  # (batch_size, seq_len)
        weighted_embeddings = x * attention_weights.unsqueeze(2)  # (batch_size, seq_len, embedding_dim)
        weighted_representation = torch.sum(weighted_embeddings, dim=1)  # (batch_size, embedding_dim)
        output = self.linear(weighted_representation)  # (batch_size, num_classes)
        output = self.softmax(output) #add LogSoftmax
        return output

# 6. Training Setup
# Instantiate the model
model = SimpleAttentionModel(vocab_size, embedding_dim, hidden_dim, num_classes).to(device) #Change this to the basemodel to test that

# Loss function and optimizer
criterion = nn.NLLLoss() #Loss for multi-clasification is NLLLoss
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

    #Pass to the model
    outputs = model(data)
    loss = criterion(outputs, labels)

    #Backpropagation
    loss.backward()
    optimizer.step()

    #Print info in steps of 10
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        writer.add_scalar('training loss',loss.item(),epoch) #Tensorboard visualization

# 8. Evaluation
model.eval() #Set the model to eval mode

with torch.no_grad():
    data = test_data.to(device)
    labels = test_labels.to(device)
    outputs = model(data) #Pass it to the model, and store the outputs
    _, predicted = torch.max(outputs.data, 1) #Get the max value of the output, and its index (the prediction)

    accuracy = (predicted == labels).sum().item() / len(test_labels) #calculate the total of correctly predicted labels

    print('Accuracy of the network on the test data: {} %'.format(100 * accuracy))
    writer.close()
