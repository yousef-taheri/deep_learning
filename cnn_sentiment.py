# -*- coding: utf-8 -*-
"""cnn.py: CNN for Text Classification (Integrated Dataset)

This script implements a Convolutional Neural Network (CNN) for text classification.
The goal is to classify text sequences into different categories.
This version integrates the TextDataset class directly into the script and
adjusts filepaths for local execution.
"""

import torch
import torch.nn as nn
import numpy as np
import gzip
import os
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple  # Import namedtuple
import torch.nn.utils.rnn as rnn_utils #Import rnn_utils

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Data Loading and Preparation

# Define data directory
data_dir = '/data'  # Adjusted data directory for local execution

# Define paths
model_save_name = 'cnn.pt'
savepath = os.path.join(data_dir, model_save_name) # save in local /data folder

# Batch namedtuple
Batch = namedtuple("Batch", ["text", "labels"])

# TextDataset class (integrated from preprocess.py)
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text: torch.LongTensor, sizes: torch.LongTensor, labels: torch.LongTensor):
        self.text = text
        self.sizes = sizes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.text[self.sizes[index]:self.sizes[index + 1]], self.labels[index].item()

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return Batch(rnn_utils.pad_sequence(data, batch_first=True), torch.LongTensor(labels))


# Load datasets
with gzip.open(os.path.join(data_dir, 'train-1000.pth'), 'rb') as f:  # Use local filepaths
    train_ds = torch.load(f)
with gzip.open(os.path.join(data_dir, 'test-1000.pth'), 'rb') as f:  # Use local filepaths
    test_ds = torch.load(f)

# 3. SentencePiece Model
s = spm.SentencePieceProcessor()
s.Load(os.path.join(data_dir, 'wp1000.model'))  # Use local filepath

print(s.encode_as_pieces('This is a test'))
print(s.encode_as_ids('This is a test'))

print(s.decode_pieces([' This', ' is', ' a', ' t', 'est']))
print(s.decode_ids([156, 22, 11, 248, 277]))

# 4. Data Loaders
train_loader = DataLoader(train_ds, collate_fn=TextDataset.collate, shuffle=True, batch_size=500)
test_loader = DataLoader(test_ds, collate_fn=TextDataset.collate, shuffle=False, batch_size=500)  # disable shuffle for testing

# 5. CNN Model Definition
class myCNN(nn.Module):
    def __init__(self, inDim, embedDim):
        super().__init__()
        self.embedding = nn.Embedding(inDim, embedDim)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=embedDim, out_channels=20, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=3, kernel_size=2, stride=1),  # Output = 3 classes
            nn.ReLU())

        self.fc = nn.Linear(3, 3)  # Add fully connected layer - this is the last layer
        self.softmax = nn.LogSoftmax(dim=1)  # apply softmax to get probabilities

    def forward(self, xx):
        # Embedding layer
        x = self.embedding(xx)  # shape = (batch_size, seq_len, embed_dim)

        # Transpose for CNN input
        x = x.transpose(1, 2)  # shape = (batch_size, embed_dim, seq_len)

        # CNN layers
        x = self.cnn(x)  # shape = (batch_size, 3, new_seq_len)

        # Max pooling
        x = torch.max(x, dim=2)[0]  # shape = (batch_size, 3)

        # Fully connected layer
        x = self.fc(x)

        # Log Softmax
        x = self.softmax(x)  # shape = (batch_size, 3)

        return x


class State:
    def __init__(self, model, optim, epoch=0, iteration=0):  # Added epoch/iteration initialization
        self.model = model
        self.optim = optim
        self.epoch = epoch
        self.iteration = iteration


# 6. Training Setup
lossFunc = nn.NLLLoss()  # Negative Log-Likelihood Loss for classification
lr = 1e-3
nTokens = 1000
ITERATIONS = 200  # Reduced iterations for demonstration

# 7. Load/Initialize Model
if os.path.isfile(savepath):
    with open(savepath, "rb") as fp:
        state = torch.load(fp, map_location=device)  # Load to the correct device
    cnnClf = state.model
    optimizer = state.optim
    print("using previous model")
else:
    cnnClf = myCNN(nTokens, 30).to(device)
    optimizer = torch.optim.Adam(params=cnnClf.parameters(), lr=lr, betas=(.9, .999), eps=1e-8)
    state = State(cnnClf, optimizer)  # Create a state
    print("initilising new model")

# 8. Training Loop
writer = SummaryWriter()
cnnClf.to(device)
for epoch in range(state.epoch, ITERATIONS):  # Start from the epoch saved
    Losses = []
    for x, y in train_loader:
        cnnClf.train()  # Set in train mode
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        yhat = cnnClf(x)
        loss = lossFunc(yhat, y)

        loss.backward()
        optimizer.step()
        Losses.append(loss.item())

    avg_loss = np.mean(Losses)
    print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
    writer.add_scalar('Loss/train', avg_loss, epoch)

    # Checkpointing
    state = State(cnnClf, optimizer, epoch + 1, state.iteration)  # create a new state object.
    with open(savepath, "wb") as fp:
        torch.save(state, fp)

writer.close()
print("Finished Training, check TensorBoard")


# 9. Visualization function
def prFunc(yy, yyhat, x, s):
    twt = x.cpu().detach().numpy()
    twt = twt[twt > 0].tolist()
    decoded_text = s.decode_ids(twt)
    if yy == yyhat:
        print("\x1b[32m" + decoded_text + "\x1b[0m")  # Green for correct
    else:
        print("\x1b[31m" + decoded_text + "\x1b[0m")  # Red for incorrect
    print("----------------------------------------------------")


# 10. Evaluation
cnnClf.eval()  # Set in eval mode
with torch.no_grad():  # disable gradients
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        yhat = cnnClf(x)  # Get the prediction

        predictions = torch.argmax(yhat, dim=1)
        for a in range(len(x)):
            prFunc(y[a].item(), predictions[a].item(), x[a], s)
        break
