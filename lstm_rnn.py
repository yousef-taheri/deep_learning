# -*- coding: utf-8 -*-
"""lstm_rnn.py: LSTM, GRU, and RNN for Text Generation

This script implements and compares LSTM, GRU, and simple RNN models for text
generation. It uses the "trump_full_speech.txt" dataset and SentencePiece
for tokenization.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from torch import optim
import sentencepiece as spm
import nltk
from nltk import tokenize
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.rnn as rnn_utils

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data Loading and Preprocessing

# Define data directory
data_dir = '/data'

# Load text data
with open(os.path.join(data_dir, 'trump_full_speech.txt'), 'r') as f:
    text = f.read()

# 3. SentencePiece Tokenization
vocab_size = 2000  #Increased vocab size for more accurate text generation
model_prefix = os.path.join(data_dir, "m")

# Train SentencePiece Model
if not os.path.exists(f"{model_prefix}.model"):
    import sentencepiece as spm
    input_file = os.path.join(data_dir, 'trump_full_speech.txt')
    spm.SentencePieceTrainer.train(
        f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3')
    print("SentencePiece model trained.")

# Load SentencePiece Model
sp = spm.SentencePieceProcessor()
sp.load(f'{model_prefix}.model')

# Print SentencePiece Examples
print(sp.encode_as_pieces('This is a test'))
print(sp.encode_as_ids('This is a test'))
print(sp.decode_pieces([' This', ' is', ' a', ' t', 'est']))
print(sp.decode_ids([156, 22, 11, 248, 277]))

# 4. Dataset Preparation
nltk.download('punkt', download_dir = data_dir)  # download punkt tokenizer.

SentenceSplt = tokenize.sent_tokenize(text)
encoded_sentences = [sp.encode_as_ids(x) for x in SentenceSplt] # encode sentences.

# Padding
max_len = max(len(s) for s in encoded_sentences) #Get max size for padding
padded_sentences = [s + [3] * (max_len - len(s)) for s in encoded_sentences] #Pad with pad ID=3

encoded_text_tensor = torch.tensor(padded_sentences, dtype=torch.long)

#Data loader.
train_data = encoded_text_tensor
batch_size = 100

train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size) #No textDataset and collate_fn is needed

# 5. Model Definitions
class LSTMspeech(nn.Module):
    def __init__(self, latent=70, embdDim=100, vocab_size=vocab_size):  # Added missing self parameter and output dimension
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embdDim, padding_idx=3)
        self.lstm = nn.LSTM(embdDim, latent, batch_first=True)  # Use nn.LSTM
        self.linear = nn.Linear(latent, vocab_size)  # Output layer to vocab_size
        self.softmax = nn.LogSoftmax(dim=2) #Added softmax.

    def forward(self, x):
        # Pass the input through the embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # Pass the embedded input to the LSTM
        output, _ = self.lstm(embedded)  # output shape: (batch_size, seq_len, hidden_size)

        # Pass the output to the linear layer
        output = self.linear(output)  # output shape: (batch_size, seq_len, vocab_size)

        # Pass the result to the log softmax
        output = self.softmax(output)  # output shape: (batch_size, seq_len, vocab_size)

        return output

class GRUspeech(nn.Module):
    def __init__(self, latent=70, embdDim=100, vocab_size=vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embdDim, padding_idx=3)
        self.gru = nn.GRU(embdDim, latent, batch_first=True)  # Use nn.GRU
        self.linear = nn.Linear(latent, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        output = self.linear(output)
        output = self.softmax(output)
        return output

class RNNspeech(nn.Module):
    def __init__(self, latent=70, embdDim=100, vocab_size=vocab_size): #Also fixed dimension output
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embdDim, padding_idx=3)
        self.rnn = nn.RNN(embdDim, latent, batch_first=True)  # Use nn.RNN
        self.linear = nn.Linear(latent, vocab_size) #Also fixes dimension output
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.linear(output)
        output = self.softmax(output)
        return output

# 6. Training Setup
writer = SummaryWriter()

# 7. Model Instantiation
model1 = LSTMspeech(15, 100, vocab_size).to(device) # Also pass the correct parameters now
model2 = GRUspeech(15, 100, vocab_size).to(device) # Corrected code
model3 = RNNspeech(15, 100, vocab_size).to(device)  #The same for all the model
print("All models to device")

# Define optimizers
lr = 0.001 #Increased learning rate for faster convergance
loss_fn = nn.NLLLoss(ignore_index = 3) #Added ignore index as 3 is now a padding index

optimizer1 = optim.Adam(model1.parameters(), lr=lr)
optimizer2 = optim.Adam(model2.parameters(), lr=lr)
optimizer3 = optim.Adam(model3.parameters(), lr=lr)

# 8. Training Loop
n_epochs = 20
for epoch in range(n_epochs):
    for X in train_loader:
        X = X.to(device)

        # Forward pass
        output1 = model1(X)
        output2 = model2(X)
        output3 = model3(X)

        # Calculate loss (Shift input by 1)
        loss1 = loss_fn(output1[:, :-1, :].transpose(1, 2), X[:, 1:])
        loss2 = loss_fn(output2[:, :-1, :].transpose(1, 2), X[:, 1:])
        loss3 = loss_fn(output3[:, :-1, :].transpose(1, 2), X[:, 1:])

        # Backward and optimize
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        loss1.backward()
        loss2.backward()
        loss3.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        print(epoch, "lstm: ", loss1.item(), " GRU: ", loss2.item(), " Rnn: ", loss3.item())

        writer.add_scalars('Run2', {
            'LSTM': loss1.item(),
            'GRU': loss2.item(),
            'RNN': loss3.item()
        }, epoch)

print("Finished Training, check TensorBoard")
