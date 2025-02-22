# -*- coding: utf-8 -*-
"""seq2seq: Part-of-Speech Tagging with LSTM

This script implements an LSTM-based Part-of-Speech (POS) tagger using the
Universal Dependencies French GSD dataset. It demonstrates data loading,
vocabulary building, model definition, and training.  The script has been
modified for local execution, removing the need to mount Google Drive and
installing packages directly within the script.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from torch import optim
import nltk
from nltk import tokenize
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import pyconll

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data Loading and Preprocessing
# Define data directory
data_dir = '/data'  # Local data directory
train_file_path = os.path.join(data_dir, "fr_gsd-ud-train.conllu")
dev_file_path = os.path.join(data_dir, "fr_gsd-ud-dev.conllu")
test_file_path = os.path.join(data_dir, "fr_gsd-ud-test.conllu")

# Load data from local .conllu files
train_file = pyconll.load_from_file(train_file_path)
dev_file = pyconll.load_from_file(dev_file_path)
test_file = pyconll.load_from_file(test_file_path)

# 3. Vocabulary Building
class dictbuilder:
    def __init__(self):
        self.word2id = {"__OOV__": 0, "__PAD__": 1}  # Added padding token
        self.id2word = {0: "__OOV__", 1: "__PAD__"} # Added padding token
        self.tag2id = {}
        self.id2tag = {}

    def addtoDict(self, cnlFiles):
        for cnlFile in cnlFiles:
            for s in cnlFile:
                for g in s:
                    if g.form not in self.word2id:
                        self.word2id[g.form] = len(self.word2id)
                        self.id2word[len(self.id2word)] = g.form

                    if g.upos not in self.tag2id:
                        self.tag2id[g.upos] = len(self.tag2id)
                        self.id2tag[len(self.tag2id)] = g.upos

vocab = dictbuilder()
vocab.addtoDict([train_file, test_file])

# 4. Dataset Preparation
Batch = namedtuple("Batch", ["words", "tags"])

class myTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataConllu):
        self.data = dataConllu

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sentence = self.data[index]
        tokens, tags = zip(*[(vocab.word2id.get(tok.form, 0), vocab.tag2id[tok.upos]) for tok in sentence if tok.form and tok.upos]) # fixed problem of tok

        return (tokens, tags)

    @staticmethod
    def collate(batch):
        words = [torch.tensor(item[0],dtype=torch.long) for item in batch] # enforce type long
        tags = [torch.tensor(item[1],dtype=torch.long) for item in batch] # enforce type long
        out1 = rnn_utils.pad_sequence(words, batch_first=True, padding_value=1)  # Pad with padding index
        out2 = rnn_utils.pad_sequence(tags, batch_first=True, padding_value=len(vocab.tag2id))  # Pad with tag padding index
        return Batch(out1, out2)

# 5. Model Definition
class LSTMspeech(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=70): #Use vocab size and tag size
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)  # Added padding_idx
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,num_layers=2,bidirectional=True)  # Use nn.LSTM
        self.linear = nn.Linear(hidden_dim*2, tag_size)
        self.log_softmax = nn.LogSoftmax(dim=2) #added log softmax, and multiplayed hidden layer by 2 because its biderectional

    def forward(self, x):
        embedded = self.embedding(x) #embedd
        output, _ = self.lstm(embedded) # the output of the LSTM and the h_n and c_n
        output = self.linear(output) #The output is put in a linear layer
        output = self.log_softmax(output) # and the softmax function is runned.
        return output

# 6. Training Setup
# Create dataset and dataloader
myDataset = myTextDataset(train_file)
data = DataLoader(myDataset, shuffle=True, batch_size=100, collate_fn=myTextDataset.collate)

#Define the model
vocab_size = len(vocab.word2id)
tag_size = len(vocab.tag2id)
model = LSTMspeech(vocab_size,tag_size).to(device) #Pass the parameters that define the dimensions, and put in the device
print("Model created")

#Parameters
writer = SummaryWriter()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.NLLLoss(ignore_index=len(vocab.tag2id)) # the loss function and its parameters
print("Loss function created")
#Check some parameters
for x, y in data:
    print(x.shape, y.shape)
    break
print("Parameters checked")
# 7. Training Loop
epochs = 30

for epoch in range(epochs):

    losses = []
    for i,(sentences, tags) in enumerate(data): #added enumerate for visualization
        sentences = sentences.to(device)
        tags = tags.to(device)
        model.train()
        optimizer.zero_grad()
        pred = model(sentences)
        pred = pred.permute(0, 2, 1) #to match the loss dimmensions with a permutation
        loss = loss_function(pred,tags)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 10 == 0: # Reduce printing frequency
            print(f"Epoch:{epoch}, batch:{i}, loss:{loss.item():.4f}")
    print("Epoch number",epoch," and loss value= ",np.mean(losses))
