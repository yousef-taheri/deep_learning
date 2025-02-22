# -*- coding: utf-8 -*-
"""sequence_classification.ipynb

This script implements a Recurrent Neural Network (RNN) for sequence classification.
The goal is to classify temperature sequences from different cities.
"""

import torch
from torch import nn
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import io
from google.colab import files

# 1. Data Loading and Preprocessing

# Upload the training data
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['tempAMAL_train.csv']))

# Drop the datetime column (not needed for this task)
df = df.drop("datetime", axis=1)

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Fill NaN values with the mean of each column
df = df.fillna(df.mean(axis=0))

# Normalize the data to the range [0, 1]
df = (df - df.min()) / (df.max() - df.min())

# 2. RNN Model Definition

class RNN(torch.nn.Module):
    def __init__(self, latent, num_cities):  # dimout is now num_cities
        super().__init__()
        self.latent = latent
        self.dim = 1  # Input dimension (temperature)
        self.singleLayer = nn.Linear(latent + self.dim, latent)
        self.decoder = nn.Linear(latent, num_cities)  # Output layer: num_cities
        self.actF = nn.Softmax(dim=1)  # Softmax for classification

    def forward(self, x):
        batch_size = x.size()[1]
        lastH = torch.zeros(batch_size, self.latent, dtype=torch.float) # changed from ones to zeros
        for b in x:  # Iterate over the time steps
            #b = b.view(-1, 1).float() # Removed because 'b' is already the correct shape (needed when not using dataloader)
            inp = torch.cat((b.unsqueeze(1), lastH), dim=1).float() #keep the dimensions consistent
            inp = self.singleLayer(inp)
            inp = torch.tanh(inp)
            lastH = inp.float().view(batch_size, self.latent)

        outp = self.decoder(lastH)
        outp = self.actF(outp)
        return outp

# 3. Data Loading and Batching
n, nbCities = df.shape
nbSampleCity = 10

def dataloader(length=30, batch_size=20):
    selectedCities = np.random.choice(nbCities, nbSampleCity, replace=False)
    cities = np.random.choice(selectedCities, batch_size)
    startSeq = np.random.randint(0, n - length, batch_size)
    x = []
    y = []
    for i, elmt in enumerate(startSeq):
        city = cities[i]  # city to sample from
        city = int(city)
        elmt = int(elmt)
        sample_seq = df.iloc[elmt:(elmt + length), city]  # sequence of with length elements starting at index elmt
        x.append(torch.from_numpy(sample_seq.values).view(1, -1).float()) # Added float() and explicit view
        y.append(city) #keep the city id as is

    x_batch = torch.cat(x, dim=0).transpose(0, 1) #Corrected: concatenation and transpose
    y_batch = torch.tensor(y, dtype=torch.long)  # City labels (integers)
    return x_batch, y_batch

# 4. Training Setup
writer = SummaryWriter()
model = RNN(17, nbSampleCity)  # num_cities should be nbSampleCity

lr = 0.001 #Increased learning rate
n_epochs = 700
loss_fn = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 5. Training Loop
for t in range(n_epochs):
    x_batch, y_batch = dataloader(length=400, batch_size=100)
    output = model(x_batch)
    loss = loss_fn(output, y_batch)  # Compare output with city labels (integers)
    if t % 13 == 0:
        print(t, loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    writer.add_scalar('TP4/LossRNNclf', loss.item(), t)

writer.close()
print("Finished Training, check TensorBoard")
