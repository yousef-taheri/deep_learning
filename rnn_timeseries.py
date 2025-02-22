# -*- coding: utf-8 -*-
"""rnn_timeseries.py

This script implements a Recurrent Neural Network (RNN) for time series forecasting.
The goal is to predict future temperatures based on historical temperature data.
"""

from google.colab import files
import torch
from torch import nn
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import io # Import io

# 1. Data Loading and Preprocessing
uploaded = files.upload()

df = pd.read_csv(io.BytesIO(uploaded['tempAMAL_train.csv']))
df = df.drop("datetime", axis=1)  # we dont need this column

df = df.fillna(df.mean(axis=0))  # fill nas with average of each column
df = (df - df.min()) / (df.max() - df.min())  # normalize data


# 2. RNN Model Definition
class RNNforc(torch.nn.Module):
    def __init__(self, latent=15, dim=30):  # Corrected default value dim=1 to dim=30
        super().__init__()
        self.latent = latent
        self.dim = dim  # Input dimension (number of cities)
        self.singleLayer = nn.Linear(latent + self.dim, latent) # Added self to dim
        self.decoder = nn.Linear(latent, 1) # Output is a single temperature value
        self.actF = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size()[1] # Corrected it to x.size()[1]
        lastH = torch.zeros(batch_size, self.latent, dtype=torch.float) # Corrected initialization.
        outsLis = []
        for b in x:  #Iterate through sequence length
            inp = torch.cat((b.unsqueeze(1), lastH), dim=1).float()  #concatenate each city with lastH
            inp = self.singleLayer(inp)
            inp = torch.tanh(inp)
            lastH = inp.float()
            outp = self.decoder(lastH)
            outp = self.actF(outp)
            outsLis.append(outp) #changed from outp.view(-1) to outp
        return torch.stack(outsLis, dim=0).squeeze()

# 3. Data Batching
def batchLoader(length=1000):
    ind = np.random.randint(0, df.shape[0] - length, 1)[0]
    df1 = df.iloc[ind:ind + length, ]
    X = torch.from_numpy(df1.values).float()  #explicitly make it float.
    return X

# 4. Training Setup and Training Loop

writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Added device

# Model 1: 1 time series in dimension R30 (considers correlation among cities)
model1 = RNNforc(17, dim=30).to(device) #Pass the correct dim, and move the model to the device
lr = 0.001 #Increased learning rate
n_epochs = 400
loss_fn = torch.nn.MSELoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
horizen = 1

for t in range(n_epochs):
    X = batchLoader(1000).to(device)  # Move data to device
    n = X.size()[0]
    output1R30 = model1(X[:-horizen].transpose(0, 1)) # forecast
    loss1R30 = loss_fn(X[horizen:, :].mean(dim=1), output1R30)  # compare the forecast with the real data

    loss1R30.backward()
    optimizer1.step()
    optimizer1.zero_grad()
    writer.add_scalar('TP4/LossRNNforc1', loss1R30.item(), t)

# Model 2: 30 time series in dimension 1 (doesn't consider correlation among cities)
model2 = RNNforc(17, dim=1).to(device) #Move the model to the device
lr = 0.001 #Increased learning rate
n_epochs = 400
loss_fn = torch.nn.MSELoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
horizen = 1
for t in range(n_epochs):
    X = batchLoader(1000).to(device) #Move data to device
    n = X.size()[0]
    output30R1 = torch.stack([model2(X[:-horizen,i].view(-1,1).transpose(0,1)) for i in range(X.size()[1])], dim=1)
    loss30R1 = loss_fn(X[horizen:], output30R1[:-horizen,:])

    loss30R1.backward()
    optimizer2.step()
    optimizer2.zero_grad()
    writer.add_scalar('TP4/LossRNNforc2', loss30R1.item(), t)

writer.close()
print("Finished Training, check TensorBoard")
