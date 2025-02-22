# -*- coding: utf-8 -*-
"""rnn_textgeneration.py

This script implements a Recurrent Neural Network (RNN) for text generation.
The goal is to train the RNN to generate text similar to Trump's speeches.
"""

import string
import unicodedata
from google.colab import files
import torch
from torch import nn
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import random

# 1. Text Preprocessing
LETTRES = string.ascii_letters + string.punctuation + string.digits + " "
id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
id2lettre[0] = ""  ## NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


def normalize(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if c in LETTRES)


def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    if type(t) != list:
        t = t.tolist()
    return "".join(id2lettre[i] for i in t)


# 2. RNN Model Definition
class RNNspeech(torch.nn.Module):
    def __init__(self, latent=15, num_chars=len(lettre2id)): # Added num_chars
        super().__init__()
        self.embedding = nn.Embedding(num_chars, 10) # Use num_chars
        self.latent = latent
        self.singleLayer = nn.Linear(latent + 10, latent)
        self.decoder = nn.Linear(latent, num_chars) # Output layer: num_chars
        self.actF = nn.LogSoftmax(dim=1) # Use LogSoftmax

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        lastH = torch.zeros(batch_size, self.latent, dtype=torch.float)

        for i in range(seq_len):
            inp = x[:, i]  # Get the character index for this time step
            embed = self.embedding(inp)  # Embedding layer
            inp = torch.cat((embed, lastH), dim=1).float()  # Concatenate embedding and hidden state
            inp = self.singleLayer(inp)  # Linear Layer
            inp = torch.tanh(inp)  # tanh activation
            lastH = inp.float()  # Update lastH

        outp = self.decoder(lastH)  # Decoding layer
        outp = self.actF(outp)  # LogSoftmax

        return outp

# 3. Data Loading
uploaded = files.upload()
with open('trump_full_speech.txt', 'r') as f:
    text = f.read()

encodedSpech = string2code(text) # All the data is encoded


# 4. Data Batching
def dataLoader(x, batch_size, seq_len):
    n = x.size(0)
    xs = []
    ys = []
    for i in range(batch_size):
        start = np.random.randint(0, n - seq_len, 1)[0]
        X1 = x[start:start + seq_len]
        Y1 = x[start + seq_len]  # Get the next character
        xs.append(X1)
        ys.append(Y1)
    x_batch = torch.stack(xs, dim=0)
    y_batch = torch.stack(ys, dim=0)
    return x_batch, y_batch


# 5. Training Setup
writer = SummaryWriter()

# 6. Model instantiation
model = RNNspeech(17, len(lettre2id)) # Instantiate the model with the correct num_chars

lr = 0.001 #Increase learning rate
n_epochs = 1000
loss_fn = nn.NLLLoss() # Use NLLLoss
optimizer = optim.Adam(model.parameters(), lr=lr)

# 7. Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Added cuda device
model.to(device) #Added cuda device
encodedSpech=encodedSpech.to(device) #Added cuda device

for t in range(n_epochs):
    X, Y = dataLoader(encodedSpech, batch_size=128, seq_len=200) #Increased the batch size, decreased the seq_len
    X=X.to(device) #Added cuda device
    Y=Y.to(device) #Added cuda device

    output = model(X)
    loss = loss_fn(output, Y)

    if t % 100 == 0:
        print(t, loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    writer.add_scalar('TP4/LossSpeech', loss.item(), t)

writer.close()
print("Finished Training, check TensorBoard")

# 8. Text Generation
def gen(model, init, Nseq, device, num_chars): #Added num_chars,model, device to args
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
      input_seq = string2code(init).to(device)
      generated_text = init
      for _ in range(Nseq):
        output = model(input_seq.unsqueeze(0)) # pass input and put in batch
        probabilities = torch.exp(output[:, -1])  # convert log-probs to probabilities
        predicted_id = torch.multinomial(probabilities, 1).item() # sample from distribution
        generated_text += id2lettre[predicted_id]  # append character

        input_seq = torch.cat((input_seq[1:], torch.tensor([predicted_id]).to(device))), # shift input and append prediction
    return generated_text

print(gen(model, "I love Iran! ", 100, device, len(lettre2id)))
