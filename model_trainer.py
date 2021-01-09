
from os import path
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


import dataset
import train
import model


CUDA_VISIBLE_DEVICES=1,2,3
batch = 1

data_path = path.join(os.getcwd(), "data")
ds = dataset.create_dataset(data_path)

padSequence = dataset.PadSequence()

train_loader = DataLoader(
    ds.train, batch_size=batch, shuffle=True) #, collate_fn=padSequence)
dev_loader = DataLoader(
    ds.dev, batch_size=batch, shuffle=False) #, collate_fn=padSequence)

learning_rate = 0.001

model = model.Basic()
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
print (device)
model.to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)
trainer = train.Trainer(model=model,
              device=device,
              optimizer=optimizer,
              train_loader=train_loader,
              dev_loader=dev_loader)

trainer.train_model()
print ("END")