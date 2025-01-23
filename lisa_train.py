import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision import models
from torchmetrics import Accuracy

import quantus
import captum
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import copy
import gc
import math

import warnings
warnings.filterwarnings('ignore')

from lisa import LISA
from itertools import chain
from model import *
from pathlib import Path

device = torch.device("cpu" if torch.cuda.is_available() else "cuda")
epochs = 100
batch_size = 128

normalize = transforms.Normalize(mean=[0.4563, 0.4076, 0.3895], std=[0.2298, 0.2144, 0.2259])

lisa_transforms = transforms.Compose([ transforms.ToPILImage(),transforms.ToTensor(),normalize])

train_dataset = LISA(root='./datasets', download=True, train=True, transform = lisa_transforms)
test_dataset = LISA(root='./datasets', download=True, train=False, transform = lisa_transforms)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,) # num_workers=4,
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = vgg16()
learning_rate = 0.01
criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)

def evaluate_model(model, dataloader, device):
    """
    This function evaluates the model using test dataset
    """
    model.eval()
    prediction = torch.Tensor().to(device)
    labels = torch.LongTensor().to(device)

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = torch.cat([prediction, model(x_batch)])
            labels = torch.cat([labels, y_batch])
            
    # passing the logits through Softmax layer to get predicted class
    prediction = torch.nn.functional.softmax(prediction, dim=1)
    
    return prediction, labels

def train_model(model, epochs):
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate model!
        if epochs%10==0:
            predictions, labels = evaluate_model(model, test_dataloader, device)
            test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())
            print(f"Epoch {epoch+1}/{epochs} - test accuracy: {(100 * test_acc):.2f}% and CE loss {loss.item():.2f}")
    return model

model_normal = train_model(model = model.to(device), epochs = epochs)

model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

model_name = "lisa_normal.pth"
model_save_path = model_path / model_name

print(f"Saving the model: {model_save_path}")
torch.save(obj=model_normal.state_dict(), f=model_save_path)

model = vgg16().to(device)
model.load_state_dict(torch.load(model_save_path))

model.to(device)
model.eval()

# Check test set performance.
predictions, labels = evaluate_model(model, test_dataloader, device)
test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())        
print(f"Test accuracy for VGG LISA Normal is: {(100 * test_acc):.2f}%")