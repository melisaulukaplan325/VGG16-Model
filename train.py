import numpy 
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from data_loader import MNIST_Dataset
import glob
import numpy
import cv2
import image_script
import cv2
import torch.nn as tnn

train_loader_path = "C:\\staj\\MNIST_train\\MNIST\\MNIST\\MNIST\\train.txt"
valid_loader_path = "C:\\staj\\MNIST_train\\MNIST\\MNIST\\MNIST\\validation.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(train_loader_path, batch_size=1, shuffle=False)
valid_loader = DataLoader(valid_loader_path, batch_size=1, shuffle=False)

model = torchvision.models.vgg16()

cost = tnn.CrossEntropyLoss()#calculates loss in multiclass and binary classification
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01)#optimizing parameters 


for epoch in range(100):
    avg_loss=0
    count=0
    for(input, label)in enumerate(train_loader):#enumerate adds a iterable counter
        outputs= model(input)
        optimizer.zero_grad()#reset gradient to zero 
        loss = cost(outputs, label)
        loss.backward()#calculates gradient of losses respect to weights etc.
        optimizer.step()#adjust parameters to gradient
       
        avg_loss += loss
        print("[Epoch: %d] loss:%f, avg_loss: %f" % (epoch, loss, avg_loss/len(train_loader)) )

for epoch in range(100):
    total_corrects = 0
    with torch.no_grad():#notify layers for validation mode instead f training
        model.eval()
        for(input, label) in enumerate(valid_loader):
            outputs = model(input)
            prediction = torch.max(outputs, 1)
            total_corrects += (prediction==label).float().item()#if the prediction and label is equal total corrects increases
        

    accuracy = 100* total_corrects/len(valid_loader)
    print("Epoch: %f , accuracy: %f")

        

