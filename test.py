import numpy as np
import torch
import torch.nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from test_data_loader import Test_Dataset
import glob
import numpy
import cv2
import torch.nn as tnn
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch.nn as nn
import torch.nn.functional as F
import time
import sklearn.metrics
from pandas.core.common import flatten

test_path_loader = "MNIST/MNIST_Dataset/testnew.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_loader = Test_Dataset(image_path= test_path_loader, batch_size=1, shuffle=True)

#model = models.vgg16(pretrained=True)

# inlayer = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# outlayer = nn.Sequential(nn.Linear(4096, 10, bias = True), nn.Softmax(dim=1))

# model.features[0] = inlayer
# model.classifier[-1] = outlayer
# model.cuda()

model = torch.load("model path")
model.eval()

cost = torch.nn.CrossEntropyLoss().cuda()#calculates loss in multiclass and binary classification
optimizer = torch.optim.SGD(model.parameters(), lr= 0.00001)#optimizing parameters 

accuracy_metrics =Accuracy(task = 'multiclass', num_classes = 10).to(device)
f1score_metrics = F1Score(task = 'multiclass', num_classes = 10).to(device)
precision_metrics = Precision(task = 'multiclass', num_classes = 10).to(device)
recall_metrics = Recall(task = 'multiclass', num_classes = 10).to(device)


for epoch in range(100):
    avgloss= 1500
    loss = 0
    
    
    for i in range(len(test_loader)):
        if i >= len(test_loader):
            break
        input,label = test_loader[i]
        optimizer.zero_grad()#reset gradient to zero 
        outputs = F.softmax(model(input), dim=1) 
        #outputs = torch.argmax(outputs)
        loss= cost(outputs, label.long())
        
        avgloss += loss
        loss.backward()
  

        
        #TORCH METRICS ACCURACY CALCULATION
        accuracy_metrics(outputs, label)
        precision_metrics(outputs, label)
        recall_metrics(outputs, label)
        f1score_metrics(outputs, label)
        
        accuracy = accuracy_metrics.compute()
        precision = precision_metrics.compute()
        recall = recall_metrics.compute()
        f1_score = f1score_metrics.compute()
        
        print("Epoch: %d  Loss: %.2f  Test Accuracy: %.2f  Test f1 score: %.2f  Test precision:%.2f  Test recall: %.2f" %(epoch, loss, accuracy, f1_score, precision, recall), end="\r")    



    
   
       

    accuracy_metrics.reset()
    precision_metrics.reset()
    recall_metrics.reset()
    f1score_metrics.reset()
    
