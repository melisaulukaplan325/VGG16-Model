import numpy as np
import torch
import torch.nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from data_loader import MNIST_Dataset
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



train_loader_path = "C:\\Users\\visea\\Desktop\\melisa_staj\\MNIST\\MNIST_Dataset\\train.txt"
valid_loader_path = "C:\\Users\\visea\\Desktop\\melisa_staj\\MNIST\\MNIST_Dataset\\validation.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = MNIST_Dataset(image_paths= train_loader_path, batch_size=1, shuffle=True)
valid_loader = MNIST_Dataset(image_paths= valid_loader_path, batch_size=1, shuffle=False)


model = models.vgg16(pretrained=True)

inlayer = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
outlayer = nn.Sequential(nn.Linear(4096, 10, bias = True), nn.Softmax(dim=1))

model.features[0] = inlayer
model.classifier[-1] = outlayer
model.cuda()


cost = torch.nn.CrossEntropyLoss().cuda()#calculates loss in multiclass and binary classification
optimizer = torch.optim.SGD(model.parameters(), lr= 0.00001)#optimizing parameters 

accuracy_metrics =Accuracy(task = 'multiclass', num_classes = 10).to(device)
f1score_metrics = F1Score(task = 'multiclass', num_classes = 10).to(device)
precision_metrics = Precision(task = 'multiclass', num_classes = 10).to(device)
recall_metrics = Recall(task = 'multiclass', num_classes = 10).to(device)


for epoch in range(100):
    avg_loss=1500
    loss=0 
    
    model.train()
    #for i , (input, label)in enumerate(train_loader):#enumerate adds a iterable counter
    for i in range(len(train_loader)):
        input,label = train_loader[i]
        optimizer.zero_grad()#reset gradient to zero 
        outputs= F.softmax(model(input), dim =1)
        #outputs = model(input)
    
       
        loss= cost(outputs, label.long())
        
        avg_loss += loss
        loss.backward()#calculates gradient of losses respect to weights etc.
        optimizer.step()#adjust parameters to gradient
        
        
        
        #TORCH METRICS ACCURACY CALCULATION
        accuracy_metrics(outputs, label)
        precision_metrics(outputs, label)
        recall_metrics(outputs, label)
        f1score_metrics(outputs, label)
        
        accuracy = accuracy_metrics.compute()
        precision = precision_metrics.compute()
        recall = recall_metrics.compute()
        f1_score = f1score_metrics.compute()
        
        
        print("Epoch: %d loss: %.2f avg_loss: %.2f Train Accuracy: %.2f Train f1 score: %.2f  Train precision:%.2f  Train recall: %.2f " %(epoch, loss, avg_loss/len(train_loader), accuracy,f1_score, precision, recall), end="\r")
       
        
    accuracy_metrics.reset()
    precision_metrics.reset()
    recall_metrics.reset()
    f1score_metrics.reset()  
        
        
        
        
        

# for epoch in range(30):
    #with torch.no_grad():#notify layers for validation mode instead f training
    model.eval()
    #for i, (input, label) in enumerate(valid_loader):
    
    for i in range(len(valid_loader)):
        input,label = valid_loader[i]
        outputs = F.softmax(model(input), dim=1) 
        #outputs = torch.argmax(outputs)
      
  

        
        #TORCH METRICS ACCURACY CALCULATION
        accuracy_metrics(outputs, label)
        precision_metrics(outputs, label)
        recall_metrics(outputs, label)
        f1score_metrics(outputs, label)
        
        accuracy = accuracy_metrics.compute()
        precision = precision_metrics.compute()
        recall = recall_metrics.compute()
        f1_score = f1score_metrics.compute()
        
        print("Validation Accuracy: %.2f Validation f1 score: %.2f Validation  precision:%.2f Validation recall: %.2f                  " %(accuracy, f1_score, precision, recall), end="\r")    

        torch.save(model,"output\\VGG_train.pt" )
       

    
   
       

    accuracy_metrics.reset()
    precision_metrics.reset()
    recall_metrics.reset()
    f1score_metrics.reset()
        
        
    
    
    
    
    #SKLEARN METRICS CALCULATION
    # accuracy = sklearn.metrics.accuracy_score(label, outputs)
    # f1_score = sklearn.metrics.f1_score(label, outputs, pos_label='PAIDOFF')
    # precision = sklearn.metrics.precision_score(label, outputs)
    # recall = sklearn.metrics.recall_score(label, outputs)        
        

