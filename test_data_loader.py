import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as random
import cv2
import random

class Test_Dataset(Dataset):
    def __init__(self, image_path, batch_size =1, shuffle = True):#initilazing step
        with open(image_path, 'r') as f:
            image_path = f.read().split('\n')
        self.image_path = image_path
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return(len(self.image_path))
    
    def __getitem__(self, index):#computing the image path 
        data_path = self.image_paths[index * self.batch_size : self.batch_size*index + self.batch_size]
        
        for i, path in enumerate(data_path):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, -1)
            
            label = int(path.split('/')[-2])
            
            x = torch.tensor(image/255.0, dtype= torch.float32 ).cuda().permute(2,0,1)
            y = torch.tensor(int(label), dtype= torch.float32).cuda()
            
            if i == 0:
                #global XAll, YAll
                #unsqueeze adds dimension to tensor
                XAll = x.clone().unsqueeze(0)
                YAll = y.clone().unsqueeze(0)
            else:
                #torch.cat combines 2 tensor with dimension if dim is 0 it combines column wise if dim is 1 it combines it row wise
                XAll = torch.cat((XAll, x.clone().unsqueeze(0)), dim=0)
                YAll = torch.cat((YAll, y.clone().unsqueeze(1)), dim=0)
        
            try:
                return XAll, YAll
            except:
                print("error")
