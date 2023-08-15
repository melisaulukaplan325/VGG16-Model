import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import glob
import numpy
import cv2
import image_script
import cv2

#idx_to_class = {i:j for i, j in enumerate(classes)}
#class_to_idx = {value:key for key,value in idx_to_class.items()}

class MNIST_Dataset(Dataset):
    #an init function we use to initlize very attribute we created
    def __init__(self, image_paths, batch_size=1, shuffle = False):
        with open () as f:
            image_paths = f.readline()
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle 

    def __len__(self):#len function for returnning length of image path
        return len(self.image_paths)
    
    def __getitem__(self, idx):#getitem function for calculating every needed attribute
        data_path = self.image_paths[idx*self.batch_size:self.batch_size*idx+self.batch_size]
        print(data_path)
        for(idx, path) in enumerate(data_path):
            image = cv2.imread(data_path)
            try:
                image = image.resize(224,224)
            except:
                print('error at resizing')
            label = path.split('/')[-2]
            #label = class_to_idx[label]

            #permute changes the dimension
            x = torch.tensor(image/255.0, dtype= torch.float32 ).cuda().permute(0,2,0)
            y = torch.tensor(float(label), dtype= torch.float32).cuda()

            if idx == 0:
                #unsqueeze adds dimension to tensor
                XAll = x.clone().unsqueeze(0)
                YAll = y.clone().unsqueeze(0)
            else:
                #torch.cat combines 2 tensor with dimension if dim is 0 it combines column wise if dim is 1 it combines it row wise
                XAll = torch.cat(XAll, x.unsqueeze(0), dim=0)
                YAll = torch.cat(YAll, y.unsqueeze(0), dim=0)
            try:
                return XAll, YAll
            except:
                print("error")

            
        
            

        



        

        

