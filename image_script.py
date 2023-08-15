from matplotlib.cbook import flatten
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import glob
import cv2
from sklearn.model_selection import train_test_split


#datapath for finding our validation and train values
train_data_path = "C:\\staj\\MNIST_train\\MNIST\\MNIST\\MNIST\\MNIST_training"
test_data_path = "C:\\staj\\MNIST_train\\MNIST\\MNIST\\MNIST\\MNIST_testing"
#empty arrays for classes and images
train_image_paths = []
classes = []

#train data path içindeki her value için class ve image path
for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/*')[-1])
    train_image_paths.append(glob.glob(data_path + '/*'))  
           
#düz birer array haline getirme
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

test_image_paths= []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))
test_image_paths = list(flatten(test_image_paths))


#print('train image path example', train_image_paths)
#print('classes', classes[0])

#0,8 train /0,2 validation 
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):]


#sorting the paths
train_image_paths= sorted(train_image_paths)
test_image_paths= sorted(test_image_paths)
valid_image_paths = sorted(valid_image_paths)

#writing paths inside of the txt files
with open('C:\\staj\\MNIST_train\\MNIST\\MNIST\\MNIST\\train.txt', 'w') as f:
    for i in range(len(train_image_paths)):
        f.write(str(train_image_paths[i]))
        f.write('\n')

with open('C:\\staj\\MNIST_train\\MNIST\\MNIST\\MNIST\\test.txt', 'w') as f:
    for i in range(len(test_image_paths)):
        f.write(str(test_image_paths[i]))
        f.write('\n')

with open('C:\\staj\\MNIST_train\\MNIST\\MNIST\\MNIST\\validation.txt', 'w') as f:
    for i in range(len(valid_image_paths)):
        f.write(str(valid_image_paths[i]))
        f.write('\n')
   

#print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_path)))






