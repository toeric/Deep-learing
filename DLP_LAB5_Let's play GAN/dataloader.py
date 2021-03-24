#!/usr/bin/env python
# coding: utf-8

# In[65]:


import json
import pandas as pd
from torch.utils import data
import numpy as np
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image 
import random
from tqdm import tqdm 


# In[72]:


input_file = open('lab5_dataset/objects.json')
object_dict = json.load(input_file)


# In[84]:


def one_hot(label):
    output_label = []
    for _ in range(24):
        output_label.append(0)
    
    for each in label:
        output_label[object_dict[each]] = 1
    
    return torch.FloatTensor(output_label)
    


# In[85]:


class Data_Loader(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.image = []
        self.label = [] 
        
        input_file = open('lab5_dataset/train.json')
        train_dict = json.load(input_file)
        for i in train_dict:
            path = self.root + "/" + str(i)
            pre_img = Image.open(path)             
            pre_img = pre_img.resize((64,64), Image.ANTIALIAS)
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.5, 0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5, 0.5))
            ])
            img = data_transform(pre_img)
            img = img[:3]
            
            self.image.append(img)
            self.label.append(train_dict[i])
    
    def __getitem__(self, index):
        return self.image[index], one_hot(self.label[index])
            
    def __len__(self):
        return len(self.label)

        

