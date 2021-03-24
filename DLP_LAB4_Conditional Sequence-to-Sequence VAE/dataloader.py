import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import random

def Char_to_Num(each_string):
    return_num = []
    for each in each_string:
        return_num.append(ord(each) - 95) 
    _max = len(each_string)
    return return_num


                
class Dataloader(data.Dataset):
    def __init__(self):
        self.total_word = []
        self.total_num = []
        f = open("train.txt")
        line = f.readline()
        while line:
            self.total_word.append(line.rstrip().split(" "))
            line = f.readline()   
            
        for each_line in self.total_word:
            num_each_line = []
            
            for each_word in each_line:
                none_shuffle = Char_to_Num(each_word)            
                num_each_line.append(none_shuffle)
                
            self.total_num.append(num_each_line)        
                
    def __len__(self):
        return len(self.total_num)
    
    def return_training_pair(self):
        _return = []
        
        x = random.randint(0, self.__len__()-1)
        for i in range(2):
            y = random.randint(0,3)
            
            _return.append(self.total_num[x][y])
            _return.append(y)

        return _return