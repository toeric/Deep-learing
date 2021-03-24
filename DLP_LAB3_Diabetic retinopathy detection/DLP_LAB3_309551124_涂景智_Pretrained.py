#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import os
from  dataloader import RetinopathyLoader
import numpy as np
from sklearn.metrics import accuracy_score
import torchvision.models as models
import random


resnet18 = models.resnet18(pretrained=True)
output_dict = {"0":0, "1":0, "2":0, "3":0, "4":0, "none":0}
resnet18.fc = nn.Linear(512, 5)


def test_model(net, is_last):
    net.eval()
    train_predicted = []
    test_predicted = []
    train_labels = []
    test_labels = []
    for i in range(int(all_test_img.__len__()/batch_size)):

        inputs = []
        for j in range(batch_size):
            _input, _label = all_test_img.__getitem__(i*batch_size + j)

            inputs.append(_input)
            test_labels.append(_label)

        inputs = [each.numpy() for each in inputs]
        inputs = np.array(inputs)
        ten_inputs = torch.FloatTensor(inputs)

        if torch.cuda.is_available():
            ten_inputs = ten_inputs.cuda()

        outputs = net(ten_inputs)
        for each in outputs.data.cpu():
            test_predicted.append(np.array(each))
   
        if i%500==0:
            print("Test Calculate accuracy " + str(i) + "th")

        

    for i in range(int(all_train_img.__len__()/batch_size)):
        inputs = []
        for j in range(batch_size):
            _input, _label = all_train_img.__getitem__(i*batch_size + j)
            inputs.append(_input)
            train_labels.append(_label)

        inputs = [each.numpy() for each in inputs]
        inputs = np.array(inputs)
        ten_inputs = torch.FloatTensor(inputs)
 
        if torch.cuda.is_available():
            ten_inputs = ten_inputs.cuda()
 
        outputs = net(ten_inputs)
        for each in outputs.data.cpu():
            train_predicted.append(np.array(each))
 
        if i%500==0:
            print("Train Calculate accuracy " + str(i) + "th")




    train_predicted_arr = []
    test_predicted_arr = []


    for each in test_predicted:
        max_idx = 0
        max_val = each[0]
        
        for i in range(len(each)):
            if each[i] > max_val:
                max_val  = each[i]
                max_idx = i

        test_predicted_arr.append(max_idx)

    for each in train_predicted:
        max_idx = 0
        max_val = each[0]

        for i in range(len(each)):
            if each[i] > max_val:
                max_val  = each[i]
                max_idx = i

        train_predicted_arr.append(max_idx)
            

    if is_last == "last":
        f = open('Resnet18_Pretrained.txt', 'w')
        f.write("[")
        for i in range(len(test_predicted_arr)-1):
            f.write(str(test_predicted_arr[i]))
            f.write(",")
        f.write(str(test_predicted_arr[len(test_predicted_arr)-1]))
        f.write("]")
        f.close()

    return accuracy_score(np.array(train_labels), np.array(train_predicted_arr)), accuracy_score(np.array(test_labels), np.array(test_predicted_arr))



print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(net, data):
    for epoch in range(10):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()

        data.Active_stuff()
       
        for i in range(int(all_train_img.__len__()*1.5/batch_size)):  
            net.train() ##
            inputs = []
            labels= []
            for j in range(batch_size):
                ran_int =  random.randint(0, all_train_img.__len__()*2)
                _input, _label = all_train_img.__getitem__(ran_int)
                output_dict[str(_label)] += 1          
                inputs.append(_input)
                labels.append(_label)

            inputs = [each.numpy() for each in inputs]    
            inputs = np.array(inputs) 
            labels = torch.tensor(labels)
            ten_inputs = torch.FloatTensor(inputs)

            if torch.cuda.is_available():
                ten_inputs, labels = ten_inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

        # forward + backward
            outputs = net(ten_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if i%500==0:
                print(str(i) + "th batch")

        data.Cancel_stuff() 
        if epoch == 9:
            Train_acc, Test_acc = test_model(net, "last")
        else:
            Train_acc, Test_acc = test_model(net, "notlast")
        
        print("Epoch: " +  str(epoch+1) + ", Train Accuracy = " + str(Train_acc) + ", Test Accuracy = " + str(Test_acc))
        print(output_dict)


all_train_img = RetinopathyLoader('data/', 'train')
all_test_img = RetinopathyLoader('data/', 'test')
net = resnet18.to(device)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) 
#optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=5e-4)
batch_size = 4

train_model(net, all_train_img)





