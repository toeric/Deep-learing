#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from dataloader import read_bci_data
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[3]:


train_data, train_label, test_data, test_label = read_bci_data()


# In[4]:


_active = "ELU"
def decide_act():
    if _active == "ELU":
        return nn.ELU()
    elif _active == "ReLU":
        return nn.ReLU()
    else:
        return nn.LeakyReLU()


# In[5]:


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            decide_act(),
            nn.AvgPool2d((1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            decide_act(),
            nn.AvgPool2d((1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        conv1_out = self.firstconv(x)
        conv2_out = self.depthwiseConv(conv1_out)
        conv3_out = self.separableConv(conv2_out)
        conv3_out = conv3_out.view(-1, 736)
        out  = self.classify(conv3_out)
        return out


# In[6]:


class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        
        self.TotalConv = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), stride=(1, 1), bias=True),
            nn.Conv2d(25, 25, (2, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(25, affine=True, track_running_stats=True),
            decide_act(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(25, 50, (1, 5), stride=(1, 1), bias=True),
            nn.BatchNorm2d(50, affine=True, track_running_stats=True),
            decide_act(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(50, 100, (1, 5), stride=(1, 1), bias=True),
            nn.BatchNorm2d(100, affine=True, track_running_stats=True),
            decide_act(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(100, 200, (1, 5), stride=(1, 1), bias=True),
            nn.BatchNorm2d(200, affine=True, track_running_stats=True),
            decide_act(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5),
            nn.Flatten(),
        )
        
        self.dense = nn.Sequential(
            torch.nn.Linear(8600, 2),
        )
        
    def forward(self, x):
        x = self.TotalConv(x)
        _out = self.dense(x)

        return _out


# In[7]:


batch_size = 60 
def cal_accuracy():
 
    predicted_train = []
    predicted_test = []

    for i in range(int(len(test_data)/batch_size)):
        s = i*batch_size
        e = i*batch_size+batch_size

        inputs_train = torch.from_numpy(train_data[s:e])
        pred_train = net(inputs_train.float())
        
        inputs_test = torch.from_numpy(test_data[s:e])
        pred_test = net(inputs_test.float())


        for each in pred_train.data.cpu():
            predicted_train.append(np.array(each))
        
        for each in pred_test.data.cpu():
            predicted_test.append(np.array(each))
        

    predicted_train_arr = []
    predicted_test_arr = [] 
    
    for each in predicted_train:
        if each[0] > each[1]:
            predicted_train_arr.append(0)
        else:
            predicted_train_arr.append(1)
            
    for each in predicted_test:
        if each[0] > each[1]:
            predicted_test_arr.append(0)
        else:
            predicted_test_arr.append(1)

    return accuracy_score(np.array(train_label), np.array(predicted_train_arr)), accuracy_score(np.array(test_label), np.array(predicted_test_arr))


# In[11]:


def train_model():
    _max = 0
    _epochs = []
    _acc_train = []
    _acc_test = []
    for epoch in range(500):
        for i in range(int(len(train_data)/batch_size)):

            s = i * batch_size
            e = i * batch_size + batch_size

            inputs = torch.from_numpy(train_data[s:e])
            labels = torch.from_numpy(np.array([train_label[s:e]]).reshape(batch_size, 1))

            optimizer.zero_grad()
            outputs = net(inputs.float())

            labels = labels.long()
            labels = torch.squeeze(labels)

            loss = loss_fn(outputs, labels)
            loss.backward()          
            optimizer.step()

        if epoch%10==0 :
            acc_train, acc_test = cal_accuracy()
            _epochs.append(epoch)
            _acc_train.append(acc_train)
            _acc_test.append(acc_test)
            print("Epoch: ", epoch, " ,", "Accuracy_train: ", acc_train, " ,", "Accuracy_test: ", acc_test)
            
            if acc_test >= 0.85 and acc_test > _max:
                _max = acc_test
                print("Perfect Model: ", acc_test)
                torch.save(net.state_dict(), "EEGNet_Perfect.pkl")

    acc_train, acc_test = cal_accuracy()
    _epochs.append(500)
    _acc_train.append(acc_train)
    _acc_test.append(acc_test)
    print("Epoch: ", 500, " ,", "Accuracy_train: ", acc_train, " ,", "Accuracy_test: ", acc_test)
    print("------------------------------")
    
    return _epochs, _acc_train, _acc_test


# In[134]:


_active = "ELU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.0025)
ELU_epochs, ELU_train, ELU_test = train_model()
torch.save(net.state_dict(), "EEGNet_ELU.pkl")


# In[98]:


_active = "ReLU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.0025)
ReLU_epochs, ReLU_train, ReLU_test = train_model()
torch.save(net.state_dict(), "EEGNet_ReLU.pkl")


# In[99]:


_active = "LeakyReLU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.0025)
LeakyReLU_epochs, LeakyReLU_train, LeakyReLU_test = train_model()
torch.save(net.state_dict(), "EEGNet_LeakyReLU.pkl")


# In[131]:


_active = "LeakyReLU"
net = EEGNet()
net.load_state_dict(torch.load("EEGNet_LeakyReLU.pkl"))
acc_train, acc_test = cal_accuracy()
print("Accuracy_train: ", acc_train, " ,", "Accuracy_test: ", acc_test)


# In[125]:


net.eval()


# In[138]:


plt.figure(figsize=(10,6))
plt.plot(ELU_epochs, ELU_train, label="ELU train")
plt.plot(ELU_epochs, ELU_test, label="ELU test")

plt.plot(ReLU_epochs, ReLU_train, label="ReLU train")
plt.plot(ReLU_epochs, ReLU_test, label="ReLU test")

plt.plot(LeakyReLU_epochs, LeakyReLU_train, label="LeakyReLU train")
plt.plot(LeakyReLU_epochs, LeakyReLU_test, label="LeakyReLU test")

plt.xlabel("epoch")
plt.ylabel("accuracy")

plt.title("Activate Function comparison (EEGNet)")
plt.legend(loc="best")

plt.show()


# In[12]:


_active = "ELU"
net = DeepConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.0025)
DeepConvNet_ELU_epochs, DeepConvNet_ELU_train, DeepConvNet_ELU_test = train_model()
torch.save(net.state_dict(), "DeepConvNet_ELU.pkl")



# In[13]:


_active = "ReLU"
net = DeepConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.0025)
DeepConvNet_ReLU_epochs, DeepConvNet_ReLU_train, DeepConvNet_ReLU_test = train_model()
torch.save(net.state_dict(), "DeepConvNet_ReLU.pkl")


# In[ ]:


_active = "LeakyReLU"
net = DeepConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.0025)
DeepConvNet_LeakyReLU_epochs, DeepConvNet_LeakyReLU_train, DeepConvNet_LeakyReLU_test = train_model()
torch.save(net.state_dict(), "DeepConvNet_LeakyReLU.pkl")


# In[34]:


plt.figure(figsize=(10,6))
plt.plot(DeepConvNet_ELU_epochs, DeepConvNet_ELU_train, label="ELU train")
plt.plot(DeepConvNet_ELU_epochs, DeepConvNet_ELU_test, label="ELU test")

plt.plot(DeepConvNet_ReLU_epochs, DeepConvNet_ReLU_train, label="ReLU train")
plt.plot(DeepConvNet_ReLU_epochs, DeepConvNet_ReLU_test, label="ReLU test")

plt.plot(DeepConvNet_LeakyReLU_epochs, DeepConvNet_LeakyReLU_train, label="LeakyReLU train")
plt.plot(DeepConvNet_LeakyReLU_epochs, DeepConvNet_LeakyReLU_test, label="LeakyReLU test")

plt.xlabel("epoch")
plt.ylabel("accuracy")

plt.title("Activate Function comparison (DeepConvNet)")
plt.legend(loc="best")

plt.show()


# In[10]:


_active = "ELU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005)
ELU_epochs1, ELU_train1, ELU_test1 = train_model()
# torch.save(net.state_dict(), "EEGNet_ELU.pkl")

_active = "ReLU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005)
ReLU_epochs1, ReLU_train1, ReLU_test1 = train_model()
# torch.save(net.state_dict(), "EEGNet_ReLU.pkl")

_active = "LeakyReLU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005)
LeakyReLU_epochs1, LeakyReLU_train1, LeakyReLU_test1 = train_model()
# torch.save(net.state_dict(), "EEGNet_LeakyReLU.pkl")

_active = "ELU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.001)
ELU_epochs2, ELU_train2, ELU_test2 = train_model()
# torch.save(net.state_dict(), "EEGNet_ELU.pkl")

_active = "ReLU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.001)
ReLU_epochs2, ReLU_train2, ReLU_test2 = train_model()
# torch.save(net.state_dict(), "EEGNet_ReLU.pkl")

_active = "LeakyReLU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.001)
LeakyReLU_epochs2, LeakyReLU_train2, LeakyReLU_test2 = train_model()
# torch.save(net.state_dict(), "EEGNet_LeakyReLU.pkl")

_active = "ELU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.002)
ELU_epochs3, ELU_train3, ELU_test3 = train_model()
# torch.save(net.state_dict(), "EEGNet_ELU.pkl")

_active = "ReLU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.002)
ReLU_epochs3, ReLU_train3, ReLU_test3 = train_model()
# torch.save(net.state_dict(), "EEGNet_ReLU.pkl")

_active = "LeakyReLU"
net = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.005, weight_decay = 0.002)
LeakyReLU_epochs3, LeakyReLU_train3, LeakyReLU_test3 = train_model()
# torch.save(net.state_dict(), "EEGNet_LeakyReLU.pkl")

