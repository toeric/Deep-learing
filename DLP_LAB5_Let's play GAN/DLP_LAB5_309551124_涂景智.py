import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from dataloader import Data_Loader, one_hot
from torch.utils.data import Dataset, DataLoader
from evaluator import evaluation_model
import json



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


evm = evaluation_model(device)

train_data = Data_Loader("lab5_dataset/iclevr")
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)


batch_size = 32
nz = 25
nc = 3
nclass = 24
lr = 0.0002
epochs = 75


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + nclass, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, x, attr):
        attr = attr.view(-1, nclass, 1, 1)
        x = torch.cat([x, attr], 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_input = nn.Linear(nclass, 64 * 64)
        self.main = nn.Sequential(
            nn.Conv2d(nc + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
    
    def forward(self, x, attr):
        attr = self.feature_input(attr).view(-1, 1, 64, 64)
        x = torch.cat([x, attr], 1)
        return self.main(x).view(-1, 1)

netD = Discriminator().to(device)
netG = Generator().to(device)
criterion = nn.MSELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.00001, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
max_score = 0

def train():
    
    noise = torch.FloatTensor(batch_size, nz, 1, 1).to(device)
    label_real = torch.FloatTensor(batch_size, 1).fill_(1).to(device)
    label_fake = torch.FloatTensor(batch_size, 1).fill_(0).to(device)
    

    for epoch in range(epochs):
        netG.train()
        netD.train()
        for i, (data, attr) in enumerate(train_loader):

            netD.zero_grad()
            
            _batch_size = data.size(0)
            if _batch_size !=32:
                continue
#            label_real.data.resize(_batch_size, 1).fill_(1)
#            label_fake.data.resize(_batch_size, 1).fill_(0)
            noise.data.resize_(_batch_size, nz, 1, 1).normal_(0, 1)

            data = data.to(device)
            attr = attr.to(device)

            d_real = netD(data, attr)
            fake = netG(noise, attr) 
            d_fake = netD(fake.detach(), attr)

            d_loss = criterion(d_real, label_real) + criterion(d_fake, label_fake)
            d_loss.backward()
            optimizerD.step()

            netG.zero_grad()
            d_fake = netD(fake, attr)
            g_loss = criterion(d_fake, label_real) # trick the fake into being real
            g_loss.backward()
            optimizerG.step()
            
            
#            if i%250 == 0:
#                 print("i{} d_real: {}, d_fake: {}".format(i, d_real.mean(), d_fake.mean()))
#                print(str(i) + "th, ", str(float(g_loss.data)))
        

#         print("epoch{:03d} d_real: {}, d_fake: {}".format(epoch, d_real.mean(), d_fake.mean()))
        test(epoch)
    print(max_score)

def test(epoch):
   
    global max_score
    netG.eval()
    netD.eval()
    
    fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).to(device)
    input_file = open('lab5_dataset/test.json')
    test_attr = json.load(input_file)
    
    all_attr = []
    
    for each in test_attr:
        attr = one_hot(each)
        all_attr.append(attr.tolist())
    
    all_attr = torch.FloatTensor(all_attr).to(device)
    fixed_noise.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
    fake = netG(fixed_noise, all_attr) 
    

    ans = evm.eval(fake, all_attr)
    print("Epoch: " + str(epoch) + ", Score: " + str(ans))
    
    if ans > max_score:
        max_score = ans
        torch.save(netD.state_dict(), 'netD.pkl')
        torch.save(netG.state_dict(), 'netG.pkl')
    vutils.save_image(fake.data, '{}/epoch_{:03d}.png'.format("output_image", epoch ), normalize=True)
    

        
train()
