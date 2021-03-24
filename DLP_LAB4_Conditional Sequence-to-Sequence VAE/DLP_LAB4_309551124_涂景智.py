#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
from dataloader import Dataloader, Char_to_Num


# In[2]:


_data = Dataloader()


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 15
#----------Hyper Parameters----------#
hidden_size = 256
latent_size = 32
# learning_rate = 0.05
#The number of vocabulary
vocab_size = 28
teacher_forcing_ratio = 0.7
empty_input_ratio = 0.1
KLD_weight = 0.0
LR = 0.05


# In[5]:


def set_KLD_weight(mode, epoch):
    if mode==0:
        if epoch < 20000:
            KLD_weight =  epoch/20000
        else:
            KLD_weight = 1.0
    else:
        e = epoch/5
        if  e < 10000:
            KLD_weight = e/10000
        else:
            KLD_weight = 1.0

    return KLD_weight




def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def one_hot(label):
    if label == 0:
        target = [1,0,0,0]
    elif label == 1:
        target = [0,1,0,0]
    elif label == 2:
        target = [0,0,1,0]
    else:
        target = [0,0,0,1]
    return torch.Tensor(target).long()


def loss_function(recon_x, x, mu, logvar, epoch):
    _fn = nn.CrossEntropyLoss(reduction='sum')
    CE = _fn(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    KLD_weight = set_KLD_weight(0, epoch)
    return CE + KLD*KLD_weight, KLD


# In[6]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, latent_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)
        self.fc_c = nn.Linear(hidden_size, latent_size)
            
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(len(input), 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        z_mu = self.fc1(hidden[0])
        z_var = self.fc2(hidden[0])
        z = self.reparameterize(z_mu, z_var)
        c_0 = self.fc_c(hidden[1])
        return output, (z, c_0), z_mu, z_var

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),torch.zeros(1, 1, self.hidden_size, device=device))
    
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(latent_size + 4 , hidden_size)
        self.lstm = nn.LSTM(hidden_size, latent_size + 4)
        self.out = nn.Linear(latent_size + 4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),torch.zeros(1, 1, self.hidden_size, device=device))


# In[22]:


def train(input_tensor, target_tensor, input_label, target_label, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,  epoch, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_label = one_hot(input_label)
    target_label = one_hot(target_label).view(1, 1, -1)
    target_length = target_tensor.size(0)
    
    tmp = [[float(x)] for x in target_tensor]
    target_tensor = torch.Tensor(tmp).long()
    
    input_tensor = torch.cat((input_tensor, input_label),0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    KLD_loss = 0
    
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    target_label = target_label.to(device)

    #----------sequence to sequence part for encoder----------#
    encoder_output, encoder_hidden, mu, logvar = encoder(input_tensor, encoder_hidden)


    decoder_input = torch.tensor([[SOS_token]], device=device)
    
    new_h0 = torch.cat((encoder_hidden[0], target_label), 2).to(device)
    new_c0 = torch.cat((encoder_hidden[1], target_label), 2).to(device)

#     decoder_hidden = encoder_hidden
    decoder_hidden = (new_h0, new_c0)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            tmp_loss, tmp_KLD_loss = loss_function(decoder_output, target_tensor[di], mu, logvar, epoch)
            loss += tmp_loss
            KLD_loss += tmp_KLD_loss
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            tmp_loss, tmp_KLD_loss = loss_function(decoder_output, target_tensor[di], mu, logvar, epoch)
            loss += tmp_loss
            KLD_loss += tmp_KLD_loss

            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, KLD_loss.item() / target_length


# In[23]:


def test(input_word, target_word, input_label, target_label, encoder, decoder):
    encoder.eval()
    decoder.eval()
    
    encoder_hidden = encoder.initHidden()
    
    input_tensor = Char_to_Num(input_word)
    target_tensor = Char_to_Num(target_word)
    input_tensor = torch.LongTensor(input_tensor)
    target_tensor = torch.LongTensor(target_tensor)
    
    input_label = one_hot(input_label)
    target_label = one_hot(target_label).view(1, 1, -1)
    target_length = target_tensor.size(0)
    
    tmp = [[float(x)] for x in target_tensor]
    target_tensor = torch.Tensor(tmp).long()
    
    input_tensor = torch.cat((input_tensor, input_label),0)


    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    target_label = target_label.to(device)

    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    encoder_output, encoder_hidden, mu, logvar = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    
    new_h0 = torch.cat((encoder_hidden[0], target_label), 2).to(device)
    new_c0 = torch.cat((encoder_hidden[1], target_label), 2).to(device)

    decoder_hidden = (new_h0, new_c0)
    
    output_num = []
    output_char = ""
    
#    for di in range(target_length):
#        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#        topv, topi = decoder_output.topk(1)
#        output_num.append(topi.squeeze().detach())
#        decoder_input = target_tensor[di]

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        output_num.append(topi.squeeze().detach())
        decoder_input = topi.squeeze().detach()  # detach from history as input
    
    for each in output_num:
        output_char += chr(each+95)
        
    return output_char
        


# In[24]:


def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = "train.txt"
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)


def Generate_words(encoder, decoder):
    _return = []   
    for _ in range(100):
        idx = random.randint(0, len(_data.total_word)-1)
        tmp = []
        for i in range(4):
            ans = test(_data.total_word[idx][i], _data.total_word[idx][i], i, i, encoder, decoder)
            tmp.append(ans)
            
        _return.append(tmp)
            
    return _return

def Show_bleu(encoder, decoder):
    yourpath = "test.txt"
    word_label = [[0, 3], [0 ,2], [0, 1], [0, 1], [3, 1], [0, 2], [3, 0], [2, 0], [2, 3], [2, 1]]
    idx = 0
    score = 0
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[1] = word[1].strip('\n')
            ans = test(word[0],word[1], word_label[idx][0], word_label[idx][1], encoder, decoder)
            score += compute_bleu(word[1], ans)
            idx += 1
    
    return score/10  


# In[25]:


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=1000, learning_rate=0.005):
    start = time.time()
    plot_losses = []
    plot_KLD= []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    plot_KLD_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    max_BLUE = 0.0  
    max_Gaussion  = 0.0
#     encoder_hidden = encoder.initHidden()

    for iter in range(1, n_iters + 1):
        encoder.train()
        decoder.train()
        input_tensor, input_label, target_tensor, target_label = _data.return_training_pair()
        input_tensor = torch.LongTensor(input_tensor)
        target_tensor = torch.LongTensor(target_tensor)

        loss, KLD_loss = train(input_tensor, target_tensor, input_label, target_label, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, iter)
        print_loss_total += loss
        plot_loss_total += loss
        plot_KLD_total  +=  KLD_loss

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            plot_KLD_avg = plot_KLD_total / plot_every
            plot_KLD.append(plot_KLD_avg)
            plot_KLD_total = 0

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            
        if iter % 5000 ==0 and iter != 0 :
          
            words = Generate_words(encoder, decoder)
            b_score = Show_bleu(encoder, decoder)
            g_score = Gaussian_score(words)
            print("Average BLUE-4 score: " + str(b_score))
            print("Gaussian score: " + str(g_score))

            if b_score > max_BLUE:
                torch.save(encoder.state_dict(), 'BLUE_encoder.pkl')
                torch.save(decoder.state_dict(), 'BLUE_decoder.pkl')

            if g_score > max_Gaussion:
                torch.save(encoder.state_dict(), 'Gaussion_encoder.pkl')
                torch.save(decoder.state_dict(), 'Gaussion_decoder.pkl')
    
    print(plot_losses)
    print(plot_KLD)


# In[26]:


encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
trainIters(encoder1, decoder1, 100000, print_every=1000, plot_every=1000)


# In[ ]:




