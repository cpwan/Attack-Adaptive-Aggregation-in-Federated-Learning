from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from clients import *

import random
labels=torch.tensor([0,1,2,3,4,5,6,7,8,9])
class Attacker_LabelFlipping(Client):
    def __init__(self,cid,model,dataLoader,optimizer,device):
        super(Attacker_LabelFlipping, self).__init__(cid,model,dataLoader,optimizer,device)
   
    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.dataLoader):
            target=torch.stack(list(map(lambda x: random.choice([i for i in labels if i!=x]),target)))

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('client {} ## Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.cid, self.epoch, batch_idx * len(data), len(self.dataLoader.dataset),
                    100. * batch_idx / len(self.dataLoader), loss.item()))
        self.epoch+=1
        self.isTrained=True
        
        
class Attacker_Omniscient(Client):
    def __init__(self,cid,model,dataLoader,optimizer,device,scale=10):
        super(Attacker_Omniscient, self).__init__(cid,model,dataLoader,optimizer,device)
        self.scale=scale
    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState=self.model.state_dict()
        for param in self.originalState:
            self.stateChange[param]=newState[param]-self.originalState[param]
            self.stateChange[param]*=(-self.scale)
        self.isTrained=False