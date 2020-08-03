from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from clients import *
from backdoor_utils import Backdoor_Utils
import numpy as np
import random
labels=torch.tensor([0,1,2,3,4,5,6,7,8,9])
class Attacker_LabelFlipping(Client):
    def __init__(self,cid,model,dataLoader,optimizer,criterion=F.nll_loss, device='cpu',inner_epochs=1):
        super(Attacker_LabelFlipping, self).__init__(cid,model,dataLoader,optimizer,criterion, device ,inner_epochs)
    def data_transform(self,data,target):
        labels=self.dataLoader.dataset.classes
        target_=torch.tensor(list(map(lambda x: random.choice([i for i in labels if i!=x]),target)))
        assert target.shape==target_.shape, "Inconsistent target shape"
        return data,target_    
class Attacker_LabelFlippingDirectional(Client):
    def __init__(self,cid,model,dataLoader,optimizer,criterion=F.nll_loss, device='cpu',inner_epochs=1):
        super(Attacker_LabelFlippingDirectional, self).__init__(cid,model,dataLoader,optimizer,criterion, device ,inner_epochs)
    def data_transform(self,data,target):
        labels=self.dataLoader.dataset.classes
        target_=torch.tensor(list(map(lambda x: labels[7] if x==labels[1] else x,target)))
        assert target.shape==target_.shape, "Inconsistent target shape"
        return data,target_

class Attacker_Backdoor(Client):
    def __init__(self,cid,model,dataLoader,optimizer,criterion=F.nll_loss, device='cpu',inner_epochs=1):
        super(Attacker_Backdoor, self).__init__(cid,model,dataLoader,optimizer,criterion, device ,inner_epochs)
        self.utils=Backdoor_Utils()

    def data_transform(self,data,target):
        data, target = self.utils.get_poison_batch(data, target, backdoor_fraction=0.5, backdoor_label = self.utils.backdoor_label)        
        return data,target

        

        
        
class Attacker_Omniscient(Client):
    def __init__(self,cid,model,dataLoader,optimizer,criterion=F.nll_loss, device='cpu',scale=1,inner_epochs=1):
        super(Attacker_Omniscient, self).__init__(cid,model,dataLoader,optimizer,criterion, device ,inner_epochs)
        self.scale=scale
    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState=self.model.state_dict()
        for param in self.originalState:

            self.stateChange[param]=newState[param]-self.originalState[param]
            if not "FloatTensor" in self.originalState[param].type():
                continue
            self.stateChange[param]*=(-self.scale)
        self.isTrained=False