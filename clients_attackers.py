from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from clients import *
from backdoor_utils import Backdoor_Utils

import random
labels=torch.tensor([0,1,2,3,4,5,6,7,8,9])
class Attacker_LabelFlipping(Client):
    def __init__(self,cid,model,dataLoader,optimizer,device):
        super(Attacker_LabelFlipping, self).__init__(cid,model,dataLoader,optimizer,device)
    def data_transform(self,data,target):
        target=torch.stack(list(map(lambda x: random.choice([i for i in labels if i!=x]),target)))
        return data,target
    
class Attacker_LabelFlippingDirectional(Client):
    def __init__(self,cid,model,dataLoader,optimizer,device):
        super(Attacker_LabelFlippingDirectional, self).__init__(cid,model,dataLoader,optimizer,device)
    def data_transform(self,data,target):
        target=torch.stack(list(map(lambda x: labels[7] if x==labels[1] else x,target)))
        return data,target

class Attacker_Backdoor(Client):
    def __init__(self,cid,model,dataLoader,optimizer,device):
        super(Attacker_Backdoor, self).__init__(cid,model,dataLoader,optimizer,device)
        self.utils=Backdoor_Utils()

    def data_transform(self,data,target):
        data, target = self.utils.get_poison_batch(data, target, backdoor_fraction=0.5, backdoor_label = self.utils.backdoor_label)        
        return data,target

        

        
        
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