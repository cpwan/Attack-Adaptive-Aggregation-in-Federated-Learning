from __future__ import print_function

import torch
import torch.nn.functional as F
'''
Modified upon
https://github.com/howardmumu/Attack-Resistant-Federated-Learning/blob/70db1edde5b4b9dfb75633ca5dd5a5a7303c1f4c/FedAvg/Update.py#L335


Reference:
Fu, Shuhao, et al. "Attack-Resistant Federated Learning with Residual-based Reweighting." arXiv preprint arXiv:1912.11464 (2019).

'''
import random

defaultPattern=[[0, 0, 0], [0, 0, 1], [0, 0, 2],   [0, 0, 4], [0, 0, 5], [0, 0, 6],\
                                 
                                 [0, 2, 0], [0, 2, 1], [0, 2, 2],   [0, 2, 4], [0, 2, 5], [0, 2, 6], ]

def getRandomPattern(k=6,seed=0):
    pattern=defaultPattern
    random.seed(seed)
    c=random.randint(0,3)
    xylim=6
    x_interval=random.randint(0,6)
    y_interval=random.randint(0,6)
    x_offset=random.randint(0,32-xylim-3)
    y_offset=random.randint(0,32-xylim-3)
    pattern=[[c,p[1]+x_offset,p[2]+y_offset] for p in pattern]
    pattern[3:6]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[3:6]))
    pattern[-3:]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[-3:]))    
    pattern[6:]=list(map(lambda p: [c,p[1]+x_interval,p[2]],pattern[6:]))      
    return list(pattern)

def getDifferentPattern(y_offset, x_offset, y_interval=1, x_interval=1):
    pattern=defaultPattern

    c=0
    xylim=6
    pattern=[[c,p[1]+x_offset,p[2]+y_offset] for p in pattern]
    pattern[3:6]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[3:6]))
    pattern[-3:]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[-3:]))    
    pattern[6:]=list(map(lambda p: [c,p[1]+x_interval,p[2]],pattern[6:]))      
    return list(pattern)

class Backdoor_Utils():

    def __init__(self):
        self.backdoor_label = 2
        self.trigger_position = torch.Tensor(defaultPattern).long()
        self.trigger_value = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ])

    def get_poison_batch(self, data, targets, backdoor_fraction, backdoor_label, evaluation=False):
        #         poison_count = 0
        new_data = torch.empty(data.shape)
        new_targets = torch.empty(targets.shape)

        for index in range(0, len(data)):
            if evaluation:  # will poison all batch data when testing
                new_targets[index] = backdoor_label
                new_data[index] = self.add_backdoor_pixels(data[index])
            #                 poison_count += 1

            else:  # will poison only a fraction of data when training
                if torch.rand(1) < backdoor_fraction:
                    new_targets[index] = backdoor_label
                    new_data[index] = self.add_backdoor_pixels(data[index])
                #                     poison_count += 1
                else:
                    new_data[index] = data[index]
                    new_targets[index] = targets[index]

        new_targets = new_targets.long()
        if evaluation:
            new_data.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_data, new_targets

    def interpolate_patterns(self,size):
        patch=torch.zeros(1,32,32)
        tp=self.trigger_position
        patch[tp[:,0],tp[:,1],tp[:,2]]=self.trigger_value
        interpolate=F.interpolate(patch.unsqueeze(0),size=(size)).squeeze(0)
        sparse=interpolate.to_sparse()
        trigger_position=sparse.indices().permute(1,0)
        trigger_value=sparse.values()
        return trigger_position, trigger_value
        
    def add_backdoor_pixels(self, item):
        tp,tv=self.interpolate_patterns(item.shape[1:])
        item[tp[:,0],tp[:,1],tp[:,2]]=tv
        return item
    
    def setTrigger(self,x_offset,y_offset,x_interval,y_interval):
        self.trigger_position=getDifferentPattern(x_offset,y_offset,x_interval,y_interval)
        
    def setRandomTrigger(self,k=6,seed=0):
        '''
        Use the default pattern if seed equals 0. Otherwise, generate a random pattern.
        '''
        if seed==0:
            return
        self.trigger_position=getRandomPattern(k,seed)