from __future__ import print_function

import torch

'''
Modified upon
https://github.com/howardmumu/Attack-Resistant-Federated-Learning/blob/70db1edde5b4b9dfb75633ca5dd5a5a7303c1f4c/FedAvg/Update.py#L335


Reference:
Fu, Shuhao, et al. "Attack-Resistant Federated Learning with Residual-based Reweighting." arXiv preprint arXiv:1912.11464 (2019).

'''
import random
# def getRandomPattern(k=6,seed=0):
#     random.seed(seed)
#     c_range=[0,1,2]
#     xylim=6
#     x_range=list(range(xylim))
#     y_range=list(range(xylim))
#     x_offset=random.randint(0,32-xylim)
#     y_offset=random.randint(0,32-xylim)
#     x_range=list(map(lambda u: u+x_offset, x_range))
#     y_range=list(map(lambda u: u+y_offset, y_range))
    
#     combo = [c_range, x_range, y_range]
#     pattern = set()
#     while len(pattern) < k:
#         elem = tuple([random.choice(comp) for comp in combo])
#         pattern.add(elem)

#     return list(pattern)

def getRandomPattern(k=6,seed=0):
    pattern=[[0, 0, 0], [0, 0, 1], [0, 0, 2],   [0, 0, 4], [0, 0, 5], [0, 0, 6],\
                                 
                                 [0, 2, 0], [0, 2, 1], [0, 2, 2],   [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
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
    pattern=[[0, 0, 0], [0, 0, 1], [0, 0, 2],   [0, 0, 4], [0, 0, 5], [0, 0, 6],\
                                 
                                 [0, 2, 0], [0, 2, 1], [0, 2, 2],   [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
#     random.seed(seed)
#     c=random.randint(0,3)
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
        self.trigger_position = [[0, 0, 0], [0, 0, 1], [0, 0, 2],   [0, 0, 4], [0, 0, 5], [0, 0, 6],\
                                 
                                 [0, 2, 0], [0, 2, 1], [0, 2, 2],   [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
        self.trigger_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]

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

    def add_backdoor_pixels(self, item):

        for i in range(0, len(self.trigger_position)):
            pos = self.trigger_position[i]
            item[pos[0]][pos[1]][pos[2]] = self.trigger_value[i]
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