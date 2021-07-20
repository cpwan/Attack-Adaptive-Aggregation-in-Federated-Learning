import torch
import torch.nn as nn
import torch.nn.functional as F

'''
FoolsGold

modified from https://github.com/DistributedML/FoolsGold/blob/master/deep-fg/fg/foolsgold.py

Reference:
Fung, Clement, Chris JM Yoon, and Ivan Beschastnikh. "The Limitations of Federated Learning in Sybil Settings." 23rd International Symposium on Research in Attacks, Intrusions and Defenses ({RAID} 2020). 2020.

'''

import numpy as np
import sklearn.metrics.pairwise as smp

def torch_cosine_similarity(x):
    # x: n by d
    normed=F.normalize(x,dim=1)
    res_torch=torch.matmul(normed,normed.T)
    return res_torch

# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = torch_cosine_similarity(grads) - torch.eye(n_clients).to(grads.device)
    maxcs = torch.max(cs, axis=1)[0]
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (torch.max(cs, axis=1)[0])
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / torch.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (torch.log(wv / (1 - wv)) + 0.5)
    wv[(torch.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


def adaptor(input):
    '''
    compute foolsgold 
    
    input : 1* vector dimension * n
    
    return 
        foolsGold :  vector dimension 
    '''
    x = input.squeeze(0)
    x = x.permute(1, 0)
    w = foolsgold(x)
    print(w)
    w = w / w.sum()
    out = torch.sum(x.permute(1, 0) * w, dim=1, keepdim=True)
    return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):
        #         print(input.shape)
        '''
        input: batchsize* vector dimension * n 
        (1 by d by n)
        
        return 
            out : size =vector dimension, will be flattened afterwards
        '''
        out = adaptor(input)

        return out
