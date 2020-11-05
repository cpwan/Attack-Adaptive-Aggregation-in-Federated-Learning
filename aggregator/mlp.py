import utils
import convert_pca

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.main = torch.nn.Sequential(nn.Linear(in_channels, 2 * in_channels),
                torch.nn.LeakyReLU(),
                nn.Linear(2 * in_channels, out_channels))

    def forward(self, beta, x): ##beta does nth, just for ease of using existing pipelines
        vout=x
#         print(x.shape)
        x=x.view(x.shape[0],-1)
#         print(x.shape)
        attention_scores = self.main(x)
        attention_weights = F.softmax(attention_scores, dim=-1)
#         print(vout.shape)
#         print(attention_weights.shape)
        out = torch.einsum('bk,bjk -> bj' ,attention_weights,vout).unsqueeze(-1)

        return out
    
    def getWeight(self, beta, x):
#         print(x.shape)
        x=x.view(x.shape[0],-1)
#         print(x.shape)
        attention_scores = self.main(x).unsqueeze(1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights




class Net():
    def __init__(self):
        self.path_to_net = "./aggregator/attention.pt"

    
    def main(self,deltas:list,model):
        '''
        deltas: a list of state_dicts

        return 
            Delta: robustly aggregated state_dict

        '''


        
        stacked = utils.stackStateDicts(deltas)
        
        param_trainable=utils.getTrainableParameters(model)
        param_nontrainable=[param for param in stacked.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del stacked[param]
        
        proj_vec = convert_pca._convertWithPCA(stacked)
        
        
        
        
        
        print(proj_vec.shape)
        model = MLP(proj_vec.shape[0]*10, 10)
        model.load_state_dict(torch.load(self.path_to_net))
        model.eval()

        
        x = proj_vec.unsqueeze(0)
        beta = x.median(dim=-1,keepdims=True)[0]

#         weight = model.attention.affinity(beta,x)
#         weight = torch.nn.Threshold(0.5 * 1.0 / weight.shape[-1],0)(weight)
#         weight = F.normalize(weight,p=1,dim=-1)
        weight = model.getWeight(beta, x)
#         weight = torch.nn.Threshold( 0.8*1.0 / weight.shape[-1],0)(weight)
        weight = F.normalize(weight,p=1,dim=-1)
        
        weight = weight[0,0,:]
        print(weight)
        
        Delta = utils.applyWeight2StateDicts(deltas,weight)
#         print(Delta)


        return Delta
