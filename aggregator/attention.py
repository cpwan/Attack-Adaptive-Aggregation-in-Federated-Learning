import utils
import convert_pca

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math

class nonLinearity(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(nonLinearity, self).__init__()
        self.main = torch.nn.Sequential(nn.Conv1d(in_channels, 2 * in_channels, kernel_size=1, bias=bias),
                torch.nn.LeakyReLU(),
                nn.Conv1d(2 * in_channels, out_channels, kernel_size=1, bias=bias)) 

    def forward(self,x):
        out = self.main(x)
        return out

class Affinity(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, self_attention=True, eps=0.001, scale=10):
        super(Affinity, self).__init__()
        
        self.key_conv = nonLinearity(in_channels, out_channels, bias=bias)
        if self_attention:
            self.query_conv = self.key_conv
        else:
            self.query_conv = nonLinearity(in_channels, out_channels, bias=bias)

        self.eps=eps
        self.scale=scale
        self.Threshold=nn.Threshold(self.eps,0) # if n=10, no attackers, then each clients may get 0.1 , any client with lower than 1% of 0.1 will be discarded
        
    def forward(self,query,key):
        q_out = self.query_conv(query)
        k_out = self.key_conv(key)

        q_out = F.normalize(q_out,dim=1)
        k_out = F.normalize(k_out,dim=1)
        attention_scores = torch.bmm(q_out.transpose(1, 2), k_out)
        attention_scores*=self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_weights=self.Threshold(attention_weights)
        return attention_weights


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, eps=0.001, scale=10):
        super(AttentionConv, self).__init__()

        self.affinity = Affinity(in_channels, out_channels, bias=bias, eps=eps, scale=scale)
      # self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1,
      # bias=bias)

    def forward(self, query,key):

        v_out = key#self.value_conv(key)
        attention_weights = self.affinity(query,key)
        out = torch.einsum('bqi,bji -> bjq', attention_weights, v_out)

        return out, attention_weights



class AttentionLoop(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, nloop=2, eps=0.001, scale=10):
        super(AttentionLoop, self).__init__()

        self.attention = AttentionConv(in_channels, out_channels, bias=bias, eps=eps, scale=scale)
        self.nloop = nloop
    def forward(self, query,key):
        x = query
        for i in range(self.nloop):
            x,w  = self.attention(x,key)
        out = x
        return out
    def getWeight(self,query,key):
        x = query
        for i in range(self.nloop):
            x,w  = self.attention(x,key)
        out = w
        return out



class Net():
    def __init__(self, eps=0.001, scale=10):
        self.hidden_size = 21
        self.path_to_net = "./aggregator/attention.pt"
        self.eps=eps
        self.scale=scale
    
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
        model = AttentionLoop(proj_vec.shape[0], self.hidden_size,bias=False, nloop=5, eps=self.eps, scale=self.scale)
        model.load_state_dict(torch.load(self.path_to_net))
        model.eval()

        
        x = proj_vec.unsqueeze(0)
        beta = x.median(dim=-1,keepdims=True)[0]
        beta = x.mean(dim=-1,keepdims=True)

#         weight = model.attention.affinity(beta,x)
#         weight = torch.nn.Threshold(0.5 * 1.0 / weight.shape[-1],0)(weight)
#         weight = F.normalize(weight,p=1,dim=-1)
        weight = model.getWeight(beta,x)
#         weight = torch.nn.Threshold( 0.8*1.0 / weight.shape[-1],0)(weight)
        weight = F.normalize(weight,p=1,dim=-1)
        
        weight = weight[0,0,:]
        print(weight)
        
        Delta = utils.applyWeight2StateDicts(deltas,weight)
#         print(Delta)


        return Delta
