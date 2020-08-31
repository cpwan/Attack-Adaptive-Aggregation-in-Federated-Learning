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
    def __init__(self, in_channels, out_channels, bias=False, self_attention=True):
        super(Affinity, self).__init__()

        ##baseline
        # self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1,
        # bias=bias)
        # self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1,
        # bias=bias)
        ##non linearity
        self.key_conv = nonLinearity(in_channels, out_channels, bias=bias)
        if self_attention:
            self.query_conv = self.key_conv
        else:
            self.query_conv = nonLinearity(in_channels, out_channels, bias=bias)

    def forward(self,query,key):
        q_out = self.query_conv(query)
        k_out = self.key_conv(key)

        q_out = F.normalize(q_out,dim=1)
        k_out = F.normalize(k_out,dim=1)
        attention_scores = torch.bmm(q_out.transpose(1, 2), k_out)
        attention_scores*=10
        attention_weights = F.softmax(attention_scores, dim=-1)

        return attention_weights


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(AttentionConv, self).__init__()

        self.affinity = Affinity(in_channels, out_channels, bias=False)
      # self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1,
      # bias=bias)

    def forward(self, query,key):

        ## non residual
        v_out = key#self.value_conv(key)
        attention_weights = self.affinity(query,key)
        out = torch.einsum('bqi,bji -> bjq', attention_weights, v_out)
        # ## residual
        #     v_out=key#self.value_conv(key)
        #     attention_weights=self.affinity(query-query,key-query)
        #     out = torch.einsum('bqi,bji -> bjq', attention_weights, v_out)

        return out



class AttentionLoop(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, nloop=2):
        super(AttentionLoop, self).__init__()

        self.attention = AttentionConv(in_channels, out_channels, bias=False)
        self.nloop = nloop
    def forward(self, query,key):
        x = query
        for i in range(self.nloop):
            x = self.attention(x,key)
        out = x
        return out



class Net():
    def __init__(self):
        self.hidden_size = 11
        self.path_to_net = "./aggregator/attention_sb.pt"
    
    def main(self,deltas:list):
        '''
        deltas: a list of state_dicts

        return 
            Delta: robustly aggregated state_dict

        '''

        
        stacked = utils.stackStateDicts(deltas)
        proj_vec = convert_pca._convertWithPCA(stacked)

        model = AttentionLoop(proj_vec.shape[0], self.hidden_size, nloop=5)
        model.load_state_dict(torch.load(self.path_to_net))
        model.eval()

        
        x = proj_vec.unsqueeze(0)
        beta = x.median(dim=-1,keepdims=True)[0]
        weight = model.attention.affinity(beta,x)
        weight = torch.nn.Threshold(0.5 * 1.0 / weight.shape[-1],0)(weight)
        weight = F.normalize(weight,p=1,dim=-1)
        
        weight = weight[0,0,:]
        print(weight)
        
        Delta = utils.applyWeight2StateDicts(deltas,weight)
#         print(Delta)


        return Delta
