import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convert_pca, utils

layer_init_noise = 1e-3
init_weight = True

class nonLinearity(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(nonLinearity, self).__init__()
        self.in_channels = in_channels
        self.main = torch.nn.Sequential(nn.Conv1d(in_channels, 2 * in_channels, kernel_size=1, bias=bias),
                                        torch.nn.LeakyReLU(),
                                        nn.Conv1d(2 * in_channels, out_channels, kernel_size=1, bias=bias))
        if init_weight:
            self.init_weight()
        
    def init_weight(self):
        sigma=layer_init_noise
        
        
        device=self.main[0].weight.device
        iiv=torch.cat([torch.eye(self.in_channels), -torch.eye(self.in_channels)],dim=0).unsqueeze(-1)
        self.main[0].weight.data=iiv + torch.randn(self.main[0].weight.shape)*sigma
        self.main[0].weight.data=self.main[0].weight.data.to(device)
        
    def forward(self, x):
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

        self.eps = eps
        self.scale = scale
        self.Threshold = nn.Threshold(self.eps,
                                      0)  # if n=10, no attackers, then each clients may get 0.1 , any client with lower than 1% of 0.1 will be discarded

    def forward(self, query, key):
        q_out = self.query_conv(query)
        k_out = self.key_conv(key)

        q_out = F.normalize(q_out, dim=1)
        k_out = F.normalize(k_out, dim=1)
        align_scores = torch.bmm(q_out.transpose(1, 2), k_out)
        scaled_scores = align_scores * self.scale
        attention_weights = F.softmax(scaled_scores, dim=-1)

        n=k_out.shape[-1] # number of clients
        attention_weights = self.Threshold(attention_weights*n)/n
        return attention_weights, align_scores


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, eps=0.001, scale=10,multihead=False):
        super(AttentionConv, self).__init__()
        
        
        self.affinity = Affinity(in_channels, out_channels, bias=bias, eps=eps, scale=scale)

        # experimental
        self.multihead=multihead
        if multihead:
            self.affinity2 = Affinity(in_channels, out_channels, bias=bias, eps=eps, scale=scale)
            self.affinity3 = Affinity(in_channels, out_channels, bias=bias, eps=eps, scale=scale)
            self.pooling=nn.Conv1d(3, 1, kernel_size=1, bias=False)
    # self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1,
    # bias=bias)

    def forward(self, query, key):
        ## original
#         v_out = key  # self.value_conv(key)
#         attention_weights, align_scores = self.affinity(query, key)
#         out = torch.einsum('bqi,bji -> bjq', attention_weights, v_out)

        v_out = key  # self.value_conv(key)
        attention_weights, align_scores = self.affinity(query, key)
        if self.multihead:
            attention_weights2, align_scores2 = self.affinity2(query, key)
            attention_weights3, align_scores3 = self.affinity3(query, key)

            attention_weights=self.pooling(torch.cat([attention_weights,
                                    attention_weights2,
                                    attention_weights3],dim=1))
            align_scores=self.pooling(torch.cat([align_scores,
                            align_scores2,
                            align_scores3],dim=1))
        out = torch.einsum('bqi,bji -> bjq', attention_weights, v_out)

        return out, attention_weights, align_scores


class AttentionLoop(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, nloop=2, eps=0.001, scale=10,multihead=False):
        super(AttentionLoop, self).__init__()
        self.hparam={
            'in_channels':in_channels,
            'out_channels':out_channels,
            'bias':bias,
            'nloop':nloop,
            'eps':eps,
            'scale':scale,
            'multihead':multihead
        }

        
        self.attention = AttentionConv(in_channels, out_channels, bias=bias, eps=eps, scale=scale,multihead=multihead)
        self.nloop=nloop

    def forward(self, query, key):
        x = query
        for i in range(self.nloop):
            x, w, s = self.attention(x, key)
        out = x
        return out

    def getWeight(self, query, key):
        x = query
        for i in range(self.nloop):
            x, w, s = self.attention(x, key)
        out = w
        return out
    
    def getScores(self, query, key):
        x = query
        for i in range(self.nloop):
            x, w, s = self.attention(x, key)
        out = s
        return out
    
    def getALL(self, query, key):
        x = query
        for i in range(self.nloop):
            x, w, s = self.attention(x, key)
        out = s
        return x,w,s
    
    

class Net():
    def __init__(self, pretrain_path=None):
        assert pretrain_path!=None, 'Path to pretrained model is requried!'
        dicts=torch.load(pretrain_path,map_location='cpu')
        state_dict=dicts['state_dict']
        hparam=dicts['hparam']
        self.model=AttentionLoop(*hparam.values())
        self.model.load_state_dict(state_dict)

    def main(self, inputs, model):
        '''
        inputs: (stacked state_dicts, a list of state_dicts)

        return 
            Delta: robustly aggregated state_dict

        '''

#         stacked = utils.stackStateDicts(deltas)

#         param_trainable = utils.getTrainableParameters(model)
#         param_nontrainable = [param for param in stacked.keys() if param not in param_trainable]
#         for param in param_nontrainable:
#             del stacked[param]
        proj_vec,deltas=inputs

#         print(proj_vec.shape)
        model = self.model
        model.eval()

        x = proj_vec.unsqueeze(0)
#         beta = x.median(dim=-1, keepdims=True)[0]
        beta = x.mean(dim=-1, keepdims=True)
        weight = model.getWeight(beta, x)
        weight = F.normalize(weight, p=1, dim=-1)

        weight = weight[0, 0, :]
        print(weight)

        Delta = utils.applyWeight2StateDicts(deltas, weight)
        #         print(Delta)

        return Delta
