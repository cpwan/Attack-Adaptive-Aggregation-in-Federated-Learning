import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convert_pca, utils


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.main = torch.nn.Sequential(nn.Linear(in_channels, 2 * in_channels),
                                        torch.nn.LeakyReLU(),
                                        nn.Linear(2 * in_channels, out_channels))

    def forward(self, beta, x):  ##beta does nth, just for ease of using existing pipelines
        vout = x
        x = x.view(x.shape[0], -1)
        attention_scores = self.main(x)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.einsum('bk,bjk -> bj', attention_weights, vout).unsqueeze(-1)

        return out

    def getWeight(self, beta, x):
        x = x.view(x.shape[0], -1)
        attention_scores = self.main(x).unsqueeze(1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights
    
    def getScores(self, beta, x):
        x = x.view(x.shape[0], -1)
        attention_scores = self.main(x).unsqueeze(1)
        return attention_scores
    
    def getALL(self, beta, x):
        vout = x
        x = x.view(x.shape[0], -1)
        attention_scores = self.main(x)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.einsum('bk,bjk -> bj', attention_weights, vout).unsqueeze(-1)
        return out, attention_weights, attention_scores
    
class Net():
    def __init__(self):
        self.path_to_net = "./aaa/attention.pt"

    def main(self, inputs, model):
        '''
        deltas: a list of state_dicts

        return 
            Delta: robustly aggregated state_dict

        '''

        proj_vec,deltas=inputs

        print(proj_vec.shape)
        model = MLP(proj_vec.shape[0] * 10, 10)
        model.load_state_dict(torch.load(self.path_to_net)['state_dict'])
        model.eval()

        x = proj_vec.unsqueeze(0)
        beta = x.median(dim=-1, keepdims=True)[0]
        weight = model.getWeight(beta, x)
        weight = F.normalize(weight, p=1, dim=-1)

        weight = weight[0, 0, :]
        print(weight)

        Delta = utils.applyWeight2StateDicts(deltas, weight)

        return Delta
