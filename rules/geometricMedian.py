import torch
import torch.nn as nn

'''
Geometric median aggregation
- find the geometric median with Weiszfeld’s algorithm
- Weiszfeld’s algorithm
- Input: {x_i} , output: z
- 0. z = mean{x_i}
- 1. Repeat:
-     w_i = 1/||x_i - z||_1
-     w_i = w_i/sum{w_i}
-     z   = sum{ w_i * x_i }

Reference:
Pillutla, Krishna, Sham M. Kakade, and Zaid Harchaoui. "Robust aggregation for federated learning." arXiv preprint arXiv:1912.13445 (2019).
`https://arxiv.org/pdf/1912.13445.pdf`

'''


def inverse_diff(x, z, eps=1e-16):
    '''
    Denote tensor with [i,j,k,l]
    Assumes x has the shape batchsize* vector dimension * n * 1
    Assumes z has the shape batchsize* vector dimension * 1 * 1
    
    return
        w_ik=1/|x_ik-z_i| with shape batchsize* 1 * n * 1
    '''
    out = torch.norm(x - z, p=1, dim=1, keepdim=True) + eps
    out = 1 / out
    return out


def normalize(w, dim=2):
    '''
    Denote tensor with [i,j,k,l]
    weight w has the shape batchsize* 1 * n * 1, assumes positive values
    
    return
        w'_ik=w_ik/sum_k(w_ik) with shape batchsize* 1 * n * 1

    '''
    out = w / torch.sum(w, dim=dim, keepdim=True)
    return out


def weighted_sum(x, w):
    '''
    Denote tensor with [i,j,k,l]
    Assumes x has the shape batchsize* vector dimension * n * 1
    weight w has the shape batchsize* 1 * n * 1, assumes positive values

    return
        z_i=sum_k (x_ik*w_ik) with shape batchsize* vector dimension * 1 * 1
    '''
    out = torch.sum(x * w, dim=2, keepdim=True)
    return out


class weiszfeldBlock(nn.Module):
    '''
    Denote tensor with [i,j,k,l]
    Assumes x has the shape batchsize* vector dimension * n * 1
    Assumes z has the shape batchsize* vector dimension * 1 * 1
    
    return
        z_i=sum_k (x_ik*w_ik) with shape batchsize* vector dimension * 1 * 1
    '''

    def forward(self, x, z):
        alpha = 1.0
        # alpha should be converted to the shape batchsize* 1 * n * 1
        w = inverse_diff(x, z) * alpha
        w_n = normalize(w)
        z = weighted_sum(x, w_n)
        return z, w_n


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.wb1 = weiszfeldBlock()
        self.wb2 = weiszfeldBlock()
        self.wb3 = weiszfeldBlock()
        self.wb4 = weiszfeldBlock()
        self.wb5 = weiszfeldBlock()

    def forward(self, input):
        #         print(input.shape)
        '''
        input: batchsize* vector dimension * n 
        
        return 
            out : batchsize* vector dimension * 1
        '''
        x = input.unsqueeze(-1)
        '''
        x    : batchsize* vector dimension * n * 1
        '''
        z = torch.mean(x, dim=2, keepdim=True)
        z, w1 = self.wb1(x, z)
        z, w2 = self.wb2(x, z)
        z, w3 = self.wb3(x, z)
        z, w4 = self.wb4(x, z)
        z, w5 = self.wb5(x, z)

        out_shape = (x.shape[0], x.shape[1], 1)
        out = z.view(out_shape)
        return out
