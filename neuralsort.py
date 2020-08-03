import torch
from torch import Tensor

'''
The code has been modified to adapt the device types (cuda/cpu) automatically, in contrast to the original code that require cuda.

Credit:
@inproceedings{
grover2018stochastic,
title={Stochastic Optimization of Sorting Networks via Continuous Relaxations},
author={Aditya Grover and Eric Wang and Aaron Zweig and Stefano Ermon},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=H1eSS3CcKX},
}
'''

class NeuralSort (torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        one = torch.FloatTensor(dim, 1).fill_(1).to(scores)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
#         print("A",A_scores)
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
#         print("B",B)

        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
                   ).type(torch.FloatTensor).to(scores)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat).to(scores)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().type(torch.LongTensor).to(scores)
            r_idx = torch.arange(dim).repeat(
                [bsize, 1]).flatten().type(torch.LongTensor).to(scores)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P-P_hat).detach() + P_hat
        return P_hat