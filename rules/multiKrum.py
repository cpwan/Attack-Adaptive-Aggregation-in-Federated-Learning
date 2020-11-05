import torch
import torch.nn as nn

'''
Krum aggregation
- find the point closest to its neignborhood

Reference:
Blanchard, Peva, Rachid Guerraoui, and Julien Stainer. "Machine learning with adversaries: Byzantine tolerant gradient descent." Advances in Neural Information Processing Systems. 2017.
`https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf`

'''


def getKrum(input):
    '''
    compute krum or multi-krum of input. O(dn^2)
    
    input : batchsize* vector dimension * n
    
    return 
        krum : batchsize* vector dimension * 1
        mkrum : batchsize* vector dimension * 1
    '''

    n = input.shape[-1]
    f = n // 2  # worse case 50% malicious points
    k = n - f - 2

    # collection distance, distance from points to points
    x = input.permute(0, 2, 1)
    cdist = torch.cdist(x, x, p=2)
    # find the k+1 nbh of each point
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    # the point closest to its nbh
    i_star = torch.argmin(nbhDist.sum(2))
    # krum
    krum = input[:, :, [i_star]]
    # Multi-Krum
    mkrum = input[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)
    return krum, mkrum


class Net(nn.Module):
    def __init__(self, mode='mkrum'):
        super(Net, self).__init__()
        assert (mode in ['krum', 'mkrum'])
        self.mode = mode

    def forward(self, input):
        #         print(input.shape)
        '''
        input: batchsize* vector dimension * n 
        
        return 
            out : batchsize* vector dimension * 1
        '''
        krum, mkrum = getKrum(input)

        out = krum if self.mode == 'krum' else mkrum

        return out
