import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralsort import NeuralSort

class permutationMatrix(nn.Module):
    '''
    Denote tensor with [i,j,k,l]
    Assumes x has the shape batchsize* vector dimension * n * 1
    
    return
        pMat= permutation matrix of x_[k=1-n]  with shape batchsize* vector dimension * n * n
    '''
    def __init__(self):
        super(permutationMatrix, self).__init__()
        self.sort=NeuralSort()
    def forward(self,x):
        shape=(x.size(0),x.size(1),x.size(2),x.size(2))
        x=x.reshape(-1,x.shape[2]) # (* by n) 
        pMat=self.sort(x) # (* by n by n) 
        out= pMat.reshape(shape)
        return out

def inverse_diff(x,z,eps=1e-16):
    '''
    Denote tensor with [i,j,k,l]
    Assumes x has the shape batchsize* vector dimension * n * 1
    Assumes z has the shape batchsize* vector dimension * 1 * 1
    
    return
        w_ik=1/|x_ik-z_i| with shape batchsize* 1 * n * 1
    '''
    out=torch.norm(x-z,p=1,dim=1,keepdim=True)+eps
    out=1/out
    return out

def normalize(w,dim=2):
    '''
    Denote tensor with [i,j,k,l]
    weight w has the shape batchsize* 1 * n * 1, assumes positive values
    
    return
        w'_ik=w_ik/sum_k(w_ik) with shape batchsize* 1 * n * 1

    '''
    out=w/torch.sum(w,dim=dim,keepdim=True)
    return out
    
def weighted_sum(x,w):
    '''
    Denote tensor with [i,j,k,l]
    Assumes x has the shape batchsize* vector dimension * n * 1
    weight w has the shape batchsize* 1 * n * 1, assumes positive values

    return
        z_i=sum_k (x_ik*w_ik) with shape batchsize* vector dimension * 1 * 1
    '''
    out=torch.sum(x*w,dim=2,keepdim=True)
    return out

class weiszfeldBlock(nn.Module):
    '''
    Denote tensor with [i,j,k,l]
    Assumes x has the shape batchsize* vector dimension * n * 1
    Assumes z has the shape batchsize* vector dimension * 1 * 1
    
    return
        z_i=sum_k (x_ik*w_ik) with shape batchsize* vector dimension * 1 * 1
    '''
    def __init__(self,n):
        super(weiszfeldBlock, self).__init__()
        self.linear=nn.Linear(n,1)
        self.pm=permutationMatrix()
        
    def forward(self,x,z):
        pMat=self.pm(x)
        alpha=self.linear(pMat)
        # alpha should be converted to the shape batchsize* 1 * n * 1
        w=inverse_diff(x,z)*alpha
        w_n=normalize(w)
        z=weighted_sum(x,w_n)
        return z,w_n
    
#     def __init__(self,n,dim):
#         super(weiszfeldBlock, self).__init__()
#         self.n=n
#         self.dim=dim
#         self.linear=nn.Linear(n*dim,1)
#         self.pm=permutationMatrix()
        
#     def forward(self,x,z):
#         pMat=self.pm(x) # batchsize* vector dimension * n * n
#         pMat=pMat.permute(0,2,1,3) # batchsize * n * vector dimension * n
#         alpha=self.linear(pMat.view(-1,self.n,self.n*self.dim))
#         alpha=alpha.view(-1,1,self.n,1)
#         # alpha should be converted to the shape batchsize* 1 * n * 1
#         w=inverse_diff(x,z)*alpha
#         w_n=normalize(w)
#         z=weighted_sum(x,w_n)
#         return z,w_n
class Net(nn.Module):
    def __init__(self,in_dim, n):
        '''
        in_dim:=dimension of weight vector
        n:= number of clients
        '''
        super(Net, self).__init__()
        self.in_dim = in_dim
        self.n = n
        self.wb1=weiszfeldBlock(n)
        self.wb2=weiszfeldBlock(n)
        self.wb3=weiszfeldBlock(n)
        self.wb4=weiszfeldBlock(n)
        self.wb5=weiszfeldBlock(n)

        self.linear=nn.Linear(5,1)

    def forward(self, input):
#         print(input.shape)
        '''
        input: batchsize* vector dimension * n 
        '''
#         x = input.view(-1, self.n, self.in_dim)
        x=input.view(-1,self.in_dim,self.n,1)
        '''
        x    : batchsize* vector dimension * n * 1
        '''
        z=torch.mean(x,dim=2,keepdim=True) 
        z,w1=self.wb1(x,z)
        z,w2=self.wb2(x,z)
        z,w3=self.wb3(x,z)
        z,w4=self.wb4(x,z)
        z,w5=self.wb5(x,z)
        # z : batchsize* vector dimension * 1 * 1
        # wi: batchsize* 1 * n * 1
        
        w=torch.cat([w1,w2,w3,w4,w5],dim=-1) # batchsize* 1 * n * 1
        w=self.linear(w)
#         print(w.shape)
        w = torch.sigmoid(w)
#         print(x.shape)
#         x2= F.softmax(w,dim=1)
        x3 = (w>0.5).to(input)
#         print(x3.shape)

        x3 = normalize(x3)
#         pred_softmax = torch.sum(x2.view(-1,1,self.n)*input,dim=-1).unsqueeze(-1)
        pred_binary = weighted_sum(x,x3)
        out_shape=(x.shape[0],x.shape[1],1)
        return w.squeeze(), z.view(out_shape), pred_binary.view(out_shape)
     
    def forward_n(self, input, n):
        self.n=n
        return self.forward(input)

