import torch
import torch.nn as nn
import torch.nn.functional as F


class block(nn.Module):
    def __init__(self,in_,out_):
        super(block, self).__init__()
        self.main=torch.nn.Sequential(
                    nn.Conv2d(in_, out_, kernel_size=1),
                    torch.nn.BatchNorm2d(out_),
                    nn.ReLU(),
                    )
    def forward(self,x):
        out=self.main(x)
        return out
    
class block_no_activation(nn.Module):
    def __init__(self,in_,out_):
        super(block_no_activation, self).__init__()
        self.main=torch.nn.Sequential(
                    nn.Conv2d(in_, out_, kernel_size=1),
                    torch.nn.BatchNorm2d(out_),
                    )
    def forward(self,x):
        out=self.main(x)
        return out
    
class Net(nn.Module):
    def __init__(self,in_dim, n):
        '''
        in_dim:=dimension of weight vector
        n:= number of clients
        '''
        super(Net, self).__init__()
        self.in_dim=in_dim
        self.n=n
        

    def forward(self, input):
#         print(input.shape)
        '''
        input: batch size x window dim x num clients
        '''
#         x = input.view(-1, self.n, self.in_dim)
        x=input.view(-1,self.in_dim,self.n,1)
        '''
        x    : batch size x window dim x num clients x 1
        '''
        x = torch.zeros(x.shape[0],x.shape[2]).to(input)
#         pred=dot_product(input,x).squeeze(-1)
        x2 = F.softmax(x,dim=1)
        x3 = (x>0.5).float().to(input)
        x3 = x3/(torch.sum(x3,-1).view(-1,1)+1e-14)
        pred_softmax = torch.sum(x2.view(-1,1,self.n)*input,dim=-1).unsqueeze(-1)
        pred_binary = torch.sum(x3.view(-1,1,self.n)*input,dim=-1).unsqueeze(-1)
        return x, pred_softmax, pred_binary
     
    def forward_n(self, input, n):
        self.n=n
        self.forward(input)

