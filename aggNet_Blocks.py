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
    
class pointNetBlock(nn.Module):
    def __init__(self,in_dim,n,out=1):
        super(pointNetBlock, self).__init__()
        self.in_dim = in_dim
        self.n = n
        self.out=out
 
        self.local = torch.nn.Sequential(
                        block(self.in_dim,32)
                    )
        self.globa = torch.nn.Sequential(
                        block(32,128),
                        nn.AdaptiveMaxPool2d(1)
                    )
        self.MLP = torch.nn.Sequential(
                        block(128+32,64),
                        block_no_activation(64,self.out)
                      )
    def forward(self,x):
        '''
        x    : batch size x window dim x num clients x 1
        '''
        out= self.forward_n(x,self.n)
        return out
#         x=x.view(-1,self.in_dim,self.n,1)

#         x = self.local(x)
#         x_local=x
#         x = self.globa(x)
#         x_globa=x.repeat(1,1,self.n,1)

#         x=torch.cat([x_local,x_globa],dim=1)
#         x=self.MLP(x)
# #         x=x.squeeze()
#         out=x
#         return out
    def forward_n(self,x,n):
        x=x.view(-1,self.in_dim,n,1)

        x = self.local(x)
        x_local=x
        x = self.globa(x)
        x_globa=x.repeat(1,1,n,1)

        x=torch.cat([x_local,x_globa],dim=1)
        x=self.MLP(x)
#         x=x.squeeze()
        out=x
        return out
    
class Net(nn.Module):
    def __init__(self,in_dim, n):
        '''
        in_dim:=dimension of weight vector
        n:= number of clients
        '''
        super(Net, self).__init__()
        self.in_dim = in_dim
        self.n = n
        self.main = torch.nn.Sequential(
                        pointNetBlock(in_dim,n,4),
                        nn.ReLU(),
                        pointNetBlock(4,n,1)        
        
                        )


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
        median=torch.median(x,dim=2)[0]
        x=x-median[:,:,None,:]
        
        x=self.main(x)

#        x = F.softmax(x,dim=1)
        x=x.squeeze()
        x = torch.sigmoid(x)
#         pred=dot_product(input,x).squeeze(-1)
        x2= F.softmax(x,dim=1)
        x3 = (x>0.5).float().to(input)
        x3 = x3/(torch.sum(x3,-1).view(-1,1)+1e-14)
        pred_softmax = torch.sum(x2.view(-1,1,self.n)*input,dim=-1).unsqueeze(-1)
        pred_binary = torch.sum(x3.view(-1,1,self.n)*input,dim=-1).unsqueeze(-1)
        return x, pred_softmax, pred_binary
     
    def forward_n(self, input, n):
        '''
        input: batch size x window dim x n
        '''
        x=input.view(-1,self.in_dim,n,1)
        median=torch.median(x,dim=2)[0]
        x=x-median[:,:,None,:]
        
        x=self.main[0].forward_n(x,n)
        x=self.main[1](x)
        x=self.main[2].forward_n(x,n)
#        x = F.softmax(x,dim=1)
        x=x.squeeze()
        x = torch.sigmoid(x)
#         pred=dot_product(input,x).squeeze(-1)
        x2= F.softmax(x,dim=1)
        x3 = (x>0.5).float().to(input)
        x3 = x3/(torch.sum(x3,-1).view(-1,1)+1e-14)
        pred_softmax = torch.sum(x2.view(-1,1,n)*input,dim=-1).unsqueeze(-1)
        pred_binary = torch.sum(x3.view(-1,1,n)*input,dim=-1).unsqueeze(-1)
        return x, pred_softmax, pred_binary

if __name__ == '__main__':
    vd=1
    nc=10
    bs=64
    net = Net(vd,nc)
    y = net(torch.randn(bs,vd,nc))
    for i in y:
        print(i.size())
