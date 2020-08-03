import torch
import torch.nn as nn
import torch.nn.functional as F

class instanceNormalization(nn.Module):
    def __init__(self,along_dim):
        super(instanceNormalization, self).__init__()
        self.along_dim=along_dim
    def forward(self,x):
        mu=torch.mean(x,self.along_dim,keepdim=True)
        var=torch.var(x,self.along_dim,keepdim=True)
        out = (x-mu)/torch.sqrt(var+1e-16)        
        return out

class nblock(nn.Module):
    def __init__(self,in_,out_):
        super(nblock, self).__init__()
        self.main=torch.nn.Sequential(
                    nn.Conv2d(in_, out_, kernel_size=1),
                    instanceNormalization(along_dim=2),
                    nn.LeakyReLU(),
                    )
    def forward(self,x):
        out=self.main(x)
        return out
    
class nblock_no_activation(nn.Module):
    def __init__(self,in_,out_):
        super(nblock_no_activation, self).__init__()
        self.main=torch.nn.Sequential(
                    nn.Conv2d(in_, out_, kernel_size=1),
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
        self.in_dim = in_dim
        self.n = n
        self.main=torch.nn.Sequential(
                    nblock(self.in_dim,64),
                    nblock(64,128),
                    nblock(128,1024),
                    nblock(1024,512),
                    nblock(512,256),
                    nblock_no_activation(256,1) 
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
#         sorted_x=torch.sort(x,dim=2)[0]
        
#         Q1=int(self.n*0.25+0.5)
#         Q3=int(self.n*0.75+0.5)
#         IQR=sorted_x[:,:,Q3,:]-sorted_x[:,:,Q1,:]+1e-16
#         median=torch.median(x,dim=2)[0]
#         x=(x-median[:,:,None,:])/IQR[:,:,None,:]

        median=torch.median(x,dim=2)[0]
        x=(x-median[:,:,None,:])

        
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
        self.n=n
        self.forward(input)

if __name__ == '__main__':
    vd=1
    nc=10
    bs=64
    net = Net(vd,nc)
    print('Input size:\t', torch.randn(bs,vd,nc).shape)
    y = net(torch.randn(bs,vd,nc))
    for i in y:
        print(i.size())
