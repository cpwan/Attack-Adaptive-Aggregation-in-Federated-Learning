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
    
class PointNet(nn.Module):
    def __init__(self,in_dim, n):
        '''
        in_dim:=dimension of weight vector
        n:= number of clients
        '''
        super(PointNet, self).__init__()
        self.in_dim = in_dim
        self.n = n
        self.local = torch.nn.Sequential(
                        block(self.in_dim,64),
                        block(64,64),
                        block(64,64)
                    )
        self.globa = torch.nn.Sequential(
                        block(64,128),
                        block(128,1024),
                        nn.AdaptiveMaxPool2d(1)
                    )
        self.direct_out= block(1024,10)
        self.MLP = torch.nn.Sequential(
                        block(1088,512),
                        block(512,256),
                        block(256,128),
                        nn.Dropout(p=0.7, inplace=True),
                        block_no_activation(128,1)
                      )


        

    def forward(self, input):
#         print(input.shape)
#         x = input.view(-1, self.n, self.in_dim)
        x=input.view(-1,self.in_dim,self.n,1)
        for module in self.local:
            x = module(x)
#             print(f'local:\t {x.shape}')
        x_local=x
        for module in self.globa:
            x = module(x)
#             print(f'global:\t {x.shape}')
        x_global=x.repeat(1,1,self.n,1)
#         print(f'tile:\t {x_global.shape}')
        x=torch.cat([x_local,x_global],dim=1)
#         print(x.shape)
        for module in self.MLP:
            x = module(x)
#             print(f'MLP:\t {x.shape}')
#         x=self.direct_out(x)
        x=x.squeeze()
#        x = F.softmax(x,dim=1)
        x = torch.sigmoid(x)
#         pred=dot_product(input,x).squeeze(-1)
        x2= F.softmax(x,dim=1)
        x3 = (x>0.5).float().cuda()
        x3 = x3/(torch.sum(x3,-1).view(-1,1)+1e-14)
        pred_softmax = torch.sum(x2.view(-1,1,self.n)*input,dim=-1).unsqueeze(-1)
        pred_binary = torch.sum(x3.view(-1,1,self.n)*input,dim=-1).unsqueeze(-1)
        return x, pred_softmax, pred_binary
     
    def forward_n(self, input, n):
        self.n=n
        self.forward(input)

