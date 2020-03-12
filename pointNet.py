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
        self.direct_out= block(1024,n) #No mlp after concatenation 
        self.MLP = torch.nn.Sequential(
                        block(1088,512),
                        block(512,256),
                        block(256,128),
#                         nn.Dropout(p=0.7, inplace=True),
                        block_no_activation(128,1)
                      )


        

    def forward(self, input):
        #batchsize x vector dimension x num clients x 1
        x=input.view(-1,self.in_dim,self.n,1)
        #Local modules
        for module in self.local:
            x = module(x)
        x_local=x
        #Global modules
        for module in self.globa:
            x = module(x)
        x_global=x.repeat(1,1,self.n,1)
        #Integrate to a MLP
        x=torch.cat([x_local,x_global],dim=1)
        for module in self.MLP:
            x = module(x)
        x=x.squeeze()
        
        #Convert to probability
        x = torch.sigmoid(x)
        x2= F.softmax(x,dim=1)
        x3 = (x>0.5).float().cuda()
        x3 = x3/(torch.sum(x3,-1).view(-1,1)+1e-14) #if all clients are with low likelihood, be conservative to changes=> set the result to 0
        pred_softmax = torch.sum(x2.view(-1,1,self.n)*input,dim=-1).unsqueeze(-1)
        pred_binary = torch.sum(x3.view(-1,1,self.n)*input,dim=-1).unsqueeze(-1)
        return x, pred_softmax, pred_binary
     
    def forward_n(self, input, n):
        self.n=n
        self.forward(input)

