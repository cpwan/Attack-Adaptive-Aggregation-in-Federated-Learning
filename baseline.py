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
        self.in_dim = in_dim
        self.n = n
 
        self.local = torch.nn.Sequential(
                        block(self.in_dim,64),
#                         block(64,64),
                    )
        self.MLP = torch.nn.Sequential(
                        block_no_activation(64,1)
                      )

        

    def forward(self, input):
#         print(input.shape)
        '''
        input: batch size x window dim x num clients
        '''
        sorted_input=torch.sort(input,dim=2)
        
        
        assert input.shape[1]==1, "requires window size to be 1"
        x_sorted=sorted_input[0]
        sorted_index=sorted_input[1].squeeze()
        revert_index=torch.argsort(sorted_index,dim=1)
        assert torch.equal(torch.gather(x_sorted.squeeze(), 1, revert_index),input.squeeze()), "revert indexing failed"
        x=x_sorted.view(-1,self.in_dim,self.n,1)
        
        
        '''
        x    : batch size x window dim x num clients x 1
        '''
        median=torch.median(x,dim=2)[0]
        x=x-median[:,:,None,:]#sort along clients
        for module in self.local:
            x = module(x)

        for module in self.MLP:
            x = module(x)
        x=x.squeeze()
        
#        x = F.softmax(x,dim=1)
        x = torch.sigmoid(x)
        
        
        
#         pred=dot_product(input,x).squeeze(-1)
        x2= F.softmax(x,dim=1)
        x3 = (x>0.5).float().to(x_sorted)
        x3 = x3/(torch.sum(x3,-1).view(-1,1)+1e-14)
        pred_softmax = torch.sum(x2.view(-1,1,self.n)*x_sorted,dim=-1).unsqueeze(-1)
        pred_binary = torch.sum(x3.view(-1,1,self.n)*x_sorted,dim=-1).unsqueeze(-1)

        x = torch.gather(x,1, revert_index) #revert indexing
        return x, pred_softmax, pred_binary
     
    def forward_n(self, input, n):
        self.n=n
        self.forward(input)

if __name__ == '__main__':

    net = Net(1,10)
    y = net((torch.randn(100,1,10)))
    for item in y:
        print(item.size())
