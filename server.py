from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

class Server():
    def __init__(self,model,dataLoader,device):
        self.clients=[]
        self.model=model
        self.dataLoader=dataLoader
        self.device=device
        self.emptyStates=None
        self.init_stateChange()
        self.Delta=None
        self.iter=0
        self.GAR=self.FedAvg
        self.func=torch.mean
        self.isSaveChanges=False
        self.savePath='./AggData'
        
    def init_stateChange(self):
        states=deepcopy(self.model.state_dict())
        for param,values in states.items():
            values*=0
        self.emptyStates=states
    
    def attach(self, c):
        self.clients.append(c)
    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy=100. * correct / len(self.dataLoader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.dataLoader.dataset), accuracy))
        return test_loss,accuracy
        
    def train(self,group):
        selectedClients=[self.clients[i] for i in group]
        for c in selectedClients:
            c.train()
            c.update()
        debug=True
        
        if self.isSaveChanges:
            self.saveChanges(selectedClients)
        
        Delta=self.GAR(selectedClients)
        for param in self.model.state_dict():
            self.model.state_dict()[param]+=Delta[param]
        self.iter+=1
        
    def set_GAR(self,gar):
        if   gar=='fedavg':
            self.GAR=self.FedAvg
        elif gar=='median':
            self.GAR=self.FedMedian
        elif gar=='deepGAR':
            self.GAR=self.deepGAR
        else:
            self.GAR=self.FedFunc
# older code, now have been refactored to wrapping FedFunc
#     def FedAvg(self,clients):
#         Delta=deepcopy(self.emptyStates)
#         for param in Delta:
#             for c in clients:
#                 delta=c.getDelta()
#                 Delta[param]+=delta[param]
#             Delta[param]/=len(clients)
#         return Delta
#     def FedMedian(self,clients):
#         Delta=deepcopy(self.emptyStates)
#         deltas=[c.getDelta() for c in clients]
        
#         for param in Delta:
#             ##stacking the weight in the innerest dimension
#             param_stack=torch.stack([delta[param] for delta in deltas],-1)
#             Delta[param]=torch.median(param_stack,-1)[0]
#         return Delta

    def FedAvg(self,clients):
        return self.FedFunc(clients,func=torch.mean)
    def FedMedian(self,clients):
        return self.FedFunc(clients,func=torch.median)
    
    def load_deep_net(self):
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        class Mlp(nn.Module):
            def __init__(self,n,in_dim):
                super(Mlp, self).__init__()
                self.in_dim=in_dim
                self.fc1 = nn.Linear(self.in_dim, n,bias=False)
                self.fc2 = nn.Linear(self.in_dim, 1,bias=False)        
                self.fc3 = nn.Linear(n+1, 1,bias=False)

            def forward(self, x):
                x = x.view(-1, self.in_dim)
                x1 = F.relu(self.fc1(x))
                x2 = F.relu(self.fc2(x))

                x = (self.fc3(torch.cat([x1,x2],1)))
                return x   
        net=Mlp(10,10)
        net.load_state_dict(torch.load('deepGAR.pt'))
        return net
    
    def deepGAR(self,clients):

        net=self.load_deep_net().cuda()
        def func(arr):
            arr=torch.sort(arr)[0]
            with torch.no_grad():
                out=net(arr.cuda()).squeeze()
            return out
        return self.FedFuncPerLayer(clients,func=func)

    def FedFunc(self,clients,func=torch.mean):
        '''
        apply func to each paramters across clients
        '''
        Delta=deepcopy(self.emptyStates)
        deltas=[c.getDelta() for c in clients]

        for param in Delta:

            ##stacking the weight in the innerest dimension
            param_stack=torch.stack([delta[param] for delta in deltas],-1)
            shaped=param_stack.view(-1,len(clients))
            ##applying `func` to every array (of size `num_clients`) in the innerest dimension
            buffer=torch.stack(list(map(func,[shaped[i] for i in range(shaped.size(0))]))).reshape(Delta[param].shape)
            Delta[param]=buffer
        return Delta
    
    def saveChanges(self, clients):
        
        Delta=deepcopy(self.emptyStates)
        deltas=[c.getDelta() for c in clients]

        for param in Delta:

            ##stacking the weight in the innerest dimension
            param_stack=torch.stack([delta[param] for delta in deltas],-1)
            shaped=param_stack.view(-1,len(clients))
            Delta[param]=shaped
        savepath=f'{self.savePath}/{self.GAR.__name__}_{self.iter}.pt'
        
        torch.save(Delta,savepath)
        print(f'Weight delta has been saved to {savepath}')
    def FedFuncPerLayer(self,clients,func=torch.mean):
        '''
        apply func to each layer across clients
        '''
        Delta=deepcopy(self.emptyStates)
        deltas=[c.getDelta() for c in clients]

        for param in Delta:

            ##stacking the weight in the innerest dimension
            param_stack=torch.stack([delta[param] for delta in deltas],-1)
            shaped=param_stack.view(-1,len(clients))
            ##applying `func` to the [n by params] tensor in the innerest dimension
            buffer=func(shaped).reshape(Delta[param].shape)
            Delta[param]=buffer
        return Delta
    