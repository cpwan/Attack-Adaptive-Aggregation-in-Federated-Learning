from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from backdoor_utils import Backdoor_Utils
path_to_aggNet="./aggNet/aggNet_dim64_19.pt"

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
    
    def test_backdoor(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        utils=Backdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1, backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy=100. * correct / len(self.dataLoader.dataset)

        print('\nTest set (Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.format(
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
    def worker(self,c):
        c.train()
        c.update()
        return c
    def train_concurrent(self,group):
        import torch.multiprocessing as mp
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        selectedClients=[self.clients[i] for i in group]
        pool=mp.Pool(5)               
        selectedClients=pool.map(self.worker,selectedClients)
        pool.close()
        pool.join()
        
        
        j=0
        for i in group:
            self.clients[i]=selectedClients[j]
            j+=1
        
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
        elif gar=='deepGARNbh':
            self.GAR=self.deepGARNbh         
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    def FedAvg(self,clients):
        return self.FedFunc(clients,func=torch.mean)
    def FedMedian(self,clients):
        return self.FedFunc(clients,func=torch.median)
    
    def load_deep_net(self):
        from pointNet import PointNet
        num_clients=len(self.clients)
        net=PointNet(1,num_clients)
        net.load_state_dict(torch.load('./aggNet/pointNet_199.pt'))
        return net
    
    def load_deep_net_nbh(self):
#         from pointNet import PointNet
#         num_clients=len(self.clients)
#         self.vector_dimension=64
#         net=PointNet(self.vector_dimension,num_clients)
#         net.load_state_dict(torch.load('./aggNet/aggNet_dim64_19.pt'))
#         return net
        from aggNet import Net
        num_clients=len(self.clients)
        self.vector_dimension=64
        net=Net(self.vector_dimension,num_clients)
        net.load_state_dict(torch.load(path_to_aggNet))
        return net
    
    def deepGAR(self,clients):

        net=self.load_deep_net().cuda()
        def func(arr):
#             arr=torch.sort(arr)[0]
            with torch.no_grad():
                out=net(arr.cuda())[2].squeeze()
            return out
        return self.FedFuncPerLayer(clients,func=func)
    
    def deepGARNbh(self,clients):

        net=self.load_deep_net_nbh().cuda()
        def func(arr):
#             arr=torch.sort(arr)[0]
            with torch.no_grad():
                out=net(arr.cuda())[2][:,0].squeeze()
            return out
        return self.FedFuncNbhPerLayer(clients,func=func,vd=self.vector_dimension)

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
    def FedFuncNbhPerLayer(self,clients,func=torch.mean,vd=1):
        '''
        apply func to each layer across clients
        for each entry in the layer, sample (vd-1) other entries in the layer with that entry,
        feed it to the GAR
        '''
        Delta=deepcopy(self.emptyStates)
        deltas=[c.getDelta() for c in clients]
        
        def getNbh(t1,vd):
            '''
            -in
            t1: tensor with shape d1 x 1 x num clients
            vd: the dimension of the vector to be fed to deep GAR, equals to the number of nbh  to be sampled
            -out
            entriesWithNbh: tensor with shape d1 x vd x num clients, with entriesWithNbh[:,0,:] being the original entry 
            '''
            d1=t1.size(0)
            num_clients=t1.size(2)
            randperms=[torch.tensor(range(d1))]+[torch.randperm(t1.size(0)) for i in range(vd-1)]
            randperms_index=torch.stack(randperms,dim=1)
            entriesWithNbh=t1[randperms_index].view(-1,vd,num_clients)
            return entriesWithNbh
            
        for param in Delta:
            
            ##stacking the weight in the innerest dimension
            ## size of layer x1x number of clients
            param_stack=torch.stack([delta[param] for delta in deltas],-1) # d1 x d2 x d3 x... xnum clients
            shaped=param_stack.view(-1,1,len(clients)) #d1*d2*d3*... x 1 x num clients
            dset=torch.utils.data.TensorDataset(getNbh(shaped,vd))
            dloader=torch.utils.data.DataLoader(dset,batch_size=25000)
            result=[]
            for data in dloader:
                result.append(func(data[0]))
            result_tensor=torch.stack(result)
            buffer=result_tensor.reshape(Delta[param].shape)
            ##applying `func` to the [n by params] tensor in the innerest dimension
#             buffer=func(shaped).reshape(Delta[param].shape)
            Delta[param]=buffer
        return Delta
        
    def FedFuncPerLayer(self,clients,func=torch.mean):
        '''
        apply func to each layer across clients
        '''
        Delta=deepcopy(self.emptyStates)
        deltas=[c.getDelta() for c in clients]

        for param in Delta:
            
            ##stacking the weight in the innerest dimension
            ## size of layer x1x number of clients
            param_stack=torch.stack([delta[param].cpu() for delta in deltas],-1) # d1 x d2 x d3 x... xnum clients
            shaped=param_stack.view(-1,1,len(clients)) #d1*d2*d3*... x 1 x num clients
            dset=torch.utils.data.TensorDataset(shaped)
            dloader=torch.utils.data.DataLoader(dset,batch_size=8192)
            result=[]
            for data in dloader:
                result.append(func(data[0]))
            result_tensor=torch.stack(result)
            buffer=result_tensor.reshape(Delta[param].shape)
            ##applying `func` to the [n by params] tensor in the innerest dimension
#             buffer=func(shaped).reshape(Delta[param].shape)
            Delta[param]=buffer
        return Delta
    